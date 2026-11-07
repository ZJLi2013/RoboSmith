"""``pick`` primitive: select an executable grasp and lift the object.

The approach to each candidate's first waypoint is a ``JointPlanner`` connection; the descend, grasp and lift are
the executor's segment from there. Each candidate's whole trajectory goes through the execution gate first, and a
following single-phase place is planned from where it ends; one that passes both is tried in
simulation (``scene_state["grasp_trial"]``) and, if it lifts the target, verified again on the restored
scene and executed. Without a trial callback the first candidate that passes the gate is used.
"""

from __future__ import annotations

import numpy as np

from robotsmith.execution.gate import (ExecutionFailed, MotionRejected, Refused, Span, VerifiedMotion, poses_of,
                                       spans_of)
from robotsmith.grasp.planner import GraspPlan
from robotsmith.grasp.transforms import pose_matrix
from robotsmith.motion.joint_planner import JointPlanner
from robotsmith.motion.params import MotionParams
from robotsmith.motion.plan_state import binding
from robotsmith.skills.base import Skill, SkillContext
from robotsmith.skills.registry import bound_leaf, plan_segment, register, skill_phases


def _candidates(ctx: SkillContext) -> tuple[list[GraspPlan], dict]:
    """The planner's grasp candidates for the target, and how many each of its filters removed."""
    skill = ctx.skill
    scene_state = ctx.scene_state
    obj_h = (
        skill.params.get("object_height")
        or scene_state.get("object_heights", {}).get(skill.target)
    )
    plans = ctx.planner.plan(
        ctx.obj_pos,
        object_quat=scene_state.get("object_quats", {}).get(skill.target),
        category=ctx.category,
        asset=scene_state.get("assets", {}).get(skill.target),
        object_height=obj_h,
    )
    return plans, dict(getattr(ctx.planner, "last_filter_counts", {}) or {})


def _pick_motion_params(ctx: SkillContext) -> MotionParams:
    """Pick timing = base params, but skip the lift hold when a place follows."""
    params = ctx.params
    next_is_place = (
        ctx.index + 1 < len(ctx.skills) and ctx.skills[ctx.index + 1].name == "place"
    )
    return MotionParams(
        max_joint_vel=params.max_joint_vel,
        fine_joint_vel=params.fine_joint_vel,
        dt=params.dt,
        min_steps=params.min_steps,
        finger_steps=params.finger_steps,
        grasp_settle_steps=params.grasp_settle_steps,
        approach_steps=params.approach_steps,
        descend_steps=params.descend_steps,
        grasp_hold_steps=params.grasp_hold_steps,
        lift_steps=params.lift_steps,
        lift_hold_steps=0 if next_is_place else params.lift_hold_steps,
        transport_steps=params.transport_steps,
        place_descend_steps=params.place_descend_steps,
        release_steps=params.release_steps,
        retreat_steps=params.retreat_steps,
    )


def _ik_init(ctx: SkillContext, plan: GraspPlan) -> np.ndarray:
    """IK start for the candidate's first waypoint: home with the base joint aimed at the grasp azimuth, or the
    candidate's ``ik_seed``. From an extended pose (after a drag) IK gets stuck, and from the y=0 home Genesis IK
    under-rotates the base and lands short of off-axis targets."""
    home_qpos = ctx.scene_state.get("home_qpos")
    init = np.asarray(home_qpos if home_qpos is not None else ctx.qpos, dtype=np.float64).copy()
    init[0] = float(np.arctan2(float(plan.grasp_pos[1]), float(plan.grasp_pos[0])))
    if plan.metadata.get("ik_seed"):
        init[:7] = plan.metadata["ik_seed"]
    return init


def _select(ctx: SkillContext, trial) -> tuple[GraspPlan, VerifiedMotion]:
    """Gate, then trial, the candidates in planner order from the same start state; keep the first that passes
    both, verified again after the last trial. When the next skill is a single-phase place, a candidate whose place
    is refused from where its pick ends is not tried. When none passes, raises ``Refused("search_exhausted")`` if no
    trial ran, else ``ExecutionFailed`` with the trials' physics steps.

    ``trial(trajectory, target)`` replays the trajectory in simulation, restores the scene and returns
    ``{"lifted", "lift_m", "seconds"}``. With ``scene_state["grasp_trial_all"]`` every candidate is
    tried, for probes that need the executable fraction.
    """
    plans, filter_counts = _candidates(ctx)
    try_all = bool(ctx.scene_state.get("grasp_trial_all"))
    target = ctx.skill.target
    place = _static_place(ctx)
    trials, passing = [], []
    for plan in plans:
        md = plan.metadata
        record = {"candidate_index": md.get("candidate_index"), "kind": md.get("kind"),
                  "closing_width": md.get("closing_width"), "lifted": False, "lift_m": None, "seconds": 0.0}
        try:
            request, record["approach"] = _pick_request(ctx, plan)
            motion = ctx.gate.verify(**request)
            placeable = place is None or _place_check(ctx, plan, request, place, record)
            if placeable and trial is not None:
                record.update(trial(motion.trajectory, target), steps=len(motion.trajectory))
                if record["lifted"]:
                    motion = ctx.gate.verify(**request)
        except MotionRejected as exc:
            record["gate"] = exc.to_dict()
        except Refused as exc:
            record["approach"] = {"ok": False, **exc.evidence}
        except RuntimeError as exc:
            record["motion_error"] = str(exc)
        else:
            record["gate"] = "passed"
            if placeable and (trial is None or record["lifted"]):
                passing.append((plan, request, motion, record))
        trials.append(record)
        if passing and not try_all:
            break
    chosen = None
    for plan, request, motion, record in passing:
        if try_all:
            # Every later trial stepped physics and voided this motion's voucher.
            try:
                motion = ctx.gate.verify(**request)
            except MotionRejected as exc:
                record["reverify"] = exc.to_dict()
                continue
        chosen = (plan, motion)
        break
    ctx.trace({
        "skill": ctx.skill.name, "target": target, "category": ctx.category,
        "filter_counts": filter_counts, "no_candidates": not plans, "no_trial_success": chosen is None,
        "selected_candidate_index": chosen[0].metadata.get("candidate_index") if chosen else None,
        "trials": trials, "trial_seconds": round(sum(t["seconds"] for t in trials), 3),
    })
    if chosen is None:
        evidence = {"reason": "no pick candidate passed the gate and its trial", "candidates": len(plans),
                    "filter_counts": filter_counts, "trials": trials}
        steps = sum(t.get("steps", 0) for t in trials)
        if steps:
            raise ExecutionFailed(evidence, steps)
        raise Refused("search_exhausted", evidence)
    return chosen


def _pick_request(ctx: SkillContext, plan: GraspPlan) -> tuple[dict, dict]:
    """``gate.verify`` keywords for ``plan``, and the approach's record: a ``JointPlanner`` approach from the
    current joints to the candidate's first waypoint, the executor's pick segment from there, their spans and
    waypoint poses, and the target held at the candidate's grasp pose. The pick segment passes the gate on its own
    before the approach is searched, so a candidate the gate rejects costs no search."""
    q0 = np.asarray(ctx.qpos, dtype=np.float64)
    first = plan.waypoints[0]
    q_pre = np.asarray(ctx.motion_planner.solve_ik(first.pos, first.quat, first.finger_width,
                                                   init_qpos=_ik_init(ctx, plan)), dtype=np.float64)
    q_pre[7:] = q0[7:]
    target = ctx.skill.target
    holding = binding(target, (plan.grasp_pos, plan.grasp_quat), ctx.gate.env.object_pose(target))
    result = ctx.executor.pick(plan, ctx.motion_planner, q_pre, _pick_motion_params(ctx), grasp=holding)
    ctx.gate.verify(result.trajectory, spans_of(result.segments), target=target, q_start=q_pre, holding=holding,
                    poses=poses_of(result.segments))
    c = JointPlanner(ctx.gate.env, ctx.gate).connect(q0, q_pre, target=target, label="approach")
    approach = list(c.motion.trajectory)
    n = len(approach)
    return ({"trajectory": approach + result.trajectory,
             "spans": [Span("approach", 0, n)] + spans_of(result.segments, n),
             "poses": poses_of(result.segments, n), "target": target, "q_start": ctx.qpos, "holding": holding},
            {"ok": True, "queries": c.queries, "seconds": c.seconds, "nodes": c.nodes, "direct": c.direct})


def _static_place(ctx: SkillContext) -> Skill | None:
    """The next skill when it is a place onto a static anchor, planned in one phase; else ``None``."""
    if ctx.index + 1 >= len(ctx.skills):
        return None
    nxt = ctx.skills[ctx.index + 1]
    frame = ctx.scene_state.get("target_frames", {}).get(nxt.target)
    return nxt if nxt.name == "place" and skill_phases(nxt, frame) == ["all"] else None


def _place_check(ctx: SkillContext, plan: GraspPlan, request: dict, place: Skill, record: dict) -> bool:
    """Plan ``place`` with the arm at the end of this candidate's pick and the target at its grasp in the hand, and
    record the outcome as ``record["place"]``. The scene is unchanged on return, the planned motion is discarded and
    its trace does not reach the run's ``skill_traces``."""
    env = ctx.gate.env
    q_end = np.asarray(request["trajectory"][-1], dtype=np.float64)
    target = ctx.skill.target
    held_at = pose_matrix(*env.hand_pose(q_end)) @ request["holding"].t_hand_object
    with env.posed(q_end, {target: held_at}):
        try:
            plan_segment(place, ctx.index + 1, ctx.skills, ctx.planner, ctx.executor, None,
                         {**ctx.scene_state, "skill_traces": []}, ctx.params, q_end, ctx.category,
                         _held_place_z(ctx, plan), ctx.motion_planner, gate=ctx.gate, held=target)
        except Refused as exc:
            record["place"] = {"ok": False, "label": exc.label, **exc.evidence}
            return False
    record["place"] = {"ok": True}
    return True


def _held_place_z(ctx: SkillContext, plan: GraspPlan) -> float:
    # Grasp-relative offset: EE-flange height above the picked object's *support
    # surface* (object bottom = center − half-height), NOT above the table. A
    # later place re-adds this to the destination marker's world z so the object
    # bottom lands on that surface — correct for table→table AND shelf→shelf.
    # For a tabletop pick this equals the old (grasp_z − table_z).
    obj_h = float(ctx.scene_state.get("object_heights", {}).get(ctx.skill.target, 0.0) or 0.0)
    support_z = float(ctx.obj_pos[2]) - obj_h / 2.0
    return max(float(plan.grasp_pos[2]) - support_z, 0.0)


@register("pick", post=bound_leaf)
def run_pick(ctx: SkillContext) -> VerifiedMotion:
    plan, motion = _select(ctx, ctx.scene_state.get("grasp_trial"))
    ctx.held_category = ctx.category
    ctx.held_place_z = _held_place_z(ctx, plan)
    return motion
