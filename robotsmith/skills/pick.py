"""``pick`` primitive: select an executable grasp and lift the object.

Each candidate's whole trajectory goes through the execution gate first; one that passes is tried in
simulation (``scene_state["grasp_trial"]``) and, if it lifts the target, verified again on the restored
scene and executed. Without a trial callback the first candidate that passes the gate is used.
"""

from __future__ import annotations

import numpy as np

from robotsmith.execution.gate import (ExecutionFailed, Holding, MotionRejected, Refused, Span, VerifiedMotion,
                                       poses_of, spans_of)
from robotsmith.grasp.planner import GraspPlan
from robotsmith.grasp.transforms import pose_matrix
from robotsmith.motion.params import MotionParams
from robotsmith.skills.base import SkillContext
from robotsmith.skills.registry import bound_leaf, register


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


def _pick_seed_transit(
    ctx: SkillContext, plan: GraspPlan
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Reset to home + pre-aim the base joint at the grasp azimuth, with transit.

    Two failure modes this guards against:
      1. After an articulated drag (open/close) the arm is left in an extended
         pose; descending IK to a tabletop grasp gets stuck there.
      2. From the y=0 neutral home, Genesis IK under-rotates the base joint and
         lands short of off-axis (large |y|) targets, closing beside the object.
    The transit is a no-op when the arm already starts at this seed.
    """
    params = ctx.params
    home_qpos = ctx.scene_state.get("home_qpos")
    q0 = np.asarray(ctx.qpos, dtype=np.float64)
    seed = (
        np.asarray(home_qpos, dtype=np.float64).copy()
        if home_qpos is not None
        else q0.copy()
    )
    seed[0] = float(np.arctan2(float(plan.grasp_pos[1]), float(plan.grasp_pos[0])))
    transit: list[np.ndarray] = []
    if not np.allclose(q0[:7], seed[:7], atol=1e-3):
        steps = max(params.approach_steps, params.min_steps)
        transit = [q0 + (seed - q0) * (i + 1) / steps for i in range(steps)]
    return seed, transit


def _select(ctx: SkillContext, trial) -> tuple[GraspPlan, VerifiedMotion]:
    """Gate, then trial, the candidates in planner order from the same start state; keep the first that passes
    both, verified again after the last trial. When none does, raises ``Refused("search_exhausted")`` if no trial
    ran, else ``ExecutionFailed`` with the trials' physics steps.

    ``trial(trajectory, target)`` replays the trajectory in simulation, restores the scene and returns
    ``{"lifted", "lift_m", "seconds"}``. With ``scene_state["grasp_trial_all"]`` every candidate is
    tried, for probes that need the executable fraction.
    """
    plans, filter_counts = _candidates(ctx)
    try_all = bool(ctx.scene_state.get("grasp_trial_all"))
    target = ctx.skill.target
    trials, passing = [], []
    for plan in plans:
        md = plan.metadata
        record = {"candidate_index": md.get("candidate_index"), "kind": md.get("kind"),
                  "closing_width": md.get("closing_width"), "lifted": False, "lift_m": None, "seconds": 0.0}
        try:
            request = _pick_request(ctx, plan)
            motion = ctx.gate.verify(**request)
            if trial is not None:
                record.update(trial(motion.trajectory, target), steps=len(motion.trajectory))
                if record["lifted"]:
                    motion = ctx.gate.verify(**request)
        except MotionRejected as exc:
            record["gate"] = exc.to_dict()
        except RuntimeError as exc:
            record["motion_error"] = str(exc)
        else:
            record["gate"] = "passed"
            if trial is None or record["lifted"]:
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


def _pick_request(ctx: SkillContext, plan: GraspPlan) -> dict:
    """``gate.verify`` keywords for ``plan``: base-aim transit plus the executor's pick segment, its spans and
    waypoint poses, and the target held at the candidate's grasp pose."""
    seed, transit = _pick_seed_transit(ctx, plan)
    result = ctx.executor.pick(plan, ctx.motion_planner, seed, _pick_motion_params(ctx))
    target = ctx.skill.target
    t_hand_object = (np.linalg.inv(pose_matrix(plan.grasp_pos, plan.grasp_quat))
                     @ pose_matrix(*ctx.gate.env.object_pose(target)))
    return {"trajectory": transit + result.trajectory,
            "spans": [Span("transit", 0, len(transit))] + spans_of(result.segments, len(transit)),
            "poses": poses_of(result.segments, len(transit)), "target": target, "q_start": ctx.qpos,
            "holding": Holding(target, t_hand_object)}


@register("pick", post=bound_leaf)
def run_pick(ctx: SkillContext) -> VerifiedMotion:
    plan, motion = _select(ctx, ctx.scene_state.get("grasp_trial"))
    ctx.held_category = ctx.category
    # Grasp-relative offset: EE-flange height above the picked object's *support
    # surface* (object bottom = center − half-height), NOT above the table. A
    # later place re-adds this to the destination marker's world z so the object
    # bottom lands on that surface — correct for table→table AND shelf→shelf.
    # For a tabletop pick this equals the old (grasp_z − table_z).
    obj_h = float(ctx.scene_state.get("object_heights", {}).get(ctx.skill.target, 0.0) or 0.0)
    support_z = float(ctx.obj_pos[2]) - obj_h / 2.0
    ctx.held_place_z = max(float(plan.grasp_pos[2]) - support_z, 0.0)
    return motion
