"""The pick and place tools a policy calls; each method runs one explicit request and decides nothing.

``ManipulationTools`` is the one caller, for pick and place, of the grasp planner, the execution gate, the
``JointPlanner`` and ``trial_from_here``. A pick is the executor's pick list for one candidate: open the fingers, a
``JointPlanner`` approach to its pre-grasp, the straight descend, the close and the straight lift. A place is a
``JointPlanner`` transport with the held object to the standoff above the target, then the executor's straight
descend, release and retreat; the whole trajectory passes the gate once. Which candidate to try, whether to check
the place first and when to verify again belong to the policy (``rule_policy.RulePolicy``).
"""

from __future__ import annotations

import numpy as np

from robotsmith.execution.gate import Holding, MotionRejected, Refused, Span, VerifiedMotion, poses_of, spans_of
from robotsmith.execution.meta_actions import pick_actions, place_actions
from robotsmith.grasp.planner import GraspPlan
from robotsmith.motion import gripper
from robotsmith.motion.joint_planner import JointPlanner
from robotsmith.motion.params import MotionParams
from robotsmith.motion.plan_state import PlanState, binding
from robotsmith.skills.base import SkillContext
from robotsmith.skills.frames import _float_or_none, resolve_approach

# EE-flange height above the tray-floor world point when placing into an
# articulated opening. The opening anchor resolves to the cavity floor; lift the
# flange by the gripper's reach (finger length + margin) so an empty gripper /
# held payload settles just above the floor instead of being driven through it.
# Robot/gripper property (not asset geometry), so it lives here, not in metadata.
_DRAWER_PLACE_EE_CLEARANCE = 0.10


class ManipulationTools:
    def __init__(self, gate, planner, executor, motion_planner):
        self.env, self.gate, self.planner, self.executor, self.motion_planner = (
            gate.env, gate, planner, executor, motion_planner)

    def grasp_candidates(self, ctx: SkillContext) -> list[GraspPlan]:
        """The planner's grasp candidates for the step's target, in the planner's order."""
        target = ctx.skill.target
        return self.planner.plan(
            ctx.obj_pos,
            object_quat=ctx.scene_state.get("object_quats", {}).get(target),
            category=ctx.category,
            asset=ctx.scene_state.get("assets", {}).get(target),
        )

    def plan_pick(self, ctx: SkillContext, plan: GraspPlan, params: MotionParams) -> tuple[dict, dict]:
        """``gate.verify`` keywords for ``plan``, and the approach's record: the pick list run from the current joints,
        with the fingers opened there and a ``JointPlanner`` approach from the opened joints to the candidate's
        pre-grasp, its spans and waypoint poses, and the target held at the candidate's grasp pose. The list from the
        pre-grasp on passes the gate on its own before the approach is searched, so a candidate the gate rejects costs
        no search."""
        q0 = np.asarray(ctx.qpos, dtype=np.float64)
        env, solve_ik = self.env, self.motion_planner.solve_ik
        hand = env.hand_pose(q0)
        opened = gripper.expand(gripper.Gripper(*hand, plan.finger_open), q0, solve_ik).q_target
        q_pre = np.asarray(solve_ik(plan.pre_grasp_pos, plan.pre_grasp_quat, plan.finger_open,
                                    init_qpos=_ik_init(ctx, plan)), dtype=np.float64)
        q_pre[7:] = opened[7:]
        target = ctx.skill.target
        holding = binding(target, (plan.grasp_pos, plan.grasp_quat), env.object_pose(target))

        def run(start, start_hand, approach):
            actions = pick_actions(ctx.skill.params["actions"], plan, start, start_hand, approach, params, solve_ik,
                                   grasp=holding)
            return self.executor.run("pick", actions, self.motion_planner, PlanState(start), params)

        grasp = run(q_pre, (plan.pre_grasp_pos, plan.pre_grasp_quat), ())
        self.gate.verify(grasp.trajectory, spans_of(grasp.segments), target=target, q_start=q_pre, holding=holding,
                         poses=poses_of(grasp.segments))
        c = JointPlanner(env, self.gate).connect(opened, q_pre, target=target, label="approach")
        result = run(q0, hand, c.motion.trajectory)
        return ({"trajectory": result.trajectory, "spans": spans_of(result.segments),
                 "poses": poses_of(result.segments), "target": target, "q_start": ctx.qpos, "holding": holding},
                {"ok": True, "queries": c.queries, "seconds": c.seconds, "nodes": c.nodes, "direct": c.direct})

    def verify(self, request: dict) -> VerifiedMotion:
        """Issue the voucher for a ``plan_pick`` request against the current scene."""
        return self.gate.verify(**request)

    def trial(self, trajectory, target: str) -> dict:
        """Replay ``trajectory`` in simulation and restore the scene: ``{"lifted", "lift_m", "seconds"}``."""
        return self.env.trial_from_here(trajectory, target)

    def posed(self, qpos, objects: dict):
        """The scene with the arm at ``qpos`` and ``objects`` at their poses, restored on exit."""
        return self.env.posed(qpos, objects)

    def plan_place(self, ctx: SkillContext) -> VerifiedMotion:
        """The place of ``ctx.held`` onto ``ctx.skill.target`` for ``ctx.phase``, verified; ``Refused`` when its
        transport or its motion does not pass. Its trace record goes to ``ctx.trace``.

        Phase split (live re-anchoring): "transport" moves to straight above the target; "descend" re-reads the
        (recoiled) target and goes vertical. "all" is the single pass for static anchors. ``obj_pos`` is already
        re-resolved per phase by the runtime. See ``skill_phases``."""
        place_plan, requested_place_z, resolved_place_z = _resolve_place_plan(ctx, self.planner)
        q0 = np.asarray(ctx.qpos, dtype=np.float64)
        env = self.env
        holding = binding(ctx.held, env.hand_pose(q0), env.object_pose(ctx.held)) if ctx.held else None
        transport, motion, why = None, None, None
        transit, start = [], q0
        if ctx.phase in ("all", "transport"):
            connection, transport = self._connect(ctx, q0, place_plan, holding)
            if connection is not None and len(connection.motion.trajectory):
                transit, start = list(connection.motion.trajectory), connection.motion.trajectory[-1]
        if transport is None or transport["ok"]:
            try:
                if ctx.phase == "transport":
                    segments, tail = [], []
                else:
                    result = self.executor.run("place", place_actions(ctx.skill.params["actions"], place_plan,
                                                                      ctx.params, phase="descend"),
                                               self.motion_planner, PlanState(start, holding), ctx.params)
                    segments, tail = result.segments, result.trajectory
                spans = ([Span("transport", 0, len(transit), holding=holding is not None)]
                         + spans_of(segments, len(transit)))
                motion = self.gate.verify(transit + tail, spans, target=ctx.held, q_start=q0,
                                          rest={ctx.skill.target.split(".")[0]}, holding=holding,
                                          poses=poses_of(segments, len(transit)))
            except MotionRejected as exc:
                why = {"gate": exc.to_dict()}
            except RuntimeError as exc:
                why = {"motion_error": str(exc)}
        ctx.trace(
            {
                "skill": ctx.skill.name,
                "target": ctx.skill.target,
                "phase": ctx.phase,
                "category": ctx.category,
                "requested_place_z": requested_place_z,
                "resolved_place_z": resolved_place_z,
                "place_z_source": (
                    "requested" if requested_place_z is not None else "selected_grasp"
                ),
                "transport": transport,
                "motion_trace": getattr(self.executor, "last_motion_trace", None),
            }
        )
        if motion is None:
            if transport is not None and not transport["ok"]:
                raise Refused("search_exhausted", {"reason": "no transport to the standoff", "transport": transport})
            raise Refused("search_exhausted", {"reason": "the place motion did not pass the gate", **why})
        return motion

    def _connect(self, ctx: SkillContext, q0: np.ndarray, place_plan: GraspPlan, holding: Holding | None):
        """The planned transport from ``q0`` to the standoff, fingers kept where they are, and its trace record."""
        q_pre = np.asarray(self.motion_planner.solve_ik(place_plan.pre_grasp_pos, place_plan.pre_grasp_quat,
                                                        float(q0[7]), init_qpos=q0), dtype=np.float64)
        q_pre[7:] = q0[7:]
        try:
            c = JointPlanner(self.env, self.gate).connect(q0, q_pre, target=ctx.held, holding=holding,
                                                          label="transport")
        except Refused as exc:
            return None, {"ok": False, **exc.evidence}
        return c, {"ok": True, "queries": c.queries, "seconds": c.seconds, "nodes": c.nodes, "direct": c.direct}


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


def _resolve_place_plan(ctx: SkillContext, planner) -> tuple[GraspPlan, float | None, float | None]:
    """Build the place plan + (requested, resolved) place_z for the trace.

    Every place consumes the destination's already-resolved world point
    (``ctx.obj_pos`` = the marker / opening world xyz from the resolver) and
    lifts the EE-flange by a grasp-relative offset; the planner never re-derives
    a table-relative drop height.

    - Articulated opening: marker = live tray-floor; offset = gripper clearance.
    - Marker / tabletop: marker = target world xyz; offset = the held object's
      grasp-relative height (``held_place_z``, flange above the picked support
      surface), so the object bottom lands on the marker surface. A scenario may
      still pass an explicit ``place_z`` to override the offset.
    """
    skill = ctx.skill
    frame = ctx.scene_state.get("target_frames", {}).get(skill.target)
    # World insert direction for the final approach (default top-down -Z); declared
    # on the target's frame / placement affordance and rotated to world. Off-axis
    # (e.g. a side tuck-under shelf) routes the same standoff→insert→retract path
    # along this axis instead of straight down.
    approach = resolve_approach(frame, ctx.scene_state)
    requested_place_z = _float_or_none(skill.params.get("place_z"))
    marker = np.asarray(ctx.obj_pos, dtype=np.float64).copy()
    if getattr(frame, "kind", None) == "articulated_opening":
        marker[2] += _DRAWER_PLACE_EE_CLEARANCE
        place_plan = planner.plan_place(
            ctx.obj_pos, category=ctx.category, place_point_world=marker,
            approach=approach,
        )
        return place_plan, requested_place_z, None
    if requested_place_z is None and ctx.held_place_z is None:
        raise ValueError(
            f"Place skill for {skill.target!r} requires a preceding pick "
            f"or an explicit place_z"
        )
    offset = requested_place_z if requested_place_z is not None else ctx.held_place_z
    marker[2] += float(offset)
    place_plan = planner.plan_place(
        ctx.obj_pos, category=ctx.category, place_point_world=marker,
        approach=approach,
    )
    return place_plan, requested_place_z, float(offset)
