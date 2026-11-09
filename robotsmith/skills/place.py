"""``place`` primitive: transport the held object and release it at the target.

Supports tabletop places (explicit/held ``place_z``) and articulated-opening
places (live tray-floor anchor). The transport to the standoff above the target is a
``JointPlanner`` connection with the held object; the descend, release and retreat are the
executor's straight lines from there, and the whole trajectory passes the execution gate once.
"""

from __future__ import annotations

import numpy as np

from robotsmith.execution.gate import Holding, MotionRejected, Refused, Span, VerifiedMotion, poses_of, spans_of
from robotsmith.execution.meta_actions import place_actions
from robotsmith.grasp.planner import GraspPlan
from robotsmith.motion.joint_planner import JointPlanner
from robotsmith.motion.plan_state import PlanState, binding
from robotsmith.skills.base import SkillContext
from robotsmith.skills.frames import (
    _float_or_none,
    resolve_approach,
)
from robotsmith.skills.registry import bound_leaf, register

# EE-flange height above the tray-floor world point when placing into an
# articulated opening. The opening anchor resolves to the cavity floor; lift the
# flange by the gripper's reach (finger length + margin) so an empty gripper /
# held payload settles just above the floor instead of being driven through it.
# Robot/gripper property (not asset geometry), so it lives here, not in metadata.
_DRAWER_PLACE_EE_CLEARANCE = 0.10


def _resolve_place_plan(
    ctx: SkillContext,
) -> tuple[GraspPlan, float | None, float | None]:
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
        place_plan = ctx.planner.plan_place(
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
    place_plan = ctx.planner.plan_place(
        ctx.obj_pos, category=ctx.category, place_point_world=marker,
        approach=approach,
    )
    return place_plan, requested_place_z, float(offset)


def _connect(ctx: SkillContext, q0: np.ndarray, place_plan: GraspPlan, holding: Holding | None):
    """The planned transport from ``q0`` to the standoff, fingers kept where they are, and its trace record."""
    q_pre = np.asarray(ctx.motion_planner.solve_ik(place_plan.pre_grasp_pos, place_plan.pre_grasp_quat,
                                                   float(q0[7]), init_qpos=q0), dtype=np.float64)
    q_pre[7:] = q0[7:]
    try:
        c = JointPlanner(ctx.gate.env, ctx.gate).connect(q0, q_pre, target=ctx.held, holding=holding,
                                                         label="transport")
    except Refused as exc:
        return None, {"ok": False, **exc.evidence}
    return c, {"ok": True, "queries": c.queries, "seconds": c.seconds, "nodes": c.nodes, "direct": c.direct}


@register("place", post=bound_leaf, settle_s=0.5)
def run_place(ctx: SkillContext) -> VerifiedMotion:
    # Phase split (live re-anchoring): "transport" moves to straight above the
    # target; "descend" re-reads the (recoiled) target and goes vertical. "all" is
    # the single pass for static anchors. ``obj_pos`` is already
    # re-resolved per phase by the runtime. See ``skill_phases``.
    place_plan, requested_place_z, resolved_place_z = _resolve_place_plan(ctx)
    q0 = np.asarray(ctx.qpos, dtype=np.float64)
    env = ctx.gate.env
    holding = binding(ctx.held, env.hand_pose(q0), env.object_pose(ctx.held)) if ctx.held else None
    transport, motion, why = None, None, None
    transit, start = [], q0
    if ctx.phase in ("all", "transport"):
        connection, transport = _connect(ctx, q0, place_plan, holding)
        if connection is not None and len(connection.motion.trajectory):
            transit, start = list(connection.motion.trajectory), connection.motion.trajectory[-1]
    if transport is None or transport["ok"]:
        try:
            if ctx.phase == "transport":
                segments, tail = [], []
            else:
                result = ctx.executor.run("place", place_actions(ctx.skill.params["actions"], place_plan,
                                                                 ctx.params, phase="descend"),
                                          ctx.motion_planner, PlanState(start, holding), ctx.params)
                segments, tail = result.segments, result.trajectory
            spans = ([Span("transport", 0, len(transit), holding=holding is not None)]
                     + spans_of(segments, len(transit)))
            motion = ctx.gate.verify(transit + tail, spans, target=ctx.held, q_start=q0,
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
            "motion_trace": getattr(ctx.executor, "last_motion_trace", None),
        }
    )
    if motion is None:
        if transport is not None and not transport["ok"]:
            raise Refused("search_exhausted", {"reason": "no transport to the standoff", "transport": transport})
        raise Refused("search_exhausted", {"reason": "the place motion did not pass the gate", **why})
    # The object is only released on the descend phase; transport still carries it,
    # so hold-state carry-over must survive into descend.
    if ctx.phase in ("all", "descend"):
        ctx.held_category = None
        ctx.held_place_z = None
    return motion
