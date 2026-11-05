"""``place`` primitive: transport the held object and release it at the target.

Supports tabletop places (explicit/held ``place_z``) and articulated-opening
places (live tray-floor anchor).
"""

from __future__ import annotations

import numpy as np

from robotsmith.grasp.planner import GraspPlan
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


def _transit_lerp(qa: np.ndarray, qb: np.ndarray, params) -> list[np.ndarray]:
    """Joint-space interpolation qa→qb at transport resolution ([] if no move)."""
    if np.allclose(qa[:7], qb[:7], atol=1e-3):
        return []
    steps = max(params.transport_steps, params.min_steps)
    return [qa + (qb - qa) * (i + 1) / steps for i in range(steps)]


def _place_transport_seed(ctx: SkillContext, q0: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Start pose + transit for the inbound move: aim the base joint at the target so the
    joint-space transport segment does not under-rotate the base and release off to the side."""
    if ctx.phase not in ("all", "transport"):
        return q0, []
    seed = q0.copy()
    target_xy = np.asarray(ctx.obj_pos, dtype=np.float64)
    seed[0] = float(np.arctan2(float(target_xy[1]), float(target_xy[0])))
    return seed, _transit_lerp(q0, seed, ctx.params)


@register("place", post=bound_leaf, settle_s=0.5)
def run_place(ctx: SkillContext) -> list[np.ndarray]:
    # Phase split (live re-anchoring): "transport" moves to straight above the
    # target; "descend" re-reads the (recoiled) target and goes vertical. "all" is
    # the single-pass legacy path for static anchors. ``obj_pos`` is already
    # re-resolved per phase by the runtime. See ``skill_phases``.
    place_plan, requested_place_z, resolved_place_z = _resolve_place_plan(ctx)
    q0 = np.asarray(ctx.qpos, dtype=np.float64)
    start, transit = _place_transport_seed(ctx, q0)
    place_seg = ctx.executor.place(
        place_plan,
        ctx.motion_planner,
        start,
        ctx.params,
        phase=ctx.phase,
    )
    seg = transit + place_seg
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
            "motion_trace": getattr(ctx.executor, "last_motion_trace", None),
        }
    )
    # The object is only released on the descend phase; transport still carries it,
    # so hold-state carry-over must survive into descend.
    if ctx.phase in ("all", "descend"):
        ctx.held_category = None
        ctx.held_place_z = None
    return seg
