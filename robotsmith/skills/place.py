"""``place`` primitive: transport the held object and release it at the target.

Supports tabletop places (explicit/held ``place_z``) and articulated-opening
places (live tray-floor anchor). With a collision-aware planner the transport
hover height is authored from reachability instead of a template constant
(see docs/features/drawer_feature.md).
"""

from __future__ import annotations

import logging

import numpy as np

from robotsmith.grasp.planner import GraspPlan
from robotsmith.motion.planner import Waypoint
from robotsmith.motion.rocrobo_world import build_obstacle_world
from robotsmith.skills.base import SkillContext
from robotsmith.skills.frames import _float_or_none, _manipulated_exempt_links
from robotsmith.skills.registry import register

logger = logging.getLogger(__name__)

# EE-flange height above the tray-floor world point when placing into an
# articulated opening. The opening anchor resolves to the cavity floor; lift the
# flange by the gripper's reach (finger length + margin) so an empty gripper /
# held payload settles just above the floor instead of being driven through it.
# Robot/gripper property (not asset geometry), so it lives here, not in metadata.
_DRAWER_PLACE_EE_CLEARANCE = 0.10

# Adaptive transport-hover authoring (docs/features/drawer_feature.md). A fixed
# template hover height (``retreat_z`` / ``retreat_z − place_z``) is blind to
# whether the goal is reachable from this base/arm pose with the carried payload;
# for a drawer-interior place it can pin the hover too low, so the collision-aware
# plan_motion misses the endpoint and falls back to a wall-scraping straight line.
# Instead we sweep the hover endpoint up and take the LOWEST height whose
# single-waypoint plan_motion succeeds — the serve returns a trajectory only when
# within_limits & end_pos<tol & world_clear all pass, so a successful plan IS an
# authorable (reachable, collision-free) endpoint. Reachability-driven, removing
# the per-scenario hover knob (no constant, no branch choice).
_PLACE_HOVER_AUTHOR_MIN = 0.12   # lowest standoff above the drop point to try (m)
_PLACE_HOVER_AUTHOR_MAX = 0.45   # ceiling above the drop point (m)
_PLACE_HOVER_AUTHOR_STEP = 0.04  # sweep resolution (m)


def _author_transport_hover(ctx: SkillContext, place_plan, seed, world):
    """Raise ``place_plan``'s transport-hover endpoint to the lowest reachable height.

    Sweeps the hover z above the drop point (``grasp_pos``) and stops at the first
    height whose single-waypoint ``plan_motion`` succeeds (all serve gates pass).
    Mutates and returns ``place_plan`` (``pre_grasp_pos`` / ``retreat_pos`` keep the
    drop x/y, share the authored z). No-op when no height clears — the template
    height is left in place and the executor's straight-line fallback runs.
    """
    mp = ctx.motion_planner
    drop = np.asarray(place_plan.grasp_pos, dtype=np.float64)
    quat = place_plan.pre_grasp_quat
    finger = place_plan.finger_closed
    attach = ctx.executor.held_attach
    s = _PLACE_HOVER_AUTHOR_MIN
    while s <= _PLACE_HOVER_AUTHOR_MAX + 1e-9:
        cand = np.array([drop[0], drop[1], drop[2] + s], dtype=np.float64)
        plan = mp.plan_motion(
            seed, [Waypoint(cand, quat, finger)], world=world, attach=attach
        )
        if plan.success and plan.trajectory:
            place_plan.pre_grasp_pos = cand
            place_plan.retreat_pos = cand.copy()
            logger.debug(
                "[skills] authored place transport hover z=%.3f (standoff %.2f m "
                "above drop) — lowest reachable endpoint",
                float(cand[2]), s,
            )
            return place_plan
        s += _PLACE_HOVER_AUTHOR_STEP
    logger.warning(
        "[skills] no place transport hover cleared in [%.2f, %.2f] m above drop; "
        "keeping template height (fallback handles it)",
        _PLACE_HOVER_AUTHOR_MIN, _PLACE_HOVER_AUTHOR_MAX,
    )
    return place_plan


def _resolve_place_plan(
    ctx: SkillContext,
) -> tuple[GraspPlan, float | None, float | None]:
    """Build the place plan + (requested, resolved) place_z for the trace.

    Articulated-opening anchors place at the live tray-floor world point (lifted
    by the gripper clearance), with no table-relative place_z. Tabletop places use
    an explicit ``place_z`` override, else reuse the held object's pick height.
    """
    skill = ctx.skill
    frame = ctx.scene_state.get("target_frames", {}).get(skill.target)
    requested_place_z = _float_or_none(skill.params.get("place_z"))
    if getattr(frame, "kind", None) == "articulated_opening":
        place_point = np.asarray(ctx.obj_pos, dtype=np.float64).copy()
        place_point[2] += _DRAWER_PLACE_EE_CLEARANCE
        place_plan = ctx.planner.plan_place(
            ctx.obj_pos, category=ctx.category, place_point_world=place_point
        )
        return place_plan, requested_place_z, None
    if requested_place_z is None and ctx.held_place_z is None:
        raise ValueError(
            f"Place skill for {skill.target!r} requires a preceding pick "
            f"or an explicit place_z"
        )
    resolved_place_z = (
        requested_place_z if requested_place_z is not None else ctx.held_place_z
    )
    place_plan = ctx.planner.plan_place(
        ctx.obj_pos, category=ctx.category, place_z_override=resolved_place_z
    )
    return place_plan, requested_place_z, resolved_place_z


def _author_place_world(
    ctx: SkillContext, place_plan: GraspPlan, q0: np.ndarray
) -> tuple[list[dict] | None, GraspPlan]:
    """Collision world + reachability-authored transport hover (collision-aware only).

    The world is built from the live scene_state (articulated furniture envelopes +
    ground); the manipulated asset's moving link is the *destination* (reaching into
    an open drawer tray), not an obstacle, so it is exempt. The hover endpoint height
    is authored from reachability instead of the template constant so the carried-in
    plan_motion clears the cabinet rather than scraping (drawer_feature.md). Tabletop
    / collision-blind runs build no world and leave the plan untouched.
    """
    if not ctx.motion_planner.collision_aware:
        return None, place_plan
    place_world = build_obstacle_world(
        ctx.scene_state,
        ground_z=float(ctx.scene_state.get("table_surface_z", 0.0)),
        exempt_links=_manipulated_exempt_links(ctx),
    )
    return place_world, _author_transport_hover(ctx, place_plan, q0, place_world)


def _place_transport_seed(
    ctx: SkillContext, place_plan: GraspPlan, q0: np.ndarray, place_world: list[dict] | None
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Start pose + transit for the inbound move ("lift then swing").

    transport/all: aim the base joint at the target (else the place inherits the
    pick-end pose and Genesis under-rotates back, releasing off to the side); with a
    collision-aware planner also raise the payload to the authored hover height here,
    so the joint0 swing happens at hover (joint0 rotation preserves EE z) instead of
    sweeping the carried object through the drawer wall at the low pick-retreat
    height. descend: start from the post-swing pose unchanged (no re-seed/re-swing).
    """
    if ctx.phase not in ("all", "transport"):
        return q0, []
    seed = q0.copy()
    target_xy = np.asarray(ctx.obj_pos, dtype=np.float64)
    seed[0] = float(np.arctan2(float(target_xy[1]), float(target_xy[0])))
    if ctx.motion_planner.collision_aware:
        q_hover = ctx.motion_planner.solve_ik(
            place_plan.pre_grasp_pos,
            place_plan.pre_grasp_quat,
            place_plan.finger_closed,
            init_qpos=seed,
            world=place_world,
            attach=ctx.executor.held_attach,
        )
        seed[:7] = np.asarray(q_hover, dtype=np.float64)[:7]
    transit: list[np.ndarray] = []
    if not np.allclose(q0[:7], seed[:7], atol=1e-3):
        steps = max(ctx.params.transport_steps, ctx.params.min_steps)
        transit = [q0 + (seed - q0) * (i + 1) / steps for i in range(steps)]
    return seed, transit


@register("place")
def run_place(ctx: SkillContext) -> list[np.ndarray]:
    # Phase split (live re-anchoring): "transport" moves to straight above the
    # target; "descend" re-reads the (recoiled) target and goes vertical. "all" is
    # the single-pass legacy path for static anchors. ``obj_pos`` is already
    # re-resolved per phase by the runtime. See ``skill_phases``.
    place_plan, requested_place_z, resolved_place_z = _resolve_place_plan(ctx)
    q0 = np.asarray(ctx.qpos, dtype=np.float64)
    place_world, place_plan = _author_place_world(ctx, place_plan, q0)
    start, transit = _place_transport_seed(ctx, place_plan, q0, place_world)
    place_seg = ctx.executor.place(
        place_plan,
        ctx.motion_planner,
        start,
        ctx.params,
        world=place_world,
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
    # so hold-state carry-over (incl. the ACO) must survive into descend.
    if ctx.phase in ("all", "descend"):
        ctx.held_category = None
        ctx.held_place_z = None
        ctx.executor.held_attach = None
    return seg
