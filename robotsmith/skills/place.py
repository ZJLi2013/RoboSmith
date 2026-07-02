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
from robotsmith.skills.frames import (
    _float_or_none,
    resolve_approach,
)
from robotsmith.skills.registry import register

logger = logging.getLogger(__name__)

# EE-flange height above the tray-floor world point when placing into an
# articulated opening. The opening anchor resolves to the cavity floor; lift the
# flange by the gripper's reach (finger length + margin) so an empty gripper /
# held payload settles just above the floor instead of being driven through it.
# Robot/gripper property (not asset geometry), so it lives here, not in metadata.
_DRAWER_PLACE_EE_CLEARANCE = 0.10

# Adaptive pre-insert standoff authoring (docs/features/drawer_feature.md,
# place_insertion_strategy.md §4 = supporter milestone A). A fixed template hover
# height is blind to whether the goal is reachable from this base/arm pose with the
# carried payload; for a drawer-interior or shelf place it can pin the standoff in
# an unreachable/occluded spot, so the collision-aware plan_motion misses the
# endpoint and falls back to a scraping straight line. Instead we sweep the standoff
# OUTWARD ALONG THE APPROACH AXIS (``standoff = drop − approach·d``) and take the
# CLOSEST clearance ``d`` whose single-waypoint plan_motion succeeds — the serve
# returns a trajectory only when within_limits & end_pos<tol & world_clear all pass,
# so a successful plan IS an authorable (reachable, collision-free) endpoint.
# Reachability-driven, no per-scenario knob. Top-down (approach=−Z) reduces to the
# original vertical sweep above the drop (drop − (−Z)·d = drop + Z·d), so
# drawer/tabletop authoring is byte-for-byte unchanged; a side-insert (approach≈
# horizontal) sweeps the standoff out through the open mouth along the true insert
# axis instead of straight up into the shelf above it.
_PLACE_STANDOFF_AUTHOR_MIN = 0.12   # closest standoff clearance along −approach (m)
_PLACE_STANDOFF_AUTHOR_MAX = 0.45   # farthest clearance to try (m)
_PLACE_STANDOFF_AUTHOR_STEP = 0.04  # sweep resolution (m)


def _author_transport_hover(ctx: SkillContext, place_plan, seed, world, approach):
    """Set ``place_plan``'s pre-insert standoff to the closest reachable point along
    the approach axis.

    Sweeps the standoff out along ``−approach`` from the drop point (``grasp_pos``)
    and stops at the first clearance whose single-waypoint ``plan_motion`` succeeds
    (all serve gates pass). Mutates and returns ``place_plan`` (``pre_grasp_pos`` /
    ``retreat_pos`` share the authored standoff). No-op when nothing clears — the
    template standoff is left in place and the executor's straight-line fallback
    runs. For top-down approach this is the original lowest-reachable vertical hover.
    """
    mp = ctx.motion_planner
    drop = np.asarray(place_plan.grasp_pos, dtype=np.float64)
    a = np.asarray(approach, dtype=np.float64).reshape(3)
    quat = place_plan.pre_grasp_quat
    finger = place_plan.finger_closed
    attach = ctx.executor.held_attach
    d = _PLACE_STANDOFF_AUTHOR_MIN
    while d <= _PLACE_STANDOFF_AUTHOR_MAX + 1e-9:
        cand = drop - a * d
        plan = mp.plan_motion(
            seed, [Waypoint(cand, quat, finger)], world=world, attach=attach
        )
        if plan.success and plan.trajectory:
            place_plan.pre_grasp_pos = cand
            place_plan.retreat_pos = cand.copy()
            logger.debug(
                "[skills] authored place pre-insert standoff %.2f m back along "
                "approach (%.2f,%.2f,%.2f) at (%.3f,%.3f,%.3f) — closest reachable",
                d, a[0], a[1], a[2], float(cand[0]), float(cand[1]), float(cand[2]),
            )
            return place_plan
        d += _PLACE_STANDOFF_AUTHOR_STEP
    logger.warning(
        "[skills] no place standoff cleared in [%.2f, %.2f] m along −approach "
        "(%.2f,%.2f,%.2f); keeping template standoff (fallback handles it)",
        _PLACE_STANDOFF_AUTHOR_MIN, _PLACE_STANDOFF_AUTHOR_MAX,
        a[0], a[1], a[2],
    )
    return place_plan


def _resolve_place_plan(
    ctx: SkillContext,
) -> tuple[GraspPlan, float | None, float | None]:
    """Build the place plan + (requested, resolved) place_z for the trace.

    Every place consumes the destination's already-resolved world point
    (``ctx.obj_pos`` = the marker / opening world xyz from the resolver) and
    lifts the EE-flange by a grasp-relative offset; the planner never re-derives
    a table-relative drop height. See docs/features/feature12_pick_place_into_drawer.md
    (§5 W2) and docs/refactor.md R3.

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
    # along this axis instead of straight down (docs/refactor.md R4).
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


# Threshold on |approach·Z| above which a place is treated as top-down (vertical
# Cartesian descend is fine). Below it the approach is a side-insert, the case a
# planned insert is for.
_TOP_DOWN_AXIS_TOL = 0.9


def _insert_strategy(ctx: SkillContext, approach: np.ndarray) -> str:
    """Place insert/extract motion mode: ``"planned"`` vs ``"cartesian"`` (default).

    Decided purely by **geometry + backend**, no kinematic-type branch (the end
    state is a single ``planned`` place; this gate is temporary scaffolding — see
    place_insertion_strategy.md §1.1): a **side-insert** (non-vertical approach)
    under a collision-aware backend routes descend/retreat through ``plan_motion``
    so the arm weaves under the upper shelf without the near-singular wrist flip a
    forced straight line produces. Top-down places (drawer / tabletop, approach ≈
    ±Z) fail the non-vertical test and stay Cartesian, byte-for-byte unchanged.
    """
    if not ctx.motion_planner.collision_aware:
        return "cartesian"
    if abs(float(approach[2])) > _TOP_DOWN_AXIS_TOL:
        return "cartesian"
    return "planned"


def _place_destination_exempt(ctx: SkillContext) -> tuple[list[str], list[str]]:
    """Place-destination geometry to drop from the obstacle world (the target
    surface is the *goal*, not a hard obstacle, so the planner may reach the release
    pose above it).

    Unified on the **role axis** (``is_fixture``), mirroring ``build_obstacle_world``
    — NOT the kinematic axis. The destination is named by the place-target
    (``frame.opening`` == a ``place_targets`` entry); its realization decides what to
    drop:
    - the place-target **collision box** of the same name (``place_targets`` name ==
      URDF ``<collision name>``, e.g. ``shelf_lower``) — works for any fixture,
      including a jointless single-link weldment per-link exemption cannot isolate;
    - if the destination **rides a movable joint** (drawer tray), the moving link is
      exempted too (the whole tray link is the reachable cavity).

    Returns ``(exempt_links, exempt_boxes)`` (both ``[]`` for non-fixture targets,
    leaving the world unchanged).
    """
    scene_state = ctx.scene_state
    assets = scene_state.get("assets", {})
    target = ctx.skill.target
    frame = scene_state.get("target_frames", {}).get(target)
    asset = assets.get(target) or assets.get(getattr(frame, "parent", None))
    if asset is None or not getattr(asset, "is_fixture", False):
        return [], []
    exempt_boxes: list[str] = []
    target_name = getattr(frame, "opening", None)
    if target_name:
        exempt_boxes.append(target_name)
    exempt_links: list[str] = []
    if getattr(asset, "is_articulated", False):
        link = asset.primary_moving_link
        if link:
            exempt_links.append(link)
    return exempt_links, exempt_boxes


def _author_place_world(
    ctx: SkillContext, place_plan: GraspPlan, q0: np.ndarray, approach: np.ndarray
) -> tuple[list[dict] | None, GraspPlan]:
    """Collision world + reachability-authored transport hover (collision-aware only).

    The world is built from the live scene_state by ``build_obstacle_world``, which
    emits **every fixture** (``is_fixture``) as box obstacles — articulated furniture
    (drawer cabinet, ridden by its live joint value) *and* jointless weldments
    (e.g. ``two_layer_supporter``) alike; membership keys off the role axis, not
    ``is_articulated`` (see ``build_obstacle_world`` / refactor.md). The place
    destination is dropped via ``_place_destination_exempt`` (the moving tray link
    and/or the named place-target box, e.g. ``shelf_lower``) so it is the *goal*, not
    an obstacle — unified on ``is_fixture``, covering both the drawer tray and a
    jointless shelf. The hover endpoint height is authored from reachability instead
    of the template constant so the carried-in plan_motion clears the obstacles
    rather than scraping (drawer_feature.md). Tabletop / collision-blind runs build
    no world and leave the plan untouched.
    """
    if not ctx.motion_planner.collision_aware:
        return None, place_plan
    exempt_links, exempt_boxes = _place_destination_exempt(ctx)
    place_world = build_obstacle_world(
        ctx.scene_state,
        ground_z=float(ctx.scene_state.get("table_surface_z", 0.0)),
        exempt_links=exempt_links,
        exempt_boxes=exempt_boxes,
    )
    return place_world, _author_transport_hover(
        ctx, place_plan, q0, place_world, approach
    )


def _transit_lerp(qa: np.ndarray, qb: np.ndarray, params) -> list[np.ndarray]:
    """Joint-space interpolation qa→qb at transport resolution ([] if no move)."""
    if np.allclose(qa[:7], qb[:7], atol=1e-3):
        return []
    steps = max(params.transport_steps, params.min_steps)
    return [qa + (qb - qa) * (i + 1) / steps for i in range(steps)]


def _place_transport_seed(
    ctx: SkillContext, place_plan: GraspPlan, q0: np.ndarray, place_world: list[dict] | None
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Start pose + transit for the inbound move.

    Collision-aware: no pre-seed — start from the post-pick pose and let
    ``executor.place``'s ``transport`` segment plan the whole q0→standoff inbound
    move through ``plan_motion`` (seed-continuous, collision-free). The authored
    standoff (``_author_transport_hover``) is exactly the endpoint whose
    ``plan_motion(q0 → hover)`` it gated on, so this reuses the validated plan
    (refactor.md R5).

    Collision-blind (no plan_motion): keep a geometry-agnostic base aim so the
    joint-space transport segment does not under-rotate the base and release off to
    the side.
    """
    if ctx.phase not in ("all", "transport"):
        return q0, []
    if ctx.motion_planner.collision_aware:
        return q0, []
    seed = q0.copy()
    target_xy = np.asarray(ctx.obj_pos, dtype=np.float64)
    seed[0] = float(np.arctan2(float(target_xy[1]), float(target_xy[0])))
    return seed, _transit_lerp(q0, seed, ctx.params)


@register("place")
def run_place(ctx: SkillContext) -> list[np.ndarray]:
    # Phase split (live re-anchoring): "transport" moves to straight above the
    # target; "descend" re-reads the (recoiled) target and goes vertical. "all" is
    # the single-pass legacy path for static anchors. ``obj_pos`` is already
    # re-resolved per phase by the runtime. See ``skill_phases``.
    place_plan, requested_place_z, resolved_place_z = _resolve_place_plan(ctx)
    q0 = np.asarray(ctx.qpos, dtype=np.float64)
    frame = ctx.scene_state.get("target_frames", {}).get(ctx.skill.target)
    approach = resolve_approach(frame, ctx.scene_state)
    insert_strategy = _insert_strategy(ctx, approach)
    place_world, place_plan = _author_place_world(ctx, place_plan, q0, approach)
    start, transit = _place_transport_seed(ctx, place_plan, q0, place_world)
    place_seg = ctx.executor.place(
        place_plan,
        ctx.motion_planner,
        start,
        ctx.params,
        world=place_world,
        phase=ctx.phase,
        insert_strategy=insert_strategy,
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
            "insert_strategy": insert_strategy,
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
