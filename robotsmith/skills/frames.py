"""Articulated frame / anchor resolution for the skill layer.

The single seam that turns a self-describing spatial reference (``world`` /
``articulated`` / ``articulated_opening``) into a live world point, plus the
open/close handle-drag resolution and the collision-world link exemptions that
ride on the same asset geometry. Primitives consume world points and link lists
from here and never re-derive anchoring themselves.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from robotsmith.assets.geometry import rotate_vec_wxyz

if TYPE_CHECKING:
    from robotsmith.skills.base import SkillContext


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_frame(frame, scene_state: dict, *, slide_override: float | None = None):
    """Resolve a self-describing frame reference to a world point.

    Single seam for every spatial reference: primitives consume world points and
    never re-derive anchoring.

    - ``kind="world"``: return the constant ``xyz``.
    - ``kind="articulated"``: the parent asset's live pose offset by
      ``local_offset`` (in the asset frame, scaled by the asset's metric scale)
      and carried along the task joint by its live travel. ``slide_override``
      lets the open/close drag pass the travel it already read (with its nominal
      endpoint fallback); otherwise the live ``joint_positions`` reading is used.
    """
    kind = getattr(frame, "kind", "world")
    if kind == "world":
        return np.asarray(frame.xyz, dtype=np.float64)
    if kind not in ("articulated", "articulated_opening"):
        raise ValueError(f"unknown frame anchor kind {kind!r}")

    parent = frame.parent
    asset = scene_state.get("assets", {}).get(parent)
    if asset is None:
        raise ValueError(f"frame anchor {parent!r} is not an articulated asset")
    parent_pos = np.asarray(scene_state["positions"][parent], dtype=np.float64)
    quat = scene_state.get("object_quats", {}).get(parent, [1.0, 0.0, 0.0, 0.0])
    scale = float(scene_state.get("object_scales", {}).get(parent, 1.0))

    if kind == "articulated_opening":
        # Live midpoint of the asset's exposed opening: lip + open_dir*(slide*frac)
        # at tray-floor height. Geometry from the asset's place_targets metadata
        # (no per-scene numbers); tracks the real opening as the part recoils.
        spec = _place_target_spec(asset, frame.opening)
        local = np.asarray(spec["lip_local"], dtype=np.float64) * scale
        local[2] = float(spec.get("tray_floor_z", spec["lip_local"][2])) * scale
        joint_name = spec.get("joint")
        travel_fraction = float(spec.get("travel_fraction", 0.5))
    else:
        local = np.asarray(frame.local_offset, dtype=np.float64) * scale
        joint_name = frame.joint
        travel_fraction = 1.0

    world = parent_pos + rotate_vec_wxyz(quat, local)

    joint = asset.get_joint(joint_name) if joint_name else None
    if joint is not None:
        open_dir = rotate_vec_wxyz(quat, np.asarray(joint.axis, dtype=np.float64))
        norm = float(np.linalg.norm(open_dir))
        if norm > 1e-9:
            open_dir = open_dir / norm
        slide = slide_override
        if slide is None:
            slide = _float_or_none(
                scene_state.get("joint_positions", {}).get(parent, {}).get(joint_name)
            )
            if slide is None:
                slide = 0.0
        world = world + open_dir * float(slide) * travel_fraction
    return world


def _place_target_spec(asset, name: str | None) -> dict:
    """Look up a named ``place_targets`` opening in the asset metadata."""
    targets = list(getattr(asset.metadata, "place_targets", []) or [])
    if not targets:
        raise ValueError(f"asset {asset.name!r} has no place_targets metadata")
    if name is None:
        return targets[0]
    for spec in targets:
        if spec.get("name") == name:
            return spec
    raise ValueError(f"asset {asset.name!r} has no place_target {name!r}")


def _resolve_handle_drag(ctx: SkillContext, *, opening: bool):
    """Resolve (handle_world_pos, move_vec, moving_link) for an open/close drag.

    The handle is a frame anchored to the articulated asset's joint (its
    ``metadata.handles`` local position), resolved through ``resolve_frame``; the
    drag direction is the URDF joint axis in world frame and the distance is the
    remaining joint travel (or skill ``distance`` override).
    """
    skill = ctx.skill
    scene_state = ctx.scene_state
    asset = scene_state.get("assets", {}).get(skill.target)
    if asset is None:
        raise ValueError(f"{skill.name} requires an asset for {skill.target!r}")

    handles = list(getattr(asset.metadata, "handles", []) or [])
    if not handles:
        raise ValueError(f"{skill.target!r} has no handle annotation for {skill.name}")
    local_pos = handles[0].get("local_pos", [0.0, 0.0, 0.0])

    task_joints = list(getattr(asset.metadata, "task_joints", []) or [])
    joint_name = skill.params.get("joint") or (task_joints[0] if task_joints else None)
    joint = asset.get_joint(joint_name) if joint_name else None
    if joint is None:
        raise ValueError(f"{skill.target!r} has no task joint for {skill.name}")

    scale = float(scene_state.get("object_scales", {}).get(skill.target, 1.0))
    quat = scene_state.get("object_quats", {}).get(skill.target, [1.0, 0.0, 0.0, 0.0])
    open_dir_world = rotate_vec_wxyz(quat, np.asarray(joint.axis, dtype=np.float64))
    norm = float(np.linalg.norm(open_dir_world))
    if norm > 1e-9:
        open_dir_world = open_dir_world / norm

    # The URDF joint travel is in the asset's native (unscaled) frame; the entity
    # is loaded at ``metric_scale``, so world-frame travel scales with it.
    travel = float(joint.upper if joint.upper is not None else 0.3)
    distance = _float_or_none(skill.params.get("distance"))
    full_travel = distance if distance is not None else travel * scale

    # Re-anchor to the joint's live opening so the grasp lands on the handle where
    # it actually is. The runtime refreshes ``joint_positions`` before each
    # subtask; without a live reading (Genesis-free tests) fall back to the
    # nominal endpoints (closed for open, fully open for close).
    joints_live = scene_state.get("joint_positions", {})
    slide_now = (
        _float_or_none(joints_live.get(skill.target, {}).get(joint_name))
        if joint_name
        else None
    )
    if slide_now is None:
        slide_now = 0.0 if opening else full_travel

    handle_frame = SimpleNamespace(
        kind="articulated",
        parent=skill.target,
        joint=joint_name,
        local_offset=local_pos,
    )
    handle_world = resolve_frame(handle_frame, scene_state, slide_override=slide_now)

    if opening:
        move_vec = open_dir_world * (full_travel - slide_now)
    else:
        # Close: drag from the current opening all the way back to closed.
        move_vec = -open_dir_world * slide_now
    return handle_world, move_vec, joint.child


def _handle_links(ctx: SkillContext) -> list[str]:
    """URDF link names of the touchable handle(s) for the skill's target asset.

    With the handle on its own fixed-jointed link, open/close only needs to exempt
    this knob link from the collision world — the drawer body stays an obstacle.
    Returns ``[]`` when the asset has no handle annotation or no per-handle link.
    """
    asset = ctx.scene_state.get("assets", {}).get(ctx.skill.target)
    handles = list(getattr(getattr(asset, "metadata", None), "handles", []) or [])
    links = [h.get("link") for h in handles if h.get("link")]
    return list(dict.fromkeys(links))


def _manipulated_exempt_links(ctx: SkillContext) -> list[str]:
    """Moving link(s) of the articulated asset this skill manipulates/targets, to
    drop from the planner's collision world (phase-aware exemption).

    The contacted/destination part — drawer front, handle neck, tray — must be
    reachable, so it is the *goal*, not a hard obstacle; the fixed carcass (``base``)
    stays an obstacle. ``skill.target`` is either the asset itself (open/close drag)
    or a target_position anchored on one (place into an open tray). Returns ``[]``
    for non-articulated targets (e.g. tabletop place), leaving the world unchanged.
    """
    scene_state = ctx.scene_state
    assets = scene_state.get("assets", {})
    target = ctx.skill.target
    asset = assets.get(target)
    if asset is None:
        frame = scene_state.get("target_frames", {}).get(target)
        parent = getattr(frame, "parent", None)
        asset = assets.get(parent) if parent else None
    if asset is None or not getattr(asset, "is_articulated", False):
        return []
    link = asset.primary_moving_link
    return [link] if link else []
