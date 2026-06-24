"""rocRobo payload authoring — turn a robotsmith scene into serve primitives.

This is the adapter layer between robotsmith's scene/asset model and the rocRobo
serve input schema. It produces the two payloads the planner consumes:

- **collision world** (`build_obstacle_world` / `build_collision_world`): the
  articulated furniture the arm must avoid, as box obstacles at their live pose +
  joint value, plus an optional ground/support halfspace. Geometry is parsed from
  the asset URDF (single source of truth).
- **attached collision object** (`build_payload_spheres`): a grasped rigid object
  as a sphere envelope in the object frame, so a *carried* payload is kept clear
  of obstacles too.

These are pure builders with no rocRobo process dependency; callers
(`robotsmith.skills`, tooling) build the payloads here, then hand them to
``RocRoboBackend`` (rocrobo_backend.py) which talks to the serve.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from robotsmith.assets.geometry import rotate_vec_wxyz
from robotsmith.assets.schema import Asset, parse_urdf_collision_boxes

# Panda finger/hand links the grasped payload legitimately touches; exempted
# from self-collision while attached (ACO). Names match the rocRobo panda sphere
# set used by ``serve.build_attach``.
PANDA_FINGER_LINKS = ("panda_leftfinger", "panda_rightfinger", "panda_hand")

# NOTE: the current serve `build_world` reads only box center+extent (no
# orientation), so boxes are axis-aligned. The drawer_cabinet is placed with a
# 180 deg yaw, under which axis-aligned boxes are invariant — correct here. A
# non-180 yaw would need serve-side ``wxyz`` support (handshake item).


# ---------------------------------------------------------------------------
# Quaternion helpers (wxyz), local so motion/ does not depend on orchestration/.
# ---------------------------------------------------------------------------


def _rpy_to_quat(rpy: tuple[float, float, float]) -> np.ndarray:
    r, p, y = (float(v) for v in rpy)
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# World builder
# ---------------------------------------------------------------------------


def _box_primitive(center: np.ndarray, size: np.ndarray, quat: np.ndarray) -> dict:
    # serve's pyroki ``Box.from_extent`` expects the FULL box size and halves it.
    return {
        "type": "box",
        "center": [float(v) for v in center],
        "extent": [float(v) for v in size],
        "quat": [float(v) for v in quat],
    }


def ground_halfspace(z: float) -> dict:
    """Support-plane keep-out as an upward halfspace at world height ``z``.

    ``z`` is the world-frame height of the surface the arm must not penetrate —
    in practice the table top the Franka is mounted on, not literal floor z=0.
    """
    return {
        "type": "halfspace",
        "point": [0.0, 0.0, float(z)],
        "normal": [0.0, 0.0, 1.0],
    }


def _rigid_moving_links(asset: Asset, moving_link: str | None) -> set[str]:
    """``moving_link`` plus every link rigidly attached to it via fixed joints.

    The drawer slide moves its joint child *and* anything bolted to that child
    (e.g. a separated ``handle`` link), so they share the same slide displacement.
    """
    if moving_link is None:
        return set()
    moving: set[str] = {moving_link}
    joints = list(getattr(asset, "joints", []) or [])
    changed = True
    while changed:
        changed = False
        for j in joints:
            if j.type == "fixed" and j.parent in moving and j.child not in moving:
                moving.add(j.child)
                changed = True
    return moving


def build_collision_world(
    asset: Asset,
    asset_pos,
    asset_quat,
    scale: float,
    *,
    joint_value: float = 0.0,
    moving_link: str | None = None,
    joint_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    exempt_links: Sequence[str] = (),
    ground_z: float | None = None,
) -> list[dict]:
    """Turn one articulated asset into box obstacles for plan_motion to avoid.

    Any articulated furniture the arm could bump into (a drawer cabinet, a
    cupboard, …) is converted to collision boxes here so the planner steers
    clear of it. Each URDF collision box is placed at the asset's live world
    pose with the same convention as ``resolve_frame``
    (``center = asset_pos + R(asset_quat) @ (scale * origin)``); the moving link
    is additionally shifted by ``R(asset_quat) @ joint_axis * joint_value``.

    Args:
        asset: the articulated asset (URDF collision boxes + joint list).
        asset_pos: asset root position in world frame.
        asset_quat: asset root orientation in world frame (wxyz).
        scale: metric scale applied to the URDF geometry.
        joint_value: how far the joint is currently open (0 = closed/rest, e.g.
            the drawer's pulled-out distance), used to move ``moving_link`` to
            its live position. Applied as a translation, so prismatic-only — a
            revolute joint (e.g. a swinging door) would need a rotation instead
            (see overall_todo.md).
        moving_link: the joint's child link (plus anything fixed-jointed to it,
            e.g. a handle) that rides ``joint_value``. None = whole asset static.
        joint_axis: joint axis in the asset-local frame.
        exempt_links: links to drop from the world — the ones intentionally
            contacted (e.g. the handle / pulled drawer link during open/close).
        ground_z: if given, also add a support-plane halfspace at this height.
    """
    asset_pos = np.asarray(asset_pos, dtype=np.float64)
    asset_quat = np.asarray(asset_quat, dtype=np.float64)
    scale = float(scale)
    axis = np.asarray(joint_axis, dtype=np.float64)
    joint_offset_world = rotate_vec_wxyz(asset_quat, axis) * float(joint_value)

    exempt = set(exempt_links)
    # moving_link + its fixed-joint descendants ride the joint together, else a
    # separated handle link would be left stranded while the body moves.
    moving_set = _rigid_moving_links(asset, moving_link)
    boxes = parse_urdf_collision_boxes(asset.urdf_path)
    world: list[dict] = []
    for box in boxes:
        if box.link in exempt:
            continue
        center_local = scale * np.asarray(box.origin_xyz, dtype=np.float64)
        center = asset_pos + rotate_vec_wxyz(asset_quat, center_local)
        if box.link in moving_set:
            center = center + joint_offset_world
        size = scale * np.asarray(box.size, dtype=np.float64)
        quat = _quat_mul(asset_quat, _rpy_to_quat(box.origin_rpy))
        world.append(_box_primitive(center, size, quat))

    if ground_z is not None:
        world.append(ground_halfspace(ground_z))
    return world


def build_obstacle_world(
    scene_state: dict,
    *,
    ground_z: float | None = None,
    exempt_links: Sequence[str] = (),
) -> list[dict]:
    """Build a collision world from all articulated furniture in ``scene_state``.

    Every articulated asset (e.g. the drawer cabinet) becomes box obstacles at
    its live pose + joint slide; rigid pick/place objects are not added (they are
    small and/or the intentional contact target).

    ``exempt_links`` drops intentionally-contacted links (by URDF link name) from
    the world — e.g. open/close must touch the handle / pulled drawer link, so
    those are exempted while the carcass stays as an obstacle. Place contacts no
    link and passes nothing.
    """
    assets = scene_state.get("assets", {})
    positions = scene_state.get("positions", {})
    quats = scene_state.get("object_quats", {})
    scales = scene_state.get("object_scales", {})
    joint_positions = scene_state.get("joint_positions", {})

    world: list[dict] = []
    for name, asset in assets.items():
        if asset is None or not getattr(asset, "is_articulated", False):
            continue
        if name not in positions:
            continue
        # is_articulated guarantees at least one movable joint; first joint only
        # (multi-DOF assets are a future item, see overall_todo.md).
        joint = asset.movable_joints[0]
        joint_value = float(joint_positions.get(name, {}).get(joint.name, 0.0) or 0.0)
        world += build_collision_world(
            asset,
            positions[name],
            quats.get(name, (1.0, 0.0, 0.0, 0.0)),
            scales.get(name, 1.0),
            joint_value=joint_value,
            moving_link=asset.primary_moving_link,
            joint_axis=joint.axis,
            exempt_links=exempt_links,
        )
    if ground_z is not None:
        world.append(ground_halfspace(ground_z))
    return world


def build_payload_spheres(asset, scale: float = 1.0) -> list[dict]:
    """Sphere decomposition of a grasped rigid object, in the OBJECT frame.

    For the attached-collision-object (ACO) path: the held object must enter the
    collision-aware planner so a *carried* payload (e.g. a die) is kept clear of
    obstacles, not just the arm. rocRobo ``attach`` expects spheres in the payload
    frame; ``serve.build_attach`` rides them on the EE via ``T_ee_obj``.

    v1 envelope: 8 spheres on the 2x2x2 sub-box grid of the object AABB, each
    sized (center-to-corner of its sub-box) to cover it — conservative (slightly
    larger than the box) but cheap and orientation-free. Extents come from the
    mesh collision audit when present, else ``size_cm`` (cm). Returns ``[]`` when
    no size metadata is available (caller then attaches nothing).
    """
    meta = getattr(asset, "metadata", None)
    ext = None
    if meta is not None:
        ext = (getattr(meta, "mesh", {}) or {}).get("collision_extents")
        if not ext:
            size_cm = getattr(meta, "size_cm", None)
            ext = [float(v) / 100.0 for v in size_cm] if size_cm else None
    if not ext:
        return []
    half = 0.5 * np.asarray(ext, dtype=np.float64) * float(scale)
    radius = 0.5 * float(np.linalg.norm(half))  # center-to-corner of each sub-box
    spheres: list[dict] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                center = [sx * half[0] / 2.0, sy * half[1] / 2.0, sz * half[2] / 2.0]
                spheres.append({"center": [float(v) for v in center], "radius": radius})
    return spheres
