"""Relation solver: scenario object relations -> world positions and yaws.

Every rule reads only geometry derived from the asset build files (visual bounds,
collision boxes, place targets, prismatic joints) plus the robot's workspace:

- ``on: support`` / ``on: <id>.<target>``: bottom of the visual bounds rests on
  the tabletop or on a static place-target surface (centred on it).
- ``anchor_xy``: explicit planar position (fixtures).
- ``near: robot``: sampled in the Franka workspace annulus.
- ``clear_of``: xy footprints do not overlap; ``<id>.swept`` adds the volume the
  object's prismatic parts sweep over their joint range.
- ``facing: robot``: the asset's front (prismatic axis, else the reverse of a
  place target's approach) points at the robot base; ``yaw_deg`` overrides.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from robotsmith.assets import derive
from robotsmith.assets.library import AssetLibrary
from robotsmith.assets.schema import Asset, parse_urdf_collision_boxes
from robotsmith.motion.rocrobo_world import rigid_moving_links
from robotsmith.scenes.config import DEFAULT_WORKSPACE_XY, SceneConfig, table_center_xy
from robotsmith.sim.franka import REACH_HALF_ANGLE_DEG, REACH_M

SUPPORT = "support"
ROBOT = "robot"
SPAWN_CLEARANCE_M = 0.001
FOOTPRINT_MARGIN_M = 0.02
TABLE_MARGIN_M = 0.05
ROBOT_BASE_HALF_M = 0.15
MAX_SAMPLES = 500
CAMERA_OFFSET = (-1.0, -1.6, 1.2)

_DEFAULTS = SceneConfig(name="")
TABLE_TOP_Z = _DEFAULTS.table_height + _DEFAULTS.table_size[2] / 2.0


class SolveError(ValueError):
    """Raised when the relations cannot be satisfied with the given assets."""


@dataclass(frozen=True)
class Placement:
    position: tuple[float, float, float]
    yaw_deg: float

    def __post_init__(self):
        object.__setattr__(self, "position", tuple(float(v) for v in self.position))
        object.__setattr__(self, "yaw_deg", float(self.yaw_deg))


@dataclass(frozen=True)
class SolvedScene:
    placements: dict[str, Placement]
    table_size: tuple[float, float, float]
    camera_position: tuple[float, float, float]
    camera_target: tuple[float, float, float]


def place_target(asset: Asset, name: str) -> dict:
    for target in asset.metadata.place_targets:
        if target["name"] == name:
            return target
    raise SolveError(f"asset '{asset.name}' has no place target '{name}'")


def split_ref(ref: str) -> tuple[str, str | None]:
    obj_id, _, target = ref.partition(".")
    return obj_id, target or None


def front_direction(asset: Asset) -> np.ndarray:
    """Unit xy direction the asset is used from, in the asset frame."""
    for joint in asset.movable_joints:
        if joint.type == "prismatic":
            return _unit(np.asarray(joint.axis[:2], dtype=np.float64))
    for target in asset.metadata.place_targets:
        if target.get("approach"):
            return _unit(-np.asarray(target["approach"][:2], dtype=np.float64))
    raise SolveError(f"asset '{asset.name}' has no front direction; give yaw_deg")


def solve(objects: list[dict], library: AssetLibrary, rng: random.Random) -> SolvedScene:
    assets = {o["id"]: _asset(library, o["asset"]) for o in objects}
    bounds = {oid: derive.visual_bounds(a.urdf_path) for oid, a in assets.items()}
    placed: dict[str, Placement] = {}

    def footprint(oid: str, swept: bool = False) -> tuple[np.ndarray, np.ndarray]:
        p = placed[oid]
        return _footprint(assets[oid], bounds[oid], p.position, p.yaw_deg, swept)

    # Anchored and surface-mounted objects first, then sampled ones.
    order = sorted(objects, key=lambda o: "near" in o)
    for obj in order:
        oid, asset = obj["id"], assets[obj["id"]]
        lo_z = float(bounds[oid][0][2])
        clearance = 0.0 if asset.is_fixture else SPAWN_CLEARANCE_M
        parent, target = split_ref(obj["on"])
        if parent != SUPPORT:
            if target is None or parent not in placed:
                raise SolveError(f"'{oid}': on '{obj['on']}' must name a placed object's target")
            surface = place_target(assets[parent], target).get("surface_local")
            if surface is None:
                raise SolveError(f"'{oid}': '{obj['on']}' is not a static surface")
            world = _to_world(placed[parent], surface)
            placed[oid] = Placement((world[0], world[1], world[2] - lo_z + clearance),
                                    _yaw(obj, asset, world[:2]))
            continue
        z = TABLE_TOP_Z - lo_z + clearance
        if "anchor_xy" in obj:
            xy = np.asarray(obj["anchor_xy"], dtype=np.float64)
            placed[oid] = Placement((xy[0], xy[1], z), _yaw(obj, asset, xy))
            continue
        if obj.get("near") != ROBOT:
            raise SolveError(f"'{oid}' on support needs anchor_xy or near: robot")
        others = [o["id"] for o in objects if o["id"] in placed and split_ref(o["on"])[0] == SUPPORT]
        swept = {split_ref(r)[0] for r in obj.get("clear_of", []) if r.endswith(".swept")}
        blockers = [footprint(other, swept=other in swept) for other in others]
        for _ in range(MAX_SAMPLES):
            r = rng.uniform(*REACH_M)
            theta = math.radians(rng.uniform(-REACH_HALF_ANGLE_DEG, REACH_HALF_ANGLE_DEG))
            xy = np.array([r * math.cos(theta), r * math.sin(theta)])
            placed[oid] = Placement((xy[0], xy[1], z), _yaw(obj, asset, xy))
            mine = footprint(oid)
            if not any(_overlap(mine, b, FOOTPRINT_MARGIN_M) for b in blockers):
                break
        else:
            raise SolveError(f"'{oid}': no free spot near the robot after {MAX_SAMPLES} samples")

    prints = [footprint(oid) for oid in placed]
    cx, cy = table_center_xy(DEFAULT_WORKSPACE_XY)
    half_x = max([abs(cx) + ROBOT_BASE_HALF_M] + [max(abs(lo[0] - cx), abs(hi[0] - cx)) for lo, hi in prints])
    half_y = max([ROBOT_BASE_HALF_M] + [max(abs(lo[1] - cy), abs(hi[1] - cy)) for lo, hi in prints])
    table_size = (
        max(_DEFAULTS.table_size[0], 2.0 * (half_x + TABLE_MARGIN_M)),
        max(_DEFAULTS.table_size[1], 2.0 * (half_y + TABLE_MARGIN_M)),
        _DEFAULTS.table_size[2],
    )
    lo = np.min([p[0] for p in prints], axis=0)
    hi = np.max([p[1] for p in prints], axis=0)
    target = (float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), TABLE_TOP_Z)
    camera = tuple(t + d for t, d in zip(target, CAMERA_OFFSET))
    return SolvedScene(placed, table_size, camera, target)


def _asset(library: AssetLibrary, name: str) -> Asset:
    asset = library.get(name)
    if asset is None:
        raise SolveError(f"asset '{name}' not found in the library")
    return asset


def _yaw(obj: dict, asset: Asset, xy: np.ndarray) -> float:
    if "yaw_deg" in obj:
        return float(obj["yaw_deg"])
    if obj.get("facing") != ROBOT:
        return 0.0
    front = front_direction(asset)
    to_robot = -np.asarray(xy, dtype=np.float64)
    return math.degrees(math.atan2(to_robot[1], to_robot[0]) - math.atan2(front[1], front[0]))


def _to_world(p: Placement, local) -> np.ndarray:
    c, s = math.cos(math.radians(p.yaw_deg)), math.sin(math.radians(p.yaw_deg))
    x, y, z = local
    return np.array([p.position[0] + c * x - s * y, p.position[1] + s * x + c * y, p.position[2] + z])


def _corners_xy(lo, hi) -> np.ndarray:
    return np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])


def _footprint(asset: Asset, bounds, position, yaw_deg: float, swept: bool):
    points = [_corners_xy(*bounds)]
    if swept:
        tfs = derive.link_transforms(asset.urdf_path)
        boxes = parse_urdf_collision_boxes(asset.urdf_path)
        for joint in asset.movable_joints:
            if joint.type != "prismatic":
                continue
            links = rigid_moving_links(asset, joint.child)
            axis = np.asarray(joint.axis, dtype=np.float64)
            for box in boxes:
                if box.link not in links:
                    continue
                lo, hi = derive.box_bounds(box, tfs[box.link])
                for q in (joint.lower or 0.0, joint.upper or 0.0):
                    points.append(_corners_xy(lo + axis * q, hi + axis * q))
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    world = np.vstack(points) @ np.array([[c, s], [-s, c]]) + np.asarray(position[:2])
    return world.min(axis=0), world.max(axis=0)


def _overlap(a, b, margin: float) -> bool:
    return bool(np.all(a[0] < b[1] + margin) and np.all(b[0] < a[1] + margin))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise SolveError("front direction has no xy component")
    return v / n
