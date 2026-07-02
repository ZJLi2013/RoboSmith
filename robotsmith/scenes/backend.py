"""Simulator-agnostic scene resolution.

Core data flow:
    SceneConfig + AssetLibrary
        -> ProgrammaticSceneBackend.resolve()
        -> ResolvedScene(PlacedObject...)
        -> genesis_loader.load_resolved_scene()
        -> SimEnv

This module does not create simulator entities. It resolves a declarative
SceneConfig into concrete assets, upright poses, metric scales, names, and collision-checked
placements that downstream simulator adapters can load.

Main pieces:
    PlacedObject: one concrete asset instance with pose, fixed asset scale,
        and skill name.
    ResolvedScene: backend output consumed by simulator or visualization adapters.
    ProgrammaticSceneBackend: current resolver for asset lookup, upright pose,
        asset metric scale, workspace sampling, and object-object spacing.

Important helpers:
    _resolve_upright_pose: asset task pose -> table z-offset + quaternion.
    _CollisionChecker: FCL-backed spacing check with AABB fallback.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import trimesh
import trimesh.transformations as tf

try:
    import trimesh.collision as _fcl_collision

    _HAS_FCL = _fcl_collision.CollisionManager is not None
    _fcl_collision.CollisionManager()
    _HAS_FCL = True
except (ValueError, ImportError, AttributeError):
    _HAS_FCL = False

from robotsmith.assets.library import AssetLibrary
from robotsmith.assets.audit import audit_and_update
from robotsmith.assets.schema import Asset
from robotsmith.scenes.config import SceneConfig, ObjectPlacement
from robotsmith.scenes.pose_utils import task_pose_quat, task_pose_verified

logger = logging.getLogger(__name__)


@dataclass
class PlacedObject:
    """An asset placed in a scene with a concrete pose."""

    asset: Asset
    position: list[float]
    rotation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    quaternion: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    """Orientation as [w, x, y, z] quaternion (upright task pose)."""
    name: str = ""
    """Logical name for skill-target matching (e.g. 'cube', 'bowl')."""

    @property
    def metric_scale(self) -> float:
        return float(self.asset.metadata.metric_scale)

    @property
    def object_height_m(self) -> float:
        """World-frame height (Z extent) after applying upright rotation and metric scale.

        For axis-aligned objects (identity quat) this equals
        ``size_cm[2] * metric_scale``.
        For rotated objects like bowls (Y-up mesh rotated to Z-up), the
        quaternion maps the correct local axis onto world Z.
        """
        return _height_for_quat(self.asset, self.metric_scale, self.quaternion)


@dataclass
class ResolvedScene:
    """A fully resolved scene: concrete objects with poses, ready for sim loading."""

    config: SceneConfig
    placed_objects: list[PlacedObject] = field(default_factory=list)
    table_asset: Optional[Asset] = None

    def summary(self) -> str:
        lines = [f"Scene: {self.config.name} ({len(self.placed_objects)} objects)"]
        for po in self.placed_objects:
            pos = [round(v, 3) for v in po.position]
            scale_str = (
                f" metric_scale={po.metric_scale:.3f}"
                if po.metric_scale != 1.0 else ""
            )
            lines.append(
                f"  {po.name:20s} ({po.asset.name}) pos={pos} "
                f"h={po.object_height_m:.4f}m{scale_str}"
            )
        return "\n".join(lines)


def _load_collision_mesh(asset: Asset) -> Optional[trimesh.Trimesh]:
    """Load collision mesh for an asset, falling back to a bounding-box proxy."""
    if asset.collision_mesh and asset.collision_mesh.exists():
        mesh = trimesh.load(asset.collision_mesh, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            )
        return mesh
    sz = asset.metadata.size_cm
    return trimesh.creation.box(extents=[s / 100.0 for s in sz])


def _quat_to_euler(quat: list[float]) -> list[float]:
    """Convert [w,x,y,z] quaternion to [roll, pitch, yaw] Euler angles."""
    mat = tf.quaternion_matrix(quat)
    return list(tf.euler_from_matrix(mat, axes="sxyz"))


def _height_for_quat(asset: Asset, scale: float, quat: list[float]) -> float:
    sx, sy, sz = [v / 100.0 * scale for v in asset.metadata.size_cm]
    rmat = tf.quaternion_matrix(quat)[:3, :3]
    corners = np.array(
        [
            [dx * sx / 2, dy * sy / 2, dz * sz / 2]
            for dx in (-1, 1)
            for dy in (-1, 1)
            for dz in (-1, 1)
        ]
    )
    rotated = corners @ rmat.T
    return float(rotated[:, 2].max() - rotated[:, 2].min())


def _resolve_upright_pose(asset: Asset, scale: float) -> tuple[float, list[float]]:
    """Resolve the runtime object pose from canonical upright asset metadata."""
    quat = task_pose_quat(asset, "upright") or [1.0, 0.0, 0.0, 0.0]
    return _height_for_quat(asset, scale, quat) / 2.0, quat


class _CollisionChecker:
    """Collision checker with FCL backend, falling back to AABB distance."""

    def __init__(self):
        self._use_fcl = _HAS_FCL
        if self._use_fcl:
            self._mgr = _fcl_collision.CollisionManager()
        self._placed: list[tuple[str, trimesh.Trimesh, any]] = []

    def min_distance_single(self, mesh: trimesh.Trimesh, transform: any) -> float:
        if not self._placed:
            return float("inf")
        if self._use_fcl:
            return self._mgr.min_distance_single(mesh, transform)
        new_m = mesh.copy()
        new_m.apply_transform(transform)
        new_bounds = new_m.bounds  # [[xmin,ymin,zmin],[xmax,ymax,zmax]]
        min_dist = float("inf")
        for _, _, (lo, hi) in self._placed:
            dx = max(lo[0] - new_bounds[1][0], new_bounds[0][0] - hi[0], 0.0)
            dy = max(lo[1] - new_bounds[1][1], new_bounds[0][1] - hi[1], 0.0)
            dz = max(lo[2] - new_bounds[1][2], new_bounds[0][2] - hi[2], 0.0)
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            min_dist = min(min_dist, d)
        return min_dist

    def add_object(self, name: str, mesh: trimesh.Trimesh, transform: any) -> None:
        if self._use_fcl:
            self._mgr.add_object(name, mesh, transform)
        m = mesh.copy()
        m.apply_transform(transform)
        self._placed.append((name, m, (m.bounds[0].copy(), m.bounds[1].copy())))


class ProgrammaticSceneBackend:
    """Collision-aware scene backend with upright object placement.

    For each object:
    1. Resolve the canonical upright pose from asset metadata.
    2. Sample (x, y) within workspace_xy (or per-object position_range).
    3. Compute z = table_height + table_thickness/2 + upright half-height.
    4. Check collision against already-placed objects via trimesh CollisionManager.
    5. Retry up to max_placement_retries; skip object if all retries fail.

    Missing assets, illegal asset metric scales, and placement failures are warning + skip
    by design so asset-browsing scenes can degrade gracefully. Fixed positions
    are trusted as explicit layouts and bypass random sampling/retry.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def resolve(self, config: SceneConfig, library: AssetLibrary) -> ResolvedScene:
        collision_mgr = _CollisionChecker()
        placed: list[PlacedObject] = []

        table_surface_z = config.table_height + config.table_size[2] / 2.0
        ws_lo, ws_hi = config.workspace_xy
        margin = config.collision_margin
        max_retries = config.max_placement_retries

        for obj_spec in config.objects:
            asset = library.get(obj_spec.asset_query)
            if asset is None:
                logger.warning(
                    "[scene] no asset named %r in library; skipping",
                    obj_spec.asset_query,
                )
                continue

            for i in range(obj_spec.count):
                # The mesh audit only applies to mesh-based geometry; primitive
                # URDFs (fixtures, builtin primitives) carry geometry inline and
                # have no mesh to audit.
                if asset.uses_mesh_geometry:
                    if not asset.metadata.mesh:
                        audit = audit_and_update(asset)
                        mesh_issues = audit["mesh"]["issues"]
                    else:
                        mesh_issues = asset.metadata.mesh.get("issues", [])
                    if mesh_issues:
                        logger.debug(
                            "[scene] skip %s: mesh audit issues=%s",
                            asset.name,
                            mesh_issues,
                        )
                        continue
                scale = float(asset.metadata.metric_scale)
                if scale <= 0.0:
                    logger.debug(
                        "[scene] skip %s: metric_scale must be positive, got %s",
                        asset.name,
                        scale,
                    )
                    continue
                # Fixtures carry an authored upright pose + fixed base; only
                # grasp-targeted objects go through the 24-candidate upright QA.
                if (
                    not asset.is_fixture
                    and asset.metadata.source != "builtin_primitive"
                    and not task_pose_verified(asset, "upright")
                ):
                    logger.debug(
                        "[scene] skip %s: task_poses.upright is missing or unverified",
                        asset.name,
                    )
                    continue
                base_z, quat = _resolve_upright_pose(asset, scale)

                mesh = _load_collision_mesh(asset)
                if mesh is None:
                    continue

                if scale != 1.0:
                    mesh = mesh.copy()
                    mesh.apply_scale(scale)

                z_offset = base_z
                euler = _quat_to_euler(quat)

                # Optional per-scenario yaw about world +Z, composed on top of the
                # asset upright pose (e.g. face a shelf fixture's open side toward
                # the arm). Yaw about Z preserves the Z extent, so base_z stays
                # valid. quat stays [w,x,y,z] (genesis/metadata convention).
                yaw_deg = float(getattr(obj_spec, "yaw_deg", 0.0) or 0.0)
                if yaw_deg:
                    yaw_quat = tf.quaternion_about_axis(
                        math.radians(yaw_deg), (0.0, 0.0, 1.0)
                    )
                    quat = list(tf.quaternion_multiply(yaw_quat, quat))
                    euler = _quat_to_euler(quat)

                logical_name = obj_spec.name_override or f"{asset.name}_{i}"

                if obj_spec.fixed_position:
                    pos = list(obj_spec.fixed_position)
                    transform = tf.compose_matrix(
                        translate=pos,
                        angles=euler,
                    )
                    collision_mgr.add_object(logical_name, mesh, transform)
                    placed.append(
                        PlacedObject(
                            asset=asset,
                            position=pos,
                            rotation=euler,
                            quaternion=quat,
                            name=logical_name,
                        )
                    )
                    continue

                if obj_spec.position_range:
                    xy_lo = obj_spec.position_range[0][:2]
                    xy_hi = obj_spec.position_range[1][:2]
                else:
                    xy_lo = ws_lo
                    xy_hi = ws_hi

                success = False
                for _attempt in range(max_retries):
                    x = self.rng.uniform(xy_lo[0], xy_hi[0])
                    y = self.rng.uniform(xy_lo[1], xy_hi[1])
                    z = table_surface_z + z_offset
                    pos = [x, y, z]

                    transform = tf.compose_matrix(
                        translate=pos,
                        angles=euler,
                    )

                    if len(placed) == 0:
                        dist = margin + 1.0
                    else:
                        dist = collision_mgr.min_distance_single(mesh, transform)

                    if dist >= margin:
                        collision_mgr.add_object(logical_name, mesh, transform)
                        placed.append(
                            PlacedObject(
                                asset=asset,
                                position=pos,
                                rotation=euler,
                                quaternion=quat,
                                name=logical_name,
                            )
                        )
                        success = True
                        break

                if not success:
                    logger.debug(
                        "[scene] could not place %s after %s retries",
                        asset.name,
                        max_retries,
                    )

        table = library.get("table_simple")
        return ResolvedScene(
            config=config,
            placed_objects=placed,
            table_asset=table,
        )
