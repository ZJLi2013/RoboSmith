"""Scene generation backends: programmatic (MVP) + SceneSmith (future stub)."""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

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
from robotsmith.assets.schema import Asset
from robotsmith.scenes.config import SceneConfig, ObjectPlacement


@dataclass
class PlacedObject:
    """An asset placed in a scene with a concrete pose."""

    asset: Asset
    position: list[float]
    rotation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    quaternion: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    """Orientation as [w, x, y, z] quaternion (from stable pose)."""
    scale: float = 1.0
    """Uniform scale applied to the URDF mesh."""
    name: str = ""
    """Logical name for skill-target matching (e.g. 'cube', 'bowl')."""

    @property
    def object_height_m(self) -> float:
        """World-frame height (Z extent) after applying stable-pose rotation and scale.

        For axis-aligned objects (identity quat) this equals ``size_cm[2] * scale``.
        For rotated objects like bowls (Y-up mesh rotated to Z-up), the
        quaternion maps the correct local axis onto world Z.
        """
        sx, sy, sz = [v / 100.0 * self.scale for v in self.asset.metadata.size_cm]
        w, x, y, z = self.quaternion
        rmat = tf.quaternion_matrix([w, x, y, z])[:3, :3]
        import numpy as np
        corners = np.array([
            [dx * sx / 2, dy * sy / 2, dz * sz / 2]
            for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)
        ])
        rotated = corners @ rmat.T
        return float(rotated[:, 2].max() - rotated[:, 2].min())


@dataclass
class ResolvedScene:
    """A fully resolved scene: concrete objects with poses, ready for sim loading."""

    config: SceneConfig
    placed_objects: list[PlacedObject] = field(default_factory=list)
    table_asset: Optional[Asset] = None
    plane_asset: Optional[Asset] = None

    def summary(self) -> str:
        lines = [f"Scene: {self.config.name} ({len(self.placed_objects)} objects)"]
        for po in self.placed_objects:
            pos = [round(v, 3) for v in po.position]
            scale_str = f" scale={po.scale:.3f}" if po.scale != 1.0 else ""
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


def _pick_stable_pose(
    rng: random.Random, asset: Asset
) -> tuple[float, list[float]]:
    """Select a stable pose (z_offset, quaternion) from asset metadata."""
    poses = asset.metadata.stable_poses
    if not poses:
        half_h = asset.metadata.size_cm[2] / 200.0
        return half_h, [1.0, 0.0, 0.0, 0.0]

    probs = [p.get("probability", 1.0) for p in poses]
    total = sum(probs)
    r = rng.random() * total
    cumul = 0.0
    chosen = poses[0]
    for p in poses:
        cumul += p.get("probability", 1.0)
        if r <= cumul:
            chosen = p
            break
    return chosen["z"], list(chosen["quat"])


def _quat_to_euler(quat: list[float]) -> list[float]:
    """Convert [w,x,y,z] quaternion to [roll, pitch, yaw] Euler angles."""
    mat = tf.quaternion_matrix(quat)
    return list(tf.euler_from_matrix(mat, axes="sxyz"))


class _CollisionChecker:
    """Collision checker with FCL backend, falling back to AABB distance."""

    def __init__(self):
        self._use_fcl = _HAS_FCL
        if self._use_fcl:
            self._mgr = _fcl_collision.CollisionManager()
        self._placed: list[tuple[str, trimesh.Trimesh, any]] = []

    def min_distance_single(
        self, mesh: trimesh.Trimesh, transform: any
    ) -> float:
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

    def add_object(
        self, name: str, mesh: trimesh.Trimesh, transform: any
    ) -> None:
        if self._use_fcl:
            self._mgr.add_object(name, mesh, transform)
        m = mesh.copy()
        m.apply_transform(transform)
        self._placed.append((name, m, (m.bounds[0].copy(), m.bounds[1].copy())))


class SceneBackend(ABC):
    """Abstract base class for scene generation backends."""

    @abstractmethod
    def resolve(self, config: SceneConfig, library: AssetLibrary) -> ResolvedScene:
        """Resolve a SceneConfig into a ResolvedScene with concrete assets and poses."""
        ...


class ProgrammaticSceneBackend(SceneBackend):
    """Collision-aware scene backend with stable-pose sampling.

    For each object:
    1. Sample a stable resting pose (z offset + quaternion) from metadata.
    2. Sample (x, y) within workspace_xy (or per-object position_range).
    3. Compute z = table_height + table_thickness/2 + stable_pose.z.
    4. Check collision against already-placed objects via trimesh CollisionManager.
    5. Retry up to max_placement_retries; skip object if all retries fail.
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
            exact = library.get(obj_spec.asset_query)
            if exact is not None:
                assets = [exact]
            else:
                assets = library.search(obj_spec.asset_query, top_k=obj_spec.count)
            if not assets:
                print(f"[scene] WARNING: no asset for query={obj_spec.asset_query!r}")
                continue

            scale = obj_spec.scale

            for i in range(obj_spec.count):
                asset = assets[i % len(assets)]
                mesh = _load_collision_mesh(asset)
                if mesh is None:
                    continue

                if scale != 1.0:
                    mesh = mesh.copy()
                    mesh.apply_scale(scale)

                z_offset, quat = _pick_stable_pose(self.rng, asset)
                if scale != 1.0:
                    z_offset *= scale
                euler = _quat_to_euler(quat)

                logical_name = obj_spec.name_override or f"{asset.name}_{i}"

                if obj_spec.fixed_position:
                    pos = list(obj_spec.fixed_position)
                    transform = tf.compose_matrix(
                        translate=pos,
                        angles=euler,
                    )
                    collision_mgr.add_object(logical_name, mesh, transform)
                    placed.append(PlacedObject(
                        asset=asset, position=pos,
                        rotation=euler, quaternion=quat,
                        scale=scale, name=logical_name,
                    ))
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
                        placed.append(PlacedObject(
                            asset=asset, position=pos,
                            rotation=euler, quaternion=quat,
                            scale=scale, name=logical_name,
                        ))
                        success = True
                        break

                if not success:
                    print(
                        f"[scene] WARNING: could not place {asset.name} "
                        f"after {max_retries} retries"
                    )

        table = library.get("table_simple")
        plane = library.get("plane")

        return ResolvedScene(
            config=config,
            placed_objects=placed,
            table_asset=table,
            plane_asset=plane,
        )


class SceneSmithBackend(SceneBackend):
    """Stub for future SceneSmith integration (LLM -> layout -> Drake SDF).

    Not implemented in MVP. Will accept natural language scene descriptions
    and produce object placement layouts via SceneSmith's LLM planning.
    """

    def resolve(self, config: SceneConfig, library: AssetLibrary) -> ResolvedScene:
        raise NotImplementedError(
            "SceneSmithBackend is a future extension. "
            "Use ProgrammaticSceneBackend for MVP."
        )
