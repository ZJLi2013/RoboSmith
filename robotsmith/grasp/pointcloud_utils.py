"""Point cloud utilities for learned grasp planners.

Converts asset meshes to point clouds for GraspGen / similar models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from robotsmith.assets.schema import Asset
from robotsmith.grasp.transforms import pose_matrix, transform_points


def mesh_to_pointcloud(
    mesh_path: Path | str,
    n_points: int = 8192,
    scale: float = 1.0,
) -> np.ndarray:
    """Sample points from a mesh file surface.

    Returns (n_points, 3) in the mesh's local coordinate frame.
    """
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if scale != 1.0:
        mesh.apply_scale(scale)
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return np.asarray(points, dtype=np.float32)


def asset_to_pointcloud(
    asset: Asset,
    n_points: int = 8192,
    object_pos: Optional[np.ndarray] = None,
    object_quat: Optional[np.ndarray] = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Convert an Asset's mesh to a point cloud.

    If object_pos / object_quat are provided, returns points in world frame.
    Otherwise returns points in the mesh's local frame.

    For assets without an external mesh (e.g. URDF inline <box>),
    falls back to a box approximation from metadata.size_cm.

    Args:
        asset: Asset with visual_mesh or collision_mesh.
        n_points: Number of surface samples.
        object_pos: World-frame position (3,). If None, returns local frame.
        object_quat: World-frame orientation as wxyz (4,). If None, identity.
        scale: Mesh scale factor (matches Genesis URDF scale).

    Returns:
        (n_points, 3) float32 point cloud.
    """
    mesh_path = asset.visual_mesh or asset.collision_mesh

    if mesh_path is not None and Path(mesh_path).exists():
        pc = mesh_to_pointcloud(mesh_path, n_points, scale=scale)
    else:
        size_cm = asset.metadata.size_cm
        extents = np.array(size_cm, dtype=np.float32) / 100.0 * scale
        box = trimesh.creation.box(extents=extents)
        points, _ = trimesh.sample.sample_surface(box, n_points)
        pc = np.asarray(points, dtype=np.float32)

    if object_pos is not None:
        T = pose_matrix(np.asarray(object_pos), object_quat)
        pc = transform_points(pc, T)

    return pc
