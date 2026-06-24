"""Geometry helpers shared by asset, grasp, and diagnostic code."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robotsmith.assets.schema import Asset


def rotate_vec_wxyz(quat_wxyz, vec) -> np.ndarray:
    """Rotate a 3-vector by a quaternion in [w,x,y,z] order."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    v = np.asarray(vec, dtype=np.float64)
    # q * v * q^-1 via the standard expansion.
    t = 2.0 * np.cross([x, y, z], v)
    return v + w * t + np.cross([x, y, z], t)


def sample_mesh_pointcloud(
    mesh_path: str | Path,
    n_points: int,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Sample a scaled mesh surface into a float32 point cloud."""
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh")
    mesh.apply_scale(float(scale))
    points, _ = trimesh.sample.sample_surface(mesh, int(n_points))
    return np.asarray(points, dtype=np.float32)


def sample_asset_pointcloud(
    asset: Asset,
    n_points: int,
    *,
    scale: float | None = None,
    object_pos: np.ndarray | None = None,
) -> np.ndarray:
    """Sample an asset's visual/collision mesh at its metric scene scale."""
    mesh_path = asset.visual_mesh or asset.collision_mesh
    if mesh_path is None:
        raise ValueError(f"Asset {asset.name!r} has no visual or collision mesh")

    metric_scale = float(asset.metadata.metric_scale if scale is None else scale)
    points = sample_mesh_pointcloud(mesh_path, n_points, scale=metric_scale)
    if object_pos is not None:
        from robotsmith.grasp.transforms import pose_matrix, transform_points

        points = transform_points(
            points,
            pose_matrix(np.asarray(object_pos, dtype=np.float32)),
        )
    return points
