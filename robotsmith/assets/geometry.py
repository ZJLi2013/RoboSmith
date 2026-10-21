"""Geometry helpers shared by asset, grasp, and diagnostic code."""

from __future__ import annotations

from pathlib import Path

import numpy as np


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
    seed: int | None = None,
) -> np.ndarray:
    """Sample a scaled mesh surface into a float32 point cloud; ``seed`` makes the draw reproducible."""
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh")
    mesh.apply_scale(float(scale))
    points, _ = trimesh.sample.sample_surface(mesh, int(n_points), seed=seed)
    return np.asarray(points, dtype=np.float32)
