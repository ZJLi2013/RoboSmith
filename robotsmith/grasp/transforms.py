"""SE3 / quaternion / rotation utilities for grasp planning.

Convention: quaternions are always wxyz (matching GraspPlan and Genesis).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to 3x3 rotation matrix."""
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to wxyz quaternion (Shepperd's method)."""
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float32)
    return q / np.linalg.norm(q)


def pose_matrix(
    pos: np.ndarray,
    quat_wxyz: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build 4x4 homogeneous transform from position + optional wxyz quaternion."""
    T = np.eye(4)
    T[:3, 3] = pos
    if quat_wxyz is not None:
        T[:3, :3] = quat_wxyz_to_matrix(quat_wxyz)
    return T


def transform_points(
    points: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """Apply 4x4 transform to (N, 3) points. Returns (N, 3) float32."""
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([points, ones])
    return (T @ pts_h.T).T[:, :3].astype(np.float32)
