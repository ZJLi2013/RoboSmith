"""Candidate-pose screening against the support surface.

Mirrors the intent of GraspGenX ``graspgenx/utils/collision_filter.py``, which
asks whether the gripper collision mesh comes within ``collision_threshold`` of
any scene point. The support surface is the one obstacle known analytically
rather than as points (``support_z`` is derived per object in
``LearnedGraspPlanner.plan``), so comparing gripper samples against that plane
is both exact and cheaper than sampling it and running a nearest-neighbour search.

Neighbouring objects are not screened.
"""

from __future__ import annotations

import numpy as np


def support_clearance_mask(
    grasp_poses: np.ndarray,
    gripper_points: np.ndarray,
    *,
    support_z: float,
    clearance: float,
) -> np.ndarray:
    """Mask of poses whose gripper geometry clears the support plane.

    Args:
        grasp_poses: ``(K, 4, 4)`` world-frame grasp poses.
        gripper_points: ``(M, 3)`` samples of the gripper collision mesh, in the
            grasp-pose frame (GraspGen's convention: approach ``+z``, fingers
            closing along ``x``).
        support_z: world height of the surface the object rests on.
        clearance: minimum gripper-to-surface distance, metres.

    Returns:
        ``(K,)`` bool array; True == every gripper sample sits at least
        ``clearance`` above ``support_z``.
    """
    poses = np.asarray(grasp_poses, dtype=np.float64)
    points = np.asarray(gripper_points, dtype=np.float64)
    if len(poses) == 0:
        return np.zeros((0,), dtype=bool)
    if len(points) == 0:
        return np.ones((len(poses),), dtype=bool)

    # Only world z matters: z_world[m, k] = R[k][2, :] · points[m] + t[k][2]
    z_world = points @ poses[:, 2, :3].T + poses[:, 2, 3]
    return z_world.min(axis=0) >= float(support_z) + float(clearance)
