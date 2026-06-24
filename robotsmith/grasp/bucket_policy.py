"""Policy-bucket classification for grasp candidates."""

from __future__ import annotations

import numpy as np

from robotsmith.grasp.planner import GraspPlan


def pose_axis_metrics(plan: GraspPlan) -> tuple[float, float, float]:
    approach = np.asarray(
        plan.metadata.get("approach_dir", [0.0, 0.0, 1.0]),
        dtype=np.float64,
    )
    finger = np.asarray(
        plan.metadata.get("finger_dir", [1.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    approach /= max(float(np.linalg.norm(approach)), 1e-12)
    finger /= max(float(np.linalg.norm(finger)), 1e-12)
    palm = np.cross(approach, finger)
    palm /= max(float(np.linalg.norm(palm)), 1e-12)
    finger_z = abs(float(finger[2]))
    palm_z = abs(float(palm[2]))
    path_z = abs(float(approach[2]))
    return finger_z, palm_z, path_z


def assign_policy_bucket(plan: GraspPlan, table_z: float = 0.0) -> str:
    del table_z  # Reserved for future height-aware bucket policies.

    finger_z, palm_z, path_z = pose_axis_metrics(plan)
    axis_horizontalness = max(finger_z, palm_z)

    if path_z >= 0.70:
        approach_bin = "top_down"
    elif path_z <= 0.35:
        approach_bin = "side"
    else:
        approach_bin = "oblique"

    if axis_horizontalness <= 0.35:
        orientation_bin = "axis_horizontal"
    elif axis_horizontalness <= 0.70:
        orientation_bin = "axis_tilted"
    else:
        orientation_bin = "axis_vertical"

    policy_bucket = f"{approach_bin}_{orientation_bin}"
    plan.metadata["finger_axis_z_abs"] = finger_z
    plan.metadata["palm_axis_z_abs"] = palm_z
    plan.metadata["path_axis_z_abs"] = path_z
    plan.metadata["approach_bin"] = approach_bin
    plan.metadata["orientation_bin"] = orientation_bin
    plan.metadata["axis_horizontalness"] = axis_horizontalness
    plan.metadata["policy_bucket"] = policy_bucket
    return policy_bucket
