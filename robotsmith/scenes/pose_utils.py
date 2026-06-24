"""Utilities for reading semantic scene poses from asset metadata."""

from __future__ import annotations

import numpy as np

from robotsmith.assets import Asset

IDENTITY_QUAT_WXYZ = [1.0, 0.0, 0.0, 0.0]


def task_pose_quat(asset: Asset, name: str) -> list[float] | None:
    """Return a named semantic pose quaternion from asset metadata, if present."""
    poses = getattr(asset.metadata, "task_poses", {}) or {}
    pose = poses.get(name)
    if not isinstance(pose, dict):
        return None
    quat = pose.get("quat_object") or pose.get("quat") or pose.get("quaternion")
    if quat is None:
        return None
    quat = [float(v) for v in quat]
    if len(quat) != 4:
        raise ValueError(f"{asset.name} task pose {name!r} must have 4D quaternion")
    return quat


def task_pose_verified(asset: Asset, name: str) -> bool:
    """Return whether a named task pose is explicitly verified for runtime use."""
    poses = getattr(asset.metadata, "task_poses", {}) or {}
    pose = poses.get(name)
    return isinstance(pose, dict) and pose.get("verified") is True


def task_upright_quat(asset: Asset) -> np.ndarray:
    """Return the canonical upright object quaternion as a WXYZ numpy array."""
    return np.asarray(
        task_pose_quat(asset, "upright") or IDENTITY_QUAT_WXYZ,
        dtype=np.float64,
    )
