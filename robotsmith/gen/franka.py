"""Franka Panda constants and EE utility functions.

Pure numpy — no Genesis / torch dependency.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# ---- Joint configuration ----
JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "joint5", "joint6", "joint7",
    "finger_joint1", "finger_joint2",
]
N_DOFS = len(JOINT_NAMES)

HOME_QPOS = np.array(
    [0, -0.3, 0, -2.2, 0, 2.0, 0.79, 0.04, 0.04], dtype=np.float32
)
KP = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100], dtype=np.float32)
KV = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10], dtype=np.float32)
FORCE_LOWER = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100], dtype=np.float32)
FORCE_UPPER = np.array([87, 87, 87, 87, 12, 12, 12, 100, 100], dtype=np.float32)

# ---- Observation / Action names ----
ACTION_NAMES = [
    "delta_x", "delta_y", "delta_z",
    "delta_ax", "delta_ay", "delta_az",
    "gripper",
]
STATE_NAMES = [
    "ee_x", "ee_y", "ee_z",
    "ee_ax", "ee_ay", "ee_az",
    "gripper_left", "gripper_right",
]


# ---- Pure-numpy helpers ----

def to_numpy(t) -> np.ndarray:
    arr = t.cpu().numpy() if hasattr(t, "cpu") else np.array(t)
    return arr[0] if arr.ndim > 1 else arr


def quat_to_axangle(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to axis-angle (3D compact rotvec)."""
    r = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return r.as_rotvec().astype(np.float32)


def get_ee_state(ee_link, finger_dof_vals: np.ndarray) -> np.ndarray:
    """Return 8D EE state: [pos(3), axangle(3), gripper(2)]."""
    pos = to_numpy(ee_link.get_pos()).astype(np.float32)
    quat = to_numpy(ee_link.get_quat()).astype(np.float32)
    axangle = quat_to_axangle(quat)
    gripper = finger_dof_vals[:2].astype(np.float32)
    return np.concatenate([pos, axangle, gripper])


def compute_ee_delta(
    prev_state: np.ndarray,
    curr_state: np.ndarray,
    gripper_cmd: float,
) -> np.ndarray:
    """Compute 7D EE delta action: [delta_pos(3), delta_axangle(3), gripper(1)]."""
    delta_pos = curr_state[:3] - prev_state[:3]
    r_prev = Rotation.from_rotvec(prev_state[3:6])
    r_curr = Rotation.from_rotvec(curr_state[3:6])
    delta_rot = (r_curr * r_prev.inv()).as_rotvec()
    return np.concatenate([
        delta_pos.astype(np.float32),
        delta_rot.astype(np.float32),
        np.array([gripper_cmd], dtype=np.float32),
    ])
