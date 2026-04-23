"""gen — data generation infrastructure for RoboSmith.

Submodules:
  franka    — Franka Panda constants and EE utilities
  sim_env   — Genesis SimEnv (scene + robot + cameras + IK + reset)
  recorder  — LeRobot dataset recording, evaluation, summary
"""

from robotsmith.gen.franka import (
    JOINT_NAMES, N_DOFS, HOME_QPOS,
    ACTION_NAMES, STATE_NAMES,
    to_numpy, get_ee_state, compute_ee_delta,
)
from robotsmith.gen.sim_env import SimEnv, ensure_display, render_cam
from robotsmith.gen.recorder import (
    create_dataset, record_episode, evaluate_episode, save_summary,
)

__all__ = [
    "JOINT_NAMES", "N_DOFS", "HOME_QPOS",
    "ACTION_NAMES", "STATE_NAMES",
    "to_numpy", "get_ee_state", "compute_ee_delta",
    "SimEnv", "ensure_display", "render_cam",
    "create_dataset", "record_episode", "evaluate_episode", "save_summary",
]
