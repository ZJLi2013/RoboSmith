"""sim — simulation + recording substrate for RoboSmith.

The lower layer of the data-generation stack: run one episode in the simulator
and record it. The rollout pipeline (``robotsmith.rollout``) drives this to
produce datasets.

Submodules:
  franka    — Franka Panda constants and EE utilities
  sim_env   — Genesis SimEnv (scene + robot + cameras + IK + reset)
  recorder  — LeRobot dataset recording, evaluation, summary
"""

from robotsmith.sim.franka import (
    JOINT_NAMES, N_DOFS, HOME_QPOS,
    ACTION_NAMES, STATE_NAMES,
    to_numpy, get_ee_state, compute_ee_delta,
)

__all__ = [
    "JOINT_NAMES", "N_DOFS", "HOME_QPOS",
    "ACTION_NAMES", "STATE_NAMES",
    "to_numpy", "get_ee_state", "compute_ee_delta",
    "SimEnv", "ensure_display", "render_cam",
    "create_dataset", "record_episode", "evaluate_episode", "save_summary",
]


def __getattr__(name: str):
    if name in {"SimEnv", "ensure_display", "render_cam"}:
        from importlib import import_module
        sim_env = import_module("robotsmith.sim.sim_env")
        return getattr(sim_env, name)
    if name in {"create_dataset", "record_episode", "evaluate_episode", "save_summary"}:
        from importlib import import_module
        recorder = import_module("robotsmith.sim.recorder")
        return getattr(recorder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
