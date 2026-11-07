"""``move_linear``: a Cartesian straight line expanded into joint space by seed-continuous IK."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal

import numpy as np

from robotsmith.motion.constants import N_ARM_JOINTS as _N_ARM_JOINTS

DEFAULT_STEP_M = 0.01
DEFAULT_MAX_STEPS = 24
DEFAULT_MAX_JOINT_STEP = 0.8


@dataclass(frozen=True)
class MoveLinear:
    """``start_pos`` is the nominal end-effector position the line starts from (the previous waypoint), not
    the pose of the IK seed. ``on_jump="refuse"`` raises on an IK branch flip; ``"keep"`` returns the path
    with ``continuous=False`` and leaves the judgment to the gate."""

    start_pos: np.ndarray
    end_pos: np.ndarray
    quat: np.ndarray
    finger_width: float
    on_jump: Literal["refuse", "keep"]
    label: str = "cartesian IK path"


@dataclass(frozen=True)
class LinearSegment:
    """One IK solution per micro-waypoint; ``path[-1]`` is the predicted end configuration."""

    path: list[np.ndarray]
    max_joint_step: float
    continuous: bool


def expand(
    move: MoveLinear,
    q_start: np.ndarray,
    solve_ik: Callable,
    *,
    step_m: float = DEFAULT_STEP_M,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_joint_step: float = DEFAULT_MAX_JOINT_STEP,
) -> LinearSegment:
    start = np.asarray(move.start_pos, dtype=np.float64)
    end = np.asarray(move.end_pos, dtype=np.float64)
    distance = float(np.linalg.norm(end - start))
    n_steps = max(2, min(math.ceil(distance / step_m), max_steps))

    path: list[np.ndarray] = []
    q_prev = np.asarray(q_start, dtype=np.float64)
    observed_max_step = 0.0
    continuous = True

    for t in np.linspace(1.0 / n_steps, 1.0, n_steps):
        pos = (1.0 - t) * start + t * end
        q = np.asarray(
            solve_ik(pos, move.quat, move.finger_width, init_qpos=q_prev),
            dtype=np.float64,
        )
        step = float(np.max(np.abs(q[:_N_ARM_JOINTS] - q_prev[:_N_ARM_JOINTS])))
        observed_max_step = max(observed_max_step, step)
        if step > max_joint_step:
            continuous = False
            if move.on_jump == "refuse":
                raise RuntimeError(
                    f"{move.label} IK discontinuity: "
                    f"step={step:.3f} rad > {max_joint_step:.3f} rad"
                )
        path.append(q)
        q_prev = q

    return LinearSegment(path=path, max_joint_step=observed_max_step, continuous=continuous)
