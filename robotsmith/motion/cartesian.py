"""Shared Cartesian IK segment helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from robotsmith.motion.constants import N_ARM_JOINTS as _N_ARM_JOINTS

DEFAULT_CARTESIAN_STEP_M = 0.01
DEFAULT_CARTESIAN_MAX_STEPS = 24
DEFAULT_CARTESIAN_MAX_JOINT_STEP = 0.8


@dataclass(frozen=True)
class CartesianIKPath:
    """IK solutions for a Cartesian micro-waypoint segment."""

    path: list[np.ndarray]
    max_joint_step: float
    ok: bool


def solve_cartesian_ik_path(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    quat: np.ndarray,
    finger_width: float,
    q_start: np.ndarray,
    solve_ik: Callable,
    *,
    step_m: float = DEFAULT_CARTESIAN_STEP_M,
    max_steps: int = DEFAULT_CARTESIAN_MAX_STEPS,
    max_joint_step: float = DEFAULT_CARTESIAN_MAX_JOINT_STEP,
    raise_on_discontinuity: bool = False,
    error_label: str = "cartesian IK path",
) -> CartesianIKPath:
    """Solve Cartesian micro-waypoints with seed continuity and joint-step gating."""
    start = np.asarray(start_pos, dtype=np.float64)
    end = np.asarray(end_pos, dtype=np.float64)
    distance = float(np.linalg.norm(end - start))
    n_steps = max(2, min(math.ceil(distance / step_m), max_steps))

    path: list[np.ndarray] = []
    q_prev = np.asarray(q_start, dtype=np.float64)
    observed_max_step = 0.0
    ok = True

    for t in np.linspace(1.0 / n_steps, 1.0, n_steps):
        pos = (1.0 - t) * start + t * end
        q = np.asarray(
            solve_ik(pos, quat, finger_width, init_qpos=q_prev),
            dtype=np.float64,
        )
        step = float(np.max(np.abs(q[:_N_ARM_JOINTS] - q_prev[:_N_ARM_JOINTS])))
        observed_max_step = max(observed_max_step, step)
        if step > max_joint_step:
            ok = False
            if raise_on_discontinuity:
                raise RuntimeError(
                    f"{error_label} IK discontinuity: "
                    f"step={step:.3f} rad > {max_joint_step:.3f} rad"
                )
        path.append(q)
        q_prev = q

    return CartesianIKPath(
        path=path,
        max_joint_step=observed_max_step,
        ok=ok,
    )
