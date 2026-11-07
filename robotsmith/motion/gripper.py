"""``gripper``: open or close the fingers with the arm held where it is."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from robotsmith.motion.constants import N_ARM_JOINTS as _N_ARM_JOINTS


@dataclass(frozen=True)
class Gripper:
    """``pos`` / ``quat`` are the waypoint's nominal pose; only the finger joints of their IK solution are used."""

    pos: np.ndarray
    quat: np.ndarray
    finger_width: float


@dataclass(frozen=True)
class GripperSegment:
    """``q_target`` is ``q_start`` with the finger joints replaced."""

    q_target: np.ndarray


def expand(move: Gripper, q_start: np.ndarray, solve_ik: Callable) -> GripperSegment:
    q = np.array(q_start, dtype=np.float64)
    finger_q = solve_ik(move.pos, move.quat, move.finger_width)
    q[_N_ARM_JOINTS:] = finger_q[_N_ARM_JOINTS:]
    return GripperSegment(q_target=q)
