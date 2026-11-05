"""MotionPlanner protocol — the motion layer's IK seam.

``solve_ik(pos, quat, finger_pos, *, init_qpos)`` returns a 9-dim qpos for one
EE pose. ``GenesisBackend`` wraps the Genesis IK callable; it does not check
collisions.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from robotsmith.motion.types import Waypoint

__all__ = ["Waypoint", "MotionPlanner", "GenesisBackend"]


@runtime_checkable
class MotionPlanner(Protocol):
    """IK backend for the motion layer."""

    def solve_ik(self, pos, quat, finger_pos, *, init_qpos=None) -> np.ndarray:
        """Return a 9-dim qpos for the EE pose."""
        ...


class GenesisBackend:
    """Adapter wrapping the Genesis ``solve_ik`` callable as a MotionPlanner."""

    def __init__(self, solve_ik) -> None:
        self._solve_ik = solve_ik

    def solve_ik(self, pos, quat, finger_pos, *, init_qpos=None):
        return self._solve_ik(pos, quat, finger_pos, init_qpos=init_qpos)
