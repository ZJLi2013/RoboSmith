"""MotionPlanner protocol — pluggable IK / collision-free motion backend.

``MotionPlanner`` is the motion layer's seam, with two operations:

- ``solve_ik(pos, quat, finger_pos, *, init_qpos, world)`` — a 9-dim qpos for one
  EE pose; with ``world`` (collision primitives) the solved endpoint is
  collision-free.
- ``plan_motion(q_start, waypoints, *, world)`` — a whole collision-free joint
  trajectory. A collision-free endpoint does not imply a collision-free path
  between endpoints; trajopt keeps the arm links out of the drawer along the way.

Two backends implement the protocol:

- ``RocRoboBackend`` (rocrobo_backend.py) — collision-aware; implements both.
- ``GenesisBackend`` — collision-blind fallback; ``solve_ik`` ignores ``world``
  and ``plan_motion`` returns ``success=False``, so the caller uses per-waypoint
  ``solve_ik`` + straight-line interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from robotsmith.motion.types import Waypoint

__all__ = ["Waypoint", "PlanResult", "MotionPlanner", "GenesisBackend"]


@dataclass
class PlanResult:
    """Result of ``plan_motion``.

    ``trajectory`` is a list of 9-dim qpos already resampled to the caller's
    control dt (ready to replay step-by-step); empty when ``success`` is False.
    """

    success: bool
    trajectory: list[np.ndarray] = field(default_factory=list)
    reason: str = ""
    quality: dict = field(default_factory=dict)


@runtime_checkable
class MotionPlanner(Protocol):
    """Pluggable IK / motion backend (rocRobo or Genesis fallback).

    ``collision_aware`` tells callers whether this backend actually avoids
    ``world`` obstacles. The collision-blind ``GenesisBackend`` sets it False so
    the orchestration layer skips building a collision world (and the hover
    authoring that rides on it); a True backend (rocRobo) gets the full path.
    """

    collision_aware: bool

    def solve_ik(
        self,
        pos,
        quat,
        finger_pos,
        *,
        init_qpos=None,
        world: Sequence[dict] | None = None,
        attach: dict | None = None,
    ) -> np.ndarray:
        """Return a 9-dim qpos for the EE pose; collision-free if ``world`` given.

        ``attach`` is an optional attached-collision-object (a grasped payload's
        spheres + ``T_ee_obj``) so the held object is checked against ``world``.
        """
        ...

    def plan_motion(
        self,
        q_start,
        waypoints: Sequence[Waypoint],
        *,
        world: Sequence[dict] | None = None,
        control_dt: float | None = None,
        attach: dict | None = None,
    ) -> PlanResult:
        """Return a collision-free joint trajectory through ``waypoints``.

        ``attach`` (grasped-payload spheres + ``T_ee_obj``) rides on the EE so the
        carried object stays clear of ``world`` along the path, not just the arm.
        """
        ...


class GenesisBackend:
    """Adapter wrapping the Genesis ``solve_ik`` callable as a MotionPlanner.

    This is the fallback: ``solve_ik`` ignores ``world`` (collision-blind), and
    ``plan_motion`` is unsupported (returns ``success=False``) so callers fall
    back to the legacy straight-line execution path.
    """

    collision_aware = False

    def __init__(self, solve_ik) -> None:
        self._solve_ik = solve_ik

    def solve_ik(self, pos, quat, finger_pos, *, init_qpos=None, world=None, attach=None):
        # Genesis IK is collision-blind; ``world``/``attach`` accepted but ignored.
        return self._solve_ik(pos, quat, finger_pos, init_qpos=init_qpos)

    def plan_motion(self, q_start, waypoints, *, world=None, control_dt=None, attach=None):
        return PlanResult(success=False, reason="genesis-backend-no-trajopt")
