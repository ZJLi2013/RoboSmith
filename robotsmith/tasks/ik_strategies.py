"""IK waypoint generation strategies, dispatched by TaskSpec.ik_strategy.

Each strategy is a class with a `plan()` method that returns a list of
joint-space targets given a robot context and target object state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class TrajectoryParams:
    """Shared parameters for trajectory generation."""
    hover_z: float = 0.25
    grasp_z: float = 0.135
    lift_z: float = 0.30
    approach_steps: int = 40
    descend_steps: int = 30
    grasp_hold_steps: int = 20
    lift_steps: int = 30
    lift_hold_steps: int = 15
    grasp_quat: np.ndarray = field(
        default_factory=lambda: np.array([0, 1, 0, 0], dtype=np.float32)
    )
    finger_open: float = 0.04
    finger_closed: float = 0.01


class IKStrategy(ABC):
    """Base class for IK waypoint strategies."""

    @abstractmethod
    def plan(
        self,
        target_pos: np.ndarray,
        solve_ik: Callable,
        home_qpos: np.ndarray,
        params: TrajectoryParams,
        z_offset: float = 0.0,
    ) -> list[np.ndarray]:
        """Generate a list of joint-space targets.

        Args:
            target_pos: [x, y, z] of the target object.
            solve_ik: function(pos, quat, finger_pos) -> joint target array.
            home_qpos: home joint configuration.
            params: trajectory timing/geometry parameters.
            z_offset: height offset (e.g. table surface Z for scene mode).

        Returns:
            List of joint-space target arrays.
        """
        ...


def _lerp(a: np.ndarray, b: np.ndarray, n: int) -> list[np.ndarray]:
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    return [a + (b - a) * (i + 1) / max(n, 1) for i in range(n)]


class PickStrategy(IKStrategy):
    """reach → pre-grasp → grasp → lift."""

    def plan(self, target_pos, solve_ik, home_qpos, params, z_offset=0.0):
        cx, cy = float(target_pos[0]), float(target_pos[1])

        hover_pos = [cx, cy, params.hover_z + z_offset]
        grasp_pos = [cx, cy, params.grasp_z + z_offset]
        lift_pos = [cx, cy, params.lift_z + z_offset]

        q_home = home_qpos.copy()
        q_hover = solve_ik(hover_pos, params.grasp_quat, params.finger_open)
        q_grasp_open = solve_ik(grasp_pos, params.grasp_quat, params.finger_open)
        q_grasp_closed = solve_ik(grasp_pos, params.grasp_quat, params.finger_closed)
        q_lift = solve_ik(lift_pos, params.grasp_quat, params.finger_closed)

        traj = []
        traj += _lerp(q_home, q_hover, params.approach_steps)
        traj += _lerp(q_hover, q_grasp_open, params.descend_steps)
        traj += _lerp(q_grasp_open, q_grasp_closed, params.grasp_hold_steps)
        traj += _lerp(q_grasp_closed, q_lift, params.lift_steps)
        traj += [q_lift.copy() for _ in range(params.lift_hold_steps)]
        return traj


class PickAndPlaceStrategy(IKStrategy):
    """pick + move → pre-place → place → release. (Stub for future use.)"""

    def plan(self, target_pos, solve_ik, home_qpos, params, z_offset=0.0):
        raise NotImplementedError("pick_and_place strategy not yet implemented")


# ---------- Strategy Registry ----------

IK_STRATEGIES: dict[str, IKStrategy] = {
    "pick": PickStrategy(),
    "pick_and_place": PickAndPlaceStrategy(),
}
