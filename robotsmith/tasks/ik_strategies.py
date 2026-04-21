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
    place_z: float = 0.15
    approach_steps: int = 40
    descend_steps: int = 30
    grasp_hold_steps: int = 20
    lift_steps: int = 30
    lift_hold_steps: int = 15
    transport_steps: int = 40
    place_descend_steps: int = 25
    release_steps: int = 15
    retreat_steps: int = 25
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
        place_pos: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Generate a list of joint-space targets.

        Args:
            target_pos: [x, y, z] of the target object (pick location).
            solve_ik: function(pos, quat, finger_pos) -> joint target array.
            home_qpos: home joint configuration.
            params: trajectory timing/geometry parameters.
            z_offset: height offset (e.g. table surface Z for scene mode).
            place_pos: [x, y, z] of the placement target (for pick_and_place).

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

    def plan(self, target_pos, solve_ik, home_qpos, params, z_offset=0.0, place_pos=None):
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
    """reach → grasp → lift → transport → place → release → retreat.

    Extends PickStrategy with a transport + placement phase.
    Requires place_pos to specify where the grasped object should be placed.
    """

    def plan(self, target_pos, solve_ik, home_qpos, params, z_offset=0.0, place_pos=None):
        if place_pos is None:
            raise ValueError("PickAndPlaceStrategy requires place_pos")

        cx, cy = float(target_pos[0]), float(target_pos[1])
        px, py = float(place_pos[0]), float(place_pos[1])

        # Pick phase waypoints
        hover_pos = [cx, cy, params.hover_z + z_offset]
        grasp_pos = [cx, cy, params.grasp_z + z_offset]
        lift_pos = [cx, cy, params.lift_z + z_offset]

        # Place phase waypoints
        transport_pos = [px, py, params.lift_z + z_offset]
        pre_place_pos = [px, py, params.place_z + z_offset]

        q_home = home_qpos.copy()
        q_hover = solve_ik(hover_pos, params.grasp_quat, params.finger_open)
        q_grasp_open = solve_ik(grasp_pos, params.grasp_quat, params.finger_open)
        q_grasp_closed = solve_ik(grasp_pos, params.grasp_quat, params.finger_closed)
        q_lift = solve_ik(lift_pos, params.grasp_quat, params.finger_closed)
        q_transport = solve_ik(transport_pos, params.grasp_quat, params.finger_closed)
        q_pre_place = solve_ik(pre_place_pos, params.grasp_quat, params.finger_closed)
        q_release = solve_ik(pre_place_pos, params.grasp_quat, params.finger_open)
        q_retreat = solve_ik(transport_pos, params.grasp_quat, params.finger_open)

        traj = []
        # Pick phase
        traj += _lerp(q_home, q_hover, params.approach_steps)
        traj += _lerp(q_hover, q_grasp_open, params.descend_steps)
        traj += _lerp(q_grasp_open, q_grasp_closed, params.grasp_hold_steps)
        traj += _lerp(q_grasp_closed, q_lift, params.lift_steps)
        # Transport phase
        traj += _lerp(q_lift, q_transport, params.transport_steps)
        # Place phase
        traj += _lerp(q_transport, q_pre_place, params.place_descend_steps)
        traj += _lerp(q_pre_place, q_release, params.release_steps)
        traj += _lerp(q_release, q_retreat, params.retreat_steps)
        return traj


class StackStrategy(IKStrategy):
    """N rounds of pick_and_place to stack blocks on top of each other.

    Expects target_pos to be a list of N block positions (pick sources).
    place_pos is the XY center of the stack. Each round places at increasing Z.
    """

    def plan(self, target_pos, solve_ik, home_qpos, params, z_offset=0.0, place_pos=None):
        if place_pos is None:
            raise ValueError("StackStrategy requires place_pos (stack center XY)")

        block_positions = target_pos
        if not isinstance(block_positions, list):
            raise ValueError("StackStrategy expects target_pos as list of block positions")

        sx, sy = float(place_pos[0]), float(place_pos[1])
        block_h = 0.04
        pnp = PickAndPlaceStrategy()

        traj = []
        for i, bpos in enumerate(block_positions):
            stack_place_z = params.place_z + i * block_h
            round_params = TrajectoryParams(
                hover_z=params.hover_z,
                grasp_z=params.grasp_z,
                lift_z=params.lift_z,
                place_z=stack_place_z,
                approach_steps=params.approach_steps,
                descend_steps=params.descend_steps,
                grasp_hold_steps=params.grasp_hold_steps,
                lift_steps=params.lift_steps,
                lift_hold_steps=0,
                transport_steps=params.transport_steps,
                place_descend_steps=params.place_descend_steps,
                release_steps=params.release_steps,
                retreat_steps=params.retreat_steps,
                grasp_quat=params.grasp_quat,
                finger_open=params.finger_open,
                finger_closed=params.finger_closed,
            )

            block_pos = np.array(bpos, dtype=np.float64)
            stack_target = np.array([sx, sy, block_pos[2]], dtype=np.float64)

            start_qpos = traj[-1] if traj else home_qpos
            round_traj = pnp.plan(
                block_pos, solve_ik, start_qpos, round_params, z_offset,
                place_pos=stack_target,
            )
            traj += round_traj

        return traj


# ---------- Strategy Registry ----------

IK_STRATEGIES: dict[str, IKStrategy] = {
    "pick": PickStrategy(),
    "pick_and_place": PickAndPlaceStrategy(),
    "stack": StackStrategy(),
}
