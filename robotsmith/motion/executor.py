"""MotionExecutor — generates joint-space trajectories from GraspPlans.

Extracted from the old PickStrategy / PickAndPlaceStrategy / StackStrategy.
The executor knows *nothing* about object categories or grasp semantics;
it only converts 6-DoF waypoints (from GraspPlan) into IK-solved joint
targets with linear interpolation.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from robotsmith.grasp.plan import GraspPlan
from robotsmith.motion.params import MotionParams


def _interpolate(a: np.ndarray, b: np.ndarray, n: int) -> list[np.ndarray]:
    """Linear interpolation in joint space (identical to old _lerp)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return [a + (b - a) * (i + 1) / max(n, 1) for i in range(n)]


class MotionExecutor:
    """Convert GraspPlan(s) + IK solver into joint-space trajectories."""

    def pick(
        self,
        plan: GraspPlan,
        solve_ik: Callable,
        home_qpos: np.ndarray,
        params: MotionParams,
    ) -> list[np.ndarray]:
        """home → pre_grasp → grasp (close fingers) → retreat (lift)."""
        q_home = home_qpos.copy()
        q_pre = solve_ik(plan.pre_grasp_pos, plan.pre_grasp_quat, plan.finger_open)
        q_grasp_open = solve_ik(plan.grasp_pos, plan.grasp_quat, plan.finger_open)
        q_grasp_closed = solve_ik(plan.grasp_pos, plan.grasp_quat, plan.finger_closed)
        q_retreat = solve_ik(plan.retreat_pos, plan.retreat_quat, plan.finger_closed)

        traj: list[np.ndarray] = []
        traj += _interpolate(q_home, q_pre, params.approach_steps)
        traj += _interpolate(q_pre, q_grasp_open, params.descend_steps)
        traj += _interpolate(q_grasp_open, q_grasp_closed, params.grasp_hold_steps)
        traj += _interpolate(q_grasp_closed, q_retreat, params.lift_steps)
        traj += [q_retreat.copy() for _ in range(params.lift_hold_steps)]
        return traj

    def place(
        self,
        place_plan: GraspPlan,
        solve_ik: Callable,
        start_qpos: np.ndarray,
        params: MotionParams,
    ) -> list[np.ndarray]:
        """transport → pre_place → place (open fingers) → retreat.

        Assumes the robot is holding an object (fingers closed) at start_qpos.
        Finger widths come from place_plan: finger_closed while transporting,
        finger_open on release.
        """
        q_start = start_qpos.copy()
        q_transport = solve_ik(place_plan.pre_grasp_pos, place_plan.pre_grasp_quat, place_plan.finger_closed)
        q_pre_place = solve_ik(place_plan.grasp_pos, place_plan.grasp_quat, place_plan.finger_closed)
        q_release = solve_ik(place_plan.grasp_pos, place_plan.grasp_quat, place_plan.finger_open)
        q_retreat = solve_ik(place_plan.retreat_pos, place_plan.retreat_quat, place_plan.finger_open)

        traj: list[np.ndarray] = []
        traj += _interpolate(q_start, q_transport, params.transport_steps)
        traj += _interpolate(q_transport, q_pre_place, params.place_descend_steps)
        traj += _interpolate(q_pre_place, q_release, params.release_steps)
        traj += _interpolate(q_release, q_retreat, params.retreat_steps)
        return traj

    def pick_and_place(
        self,
        pick_plan: GraspPlan,
        place_plan: GraspPlan,
        solve_ik: Callable,
        home_qpos: np.ndarray,
        params: MotionParams,
    ) -> list[np.ndarray]:
        """pick (no lift_hold) → place. Convenience wrapper over pick() + place()."""
        pick_params = MotionParams(
            approach_steps=params.approach_steps,
            descend_steps=params.descend_steps,
            grasp_hold_steps=params.grasp_hold_steps,
            lift_steps=params.lift_steps,
            lift_hold_steps=0,
            transport_steps=params.transport_steps,
            place_descend_steps=params.place_descend_steps,
            release_steps=params.release_steps,
            retreat_steps=params.retreat_steps,
        )
        traj = self.pick(pick_plan, solve_ik, home_qpos, pick_params)
        traj += self.place(place_plan, solve_ik, traj[-1], params)
        return traj
