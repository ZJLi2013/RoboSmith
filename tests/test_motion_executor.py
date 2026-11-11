"""Tests for MotionExecutor velocity-adaptive step calculation."""

import math

import numpy as np
import pytest

from robotsmith.motion.chain import _compute_steps
from robotsmith.execution.executor import MotionExecutor
from robotsmith.execution.gate import Holding
from robotsmith.execution.meta_actions import pick_actions, place_actions
from robotsmith.scenario.load import pick_part, place_part
from robotsmith.motion.plan_state import PlanState
from robotsmith.motion.planner import GenesisBackend
from robotsmith.motion.params import MotionParams
from robotsmith.grasp.planner import GraspPlan

PICK, PLACE = pick_part("die"), place_part("die", "mat")

DT = 1.0 / 30  # Genesis control dt
IDENTITY = np.array([1, 0, 0, 0], dtype=np.float64)
HELD = Holding("obj", np.eye(4))


# -- _compute_steps --

class TestComputeSteps:
    def test_large_move_gets_many_steps(self):
        q1 = np.zeros(9)
        q1[0] = 2.0
        steps, dist = _compute_steps(np.zeros(9), q1, vel=1.0, dt=DT)
        assert dist == pytest.approx(2.0)
        assert steps == math.ceil(2.0 / (1.0 * DT))

    @pytest.mark.parametrize("delta", [0.0, 0.001])
    def test_small_move_respects_min_steps(self, delta):
        q1 = np.zeros(9)
        q1[0] = delta
        steps, _ = _compute_steps(np.zeros(9), q1, vel=1.0, dt=DT, min_steps=5)
        assert steps == 5

    def test_slower_vel_gives_more_steps(self):
        q1 = np.zeros(9)
        q1[2] = 0.5
        s_fast, _ = _compute_steps(np.zeros(9), q1, vel=1.0, dt=DT)
        s_slow, _ = _compute_steps(np.zeros(9), q1, vel=0.3, dt=DT)
        assert s_slow > s_fast

    def test_ignores_finger_joints(self):
        """Only arm joints (first 7) count for distance."""
        q1 = np.zeros(9)
        q1[7] = q1[8] = 0.04
        steps, dist = _compute_steps(np.zeros(9), q1, vel=1.0, dt=DT, min_steps=5)
        assert dist < 1e-6
        assert steps == 5


# -- the pick / place action lists --

def _trivial_ik(pos, quat, finger_width, init_qpos=None, **_kw):
    q = np.zeros(9)
    q[0], q[1], q[2] = pos[0], pos[1], pos[2]
    q[7] = q[8] = finger_width
    return q


def _pick_plan():
    return GraspPlan(
        grasp_pos=np.array([0.5, 0.0, 0.1]),
        grasp_quat=IDENTITY,
        pre_grasp_pos=np.array([0.5, 0.0, 0.2]),
        pre_grasp_quat=IDENTITY,
        finger_open=0.04,
        finger_closed=0.0,
    )


def _pick(exe, planner, params):
    plan = _pick_plan()
    pre = (plan.pre_grasp_pos, plan.pre_grasp_quat)
    q = _trivial_ik(*pre, 0.04)
    return exe.run("pick", pick_actions(PICK, plan, q, pre, (), params, planner.solve_ik, grasp=HELD), planner,
                   PlanState(q), params)


def _segment_frames(result, idx):
    start, end = result.segments[idx]["span"]
    return np.asarray(result.trajectory)[start:end]


class TestPickMotionChain:
    def test_finger_seg_locks_arm_joints(self):
        """IK that jumps when fingers close must not move the arm during the close."""
        def unstable_ik(pos, quat, finger_width, init_qpos=None):
            q = _trivial_ik(pos, quat, finger_width)
            if finger_width < 0.02:
                q[3] = 1.5
            return q

        exe = MotionExecutor()
        traj = _pick(exe, GenesisBackend(unstable_ik), MotionParams(finger_steps=10))
        arm = _segment_frames(traj, 3)[:, :7]
        np.testing.assert_allclose(arm, np.tile([0.5, 0.0, 0.10, 0, 0, 0, 0], (len(arm), 1)),
                                   atol=1e-6)


class TestPlaceMotionChain:
    def test_place_release_locks_arm_and_records_trace(self):
        plan = GraspPlan(
            grasp_pos=np.array([0.5, 0.0, 0.1]),
            grasp_quat=IDENTITY,
            pre_grasp_pos=np.array([0.5, 0.0, 0.2]),
            pre_grasp_quat=IDENTITY,
            retreat_pos=np.array([0.5, 0.0, 0.3]),
            retreat_quat=IDENTITY,
            finger_open=0.04,
            finger_closed=0.0,
        )

        def unstable_open_ik(pos, quat, finger_width, init_qpos=None):
            q = _trivial_ik(pos, quat, finger_width)
            if finger_width > 0.02:
                q[3] = 1.5
            return q

        params = MotionParams(transport_steps=3, release_steps=5)
        exe = MotionExecutor()
        traj = exe.run("place", place_actions(PLACE, plan, params), GenesisBackend(unstable_open_ik),
                       PlanState(np.zeros(9), HELD), params)

        segs = exe.last_motion_trace["segments"]
        labels = [s["label"] for s in segs]
        assert labels == ["transport", "descend", "release", "retreat"]
        release_idx = labels.index("release")
        assert segs[release_idx]["finger_only"] is True

        # If release re-solved IK while the fingers open, joint 3 would whip to 1.5.
        release_arm = _segment_frames(traj, release_idx)[:, :7]
        assert np.allclose(release_arm, release_arm[0])
        assert release_arm[0][3] == pytest.approx(0.0)
