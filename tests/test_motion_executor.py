"""Tests for MotionExecutor velocity-adaptive step calculation."""

import math

import numpy as np
import pytest

from robotsmith.motion.chain import _compute_steps, _same_pose
from robotsmith.execution.executor import MotionExecutor
from robotsmith.motion.planner import GenesisBackend, PlanResult
from robotsmith.motion.params import MotionParams
from robotsmith.grasp.planner import GraspPlan, Waypoint

DT = 1.0 / 30  # Genesis control dt
IDENTITY = np.array([1, 0, 0, 0], dtype=np.float64)


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

    def test_fine_vel_gives_more_steps_than_normal(self):
        q1 = np.zeros(9)
        q1[2] = 0.5
        s_normal, _ = _compute_steps(np.zeros(9), q1, vel=1.0, dt=DT)
        s_fine, _ = _compute_steps(np.zeros(9), q1, vel=0.3, dt=DT)
        assert s_fine > s_normal

    def test_ignores_finger_joints(self):
        """Only arm joints (first 7) count for distance."""
        q1 = np.zeros(9)
        q1[7] = q1[8] = 0.04
        steps, dist = _compute_steps(np.zeros(9), q1, vel=1.0, dt=DT, min_steps=5)
        assert dist < 1e-6
        assert steps == 5


# -- MotionExecutor.pick / place --

def _trivial_ik(pos, quat, finger_width, init_qpos=None, **_kw):
    q = np.zeros(9)
    q[0], q[1], q[2] = pos[0], pos[1], pos[2]
    q[7] = q[8] = finger_width
    return q


def _pick_plan():
    """4-waypoint plan: pre-approach, grasp-open, grasp-close, retreat."""
    return GraspPlan(
        grasp_pos=np.array([0.5, 0.0, 0.1]),
        grasp_quat=IDENTITY,
        pre_grasp_pos=np.array([0.5, 0.0, 0.2]),
        pre_grasp_quat=IDENTITY,
        retreat_pos=np.array([0.5, 0.0, 0.3]),
        retreat_quat=IDENTITY,
        finger_open=0.04,
        finger_closed=0.0,
        waypoints=[
            Waypoint(pos=np.array([0.5, 0.0, 0.22]), quat=IDENTITY, finger_width=0.04),
            Waypoint(pos=np.array([0.5, 0.0, 0.10]), quat=IDENTITY, finger_width=0.04),
            Waypoint(pos=np.array([0.5, 0.0, 0.10]), quat=IDENTITY, finger_width=0.00),
            Waypoint(pos=np.array([0.5, 0.0, 0.30]), quat=IDENTITY, finger_width=0.00),
        ],
    )


def _segment_frames(exe, traj, idx):
    segs = exe.last_motion_trace["segments"]
    bounds = np.cumsum([0] + [s["steps"] + s["hold_after_steps"] for s in segs])
    return np.asarray(traj)[bounds[idx]:bounds[idx + 1]]


class TestPickMotionChain:
    def test_segment_types(self):
        """Fine segment is the grasp approach (seg 1), not the retreat."""
        exe = MotionExecutor()
        exe.pick(_pick_plan(), GenesisBackend(_trivial_ik), np.zeros(9), MotionParams())
        types = [s["type"] for s in exe.last_motion_trace["segments"]]
        assert types[1:4] == ["fine", "finger", "normal"]

    def test_finger_seg_locks_arm_joints(self):
        """IK that jumps when fingers close must not move the arm during the close."""
        def unstable_ik(pos, quat, finger_width, init_qpos=None):
            q = _trivial_ik(pos, quat, finger_width)
            if finger_width < 0.02:
                q[3] = 1.5
            return q

        exe = MotionExecutor()
        traj = exe.pick(_pick_plan(), GenesisBackend(unstable_ik), np.zeros(9),
                        MotionParams(finger_steps=10))
        arm = _segment_frames(exe, traj, 2)[:, :7]
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

        params = MotionParams(
            transport_steps=3,
            place_descend_steps=4,
            release_steps=5,
            retreat_steps=6,
        )
        exe = MotionExecutor()
        traj = exe.place(plan, GenesisBackend(unstable_open_ik), np.zeros(9), params)

        segs = exe.last_motion_trace["segments"]
        labels = [s["label"] for s in segs]
        assert labels == ["transport", "descend", "release", "retreat"]
        release_idx = labels.index("release")
        assert segs[release_idx]["finger_only"] is True

        # If release re-solved IK while the fingers open, joint 3 would whip to 1.5.
        release_arm = _segment_frames(exe, traj, release_idx)[:, :7]
        assert np.allclose(release_arm, release_arm[0])
        assert release_arm[0][3] == pytest.approx(0.0)


class _RecordingPlanner:
    """Collision-aware backend stub that records the targets sent to plan_motion."""

    def __init__(self, succeed=True):
        self.collision_aware = True
        self.succeed = succeed
        self.plan_motion_targets: list[list[float]] = []
        self.goal_seeds: list = []

    def solve_ik(self, pos, quat, finger_width, init_qpos=None, **_kw):
        return _trivial_ik(pos, quat, finger_width)

    def plan_motion(self, q_start, waypoints, *, world=None, control_dt=None, attach=None, goal_seeds=None):
        wp = waypoints[-1]
        self.plan_motion_targets.append([float(v) for v in wp.pos])
        self.goal_seeds.append(goal_seeds)
        if not self.succeed:
            return PlanResult(success=False, reason="endpoint")
        return PlanResult(success=True, trajectory=[self.solve_ik(wp.pos, wp.quat, wp.finger_width)])


class TestPickCollisionWorld:
    """pick(world=...) routes free segments through plan_motion; the final
    approach stays Cartesian (world-blind)."""

    def test_free_segments_use_plan_motion_fine_stays_cartesian(self):
        planner = _RecordingPlanner()
        traj = MotionExecutor().pick(_pick_plan(), planner, np.zeros(9), MotionParams(),
                                     world=[{"type": "halfspace"}])
        assert len(traj) > 0
        # Only pre-approach (wp0) and retreat (wp3) are free arm segments.
        assert planner.plan_motion_targets == [[0.5, 0.0, 0.22], [0.5, 0.0, 0.30]]

    def test_no_world_never_calls_plan_motion(self):
        planner = _RecordingPlanner()
        MotionExecutor().pick(_pick_plan(), planner, np.zeros(9), MotionParams(), world=None)
        assert planner.plan_motion_targets == []

    def test_free_segments_send_their_endpoint_ik_as_the_goal_seed(self):
        planner = _RecordingPlanner()
        MotionExecutor().pick(_pick_plan(), planner, np.zeros(9), MotionParams(), world=[{"type": "halfspace"}])
        for target, seeds in zip(planner.plan_motion_targets, planner.goal_seeds):
            assert len(seeds) == 1
            assert np.allclose(seeds[0], _trivial_ik(np.array(target), IDENTITY, seeds[0][7]))

    def test_a_miss_falls_back_to_the_same_seed(self):
        planner = _RecordingPlanner(succeed=False)
        traj = MotionExecutor().pick(_pick_plan(), planner, np.zeros(9), MotionParams(), world=[{"type": "halfspace"}])
        assert any(np.allclose(q[:7], planner.goal_seeds[0][0][:7]) for q in traj)


def test_same_pose_ignores_finger_width():
    def wp(z, finger):
        return Waypoint(pos=np.array([0.5, 0.0, z]), quat=IDENTITY, finger_width=finger)

    assert _same_pose(wp(0.1, 0.04), wp(0.1, 0.00))
    assert not _same_pose(wp(0.1, 0.04), wp(0.3, 0.04))
