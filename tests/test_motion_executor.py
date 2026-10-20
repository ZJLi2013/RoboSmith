"""Tests for MotionExecutor velocity-adaptive step calculation."""

import math

import numpy as np
import pytest

from robotsmith.motion.chain import _compute_steps, _same_pose
from robotsmith.execution.executor import MotionExecutor
from robotsmith.motion.planner import GenesisBackend, PlanResult
from robotsmith.motion.params import MotionParams
from robotsmith.grasp.planner import GraspPlan, Waypoint


# -- _compute_steps --

class TestComputeSteps:
    DT = 1.0 / 30  # Genesis control dt

    def test_large_move_gets_many_steps(self):
        q0 = np.zeros(9)
        q1 = np.zeros(9)
        q1[0] = 2.0  # 2 rad on joint 0
        steps, dist = _compute_steps(q0, q1, vel=1.0, dt=self.DT)
        assert dist == pytest.approx(2.0)
        assert steps == math.ceil(2.0 / (1.0 * self.DT))  # 60

    def test_small_move_respects_min_steps(self):
        q0 = np.zeros(9)
        q1 = np.zeros(9)
        q1[0] = 0.001
        steps, dist = _compute_steps(q0, q1, vel=1.0, dt=self.DT, min_steps=5)
        assert steps == 5

    def test_zero_move_returns_min_steps(self):
        q = np.zeros(9)
        steps, dist = _compute_steps(q, q.copy(), vel=1.0, dt=self.DT, min_steps=5)
        assert steps == 5
        assert dist < 1e-6

    def test_fine_vel_gives_more_steps_than_normal(self):
        q0 = np.zeros(9)
        q1 = np.zeros(9)
        q1[2] = 0.5
        s_normal, _ = _compute_steps(q0, q1, vel=1.0, dt=self.DT)
        s_fine, _ = _compute_steps(q0, q1, vel=0.3, dt=self.DT)
        assert s_fine > s_normal

    def test_ignores_finger_joints(self):
        """Only arm joints (first 7) count for distance."""
        q0 = np.zeros(9)
        q1 = np.zeros(9)
        q1[7] = 0.04  # finger only
        q1[8] = 0.04
        steps, dist = _compute_steps(q0, q1, vel=1.0, dt=self.DT, min_steps=5)
        assert dist < 1e-6
        assert steps == 5


# -- MotionExecutor._pick_waypoints integration --

class TestPickWaypointsVelocityAdaptive:
    """End-to-end test with a trivial IK solver."""

    @staticmethod
    def _trivial_ik(pos, quat, finger_width, init_qpos=None):
        """Maps position to arm joints, finger to last 2 joints."""
        q = np.zeros(9)
        q[0] = pos[0]
        q[1] = pos[1]
        q[2] = pos[2]
        q[7] = finger_width
        q[8] = finger_width
        return q

    def _make_4wp_plan(self):
        """4-waypoint plan: pre-approach, grasp-open, grasp-close, retreat."""
        identity_quat = np.array([1, 0, 0, 0], dtype=np.float64)
        return GraspPlan(
            grasp_pos=np.array([0.5, 0.0, 0.1]),
            grasp_quat=identity_quat,
            pre_grasp_pos=np.array([0.5, 0.0, 0.2]),
            pre_grasp_quat=identity_quat,
            retreat_pos=np.array([0.5, 0.0, 0.3]),
            retreat_quat=identity_quat,
            finger_open=0.04,
            finger_closed=0.0,
            waypoints=[
                Waypoint(pos=np.array([0.5, 0.0, 0.22]),
                         quat=identity_quat, finger_width=0.04),
                Waypoint(pos=np.array([0.5, 0.0, 0.10]),
                         quat=identity_quat, finger_width=0.04),
                Waypoint(pos=np.array([0.5, 0.0, 0.10]),
                         quat=identity_quat, finger_width=0.00),
                Waypoint(pos=np.array([0.5, 0.0, 0.30]),
                         quat=identity_quat, finger_width=0.00),
            ],
        )

    def test_trajectory_nonempty(self):
        exe = MotionExecutor()
        plan = self._make_4wp_plan()
        home = np.zeros(9)
        params = MotionParams()
        traj = exe.pick(plan, GenesisBackend(self._trivial_ik), home, params)
        assert len(traj) > 0

    def test_finger_segment_uses_finger_steps(self):
        """Segment 2 (grasp-open → close) should use finger_steps."""
        exe = MotionExecutor()
        plan = self._make_4wp_plan()
        home = np.zeros(9)
        params = MotionParams(finger_steps=12)

        traj = exe.pick(plan, GenesisBackend(self._trivial_ik), home, params)

        q_wp1 = self._trivial_ik(*[plan.waypoints[1].pos,
                                    plan.waypoints[1].quat,
                                    plan.waypoints[1].finger_width])
        q_wp2 = self._trivial_ik(*[plan.waypoints[2].pos,
                                    plan.waypoints[2].quat,
                                    plan.waypoints[2].finger_width])
        assert np.allclose(q_wp1[:7], q_wp2[:7])

    def test_large_move_gets_more_steps_than_small(self):
        """A larger joint move should be subdivided into more steps than a small
        one (velocity-adaptive step count). Compare the two segments directly
        rather than asserting an arbitrary absolute length."""
        q_far = np.array([2.0, 1.0, 0.5, 0, 0, 0, 0, 0.04, 0.04])
        q_near = np.array([2.0, 1.0, 0.48, 0, 0, 0, 0, 0.04, 0.04])
        params = MotionParams(max_joint_vel=1.0, fine_joint_vel=0.3)

        large_steps, _ = _compute_steps(
            np.zeros(9), q_far, params.max_joint_vel, params.dt, params.min_steps,
        )
        small_steps, _ = _compute_steps(
            q_far, q_near, params.max_joint_vel, params.dt, params.min_steps,
        )
        assert large_steps > small_steps

    def test_finger_seg_locks_arm_joints(self):
        """When IK returns different arm configs for same Cartesian pose,
        the finger segment should lock arm joints from the previous solution."""
        call_count = [0]

        def _unstable_ik(pos, quat, finger_width, init_qpos=None):
            """IK that returns different arm joints for different finger widths."""
            q = np.zeros(9)
            q[0] = pos[0]
            q[1] = pos[1]
            q[2] = pos[2]
            if finger_width < 0.02:
                q[3] = 1.5  # arm joint jumps when fingers close
            q[7] = finger_width
            q[8] = finger_width
            call_count[0] += 1
            return q

        exe = MotionExecutor()
        plan = self._make_4wp_plan()
        home = np.zeros(9)
        params = MotionParams(finger_steps=10)

        traj = exe.pick(plan, GenesisBackend(_unstable_ik), home, params)

        wp1_q = np.zeros(9)
        wp1_q[0], wp1_q[1], wp1_q[2] = 0.5, 0.0, 0.10
        wp1_q[7] = wp1_q[8] = 0.04

        seg0_end = sum([
            _compute_steps(np.zeros(9),
                           np.array([0.5, 0.0, 0.22, 0, 0, 0, 0, 0.04, 0.04]),
                           1.0, 1/30, 5)[0],
        ])
        finger_seg_start = seg0_end + _compute_steps(
            np.array([0.5, 0.0, 0.22, 0, 0, 0, 0, 0.04, 0.04]),
            np.array([0.5, 0.0, 0.10, 0, 0, 0, 0, 0.04, 0.04]),
            0.3, 1/30, 5,
        )[0]
        finger_seg_q = traj[finger_seg_start]
        np.testing.assert_allclose(finger_seg_q[:7], wp1_q[:7], atol=1e-6,
                                   err_msg="Arm joints should be locked during finger segment")

    def test_fine_is_last_arm_before_finger(self, caplog):
        """Fine segment should be the approach segment (seg 1), not retreat."""
        exe = MotionExecutor()
        plan = self._make_4wp_plan()
        home = np.zeros(9)
        params = MotionParams()

        import logging
        caplog.set_level(logging.DEBUG, logger="robotsmith.motion.chain")

        exe.pick(plan, GenesisBackend(self._trivial_ik), home, params)

        lines = [
            record.getMessage()
            for record in caplog.records
            if "seg " in record.getMessage() and "type=" in record.getMessage()
        ]
        types = []
        for line in lines:
            t = line.split("type=")[1].split()[0]
            types.append(t)

        assert types[1] == "fine", f"seg 1 should be fine, got {types}"
        assert types[2] == "finger", f"seg 2 should be finger, got {types}"
        assert types[3] == "normal", f"seg 3 should be normal, got {types}"


class TestPlaceMotionChain:
    def test_place_release_locks_arm_and_records_trace(self):
        identity_quat = np.array([1, 0, 0, 0], dtype=np.float64)
        plan = GraspPlan(
            grasp_pos=np.array([0.5, 0.0, 0.1]),
            grasp_quat=identity_quat,
            pre_grasp_pos=np.array([0.5, 0.0, 0.2]),
            pre_grasp_quat=identity_quat,
            retreat_pos=np.array([0.5, 0.0, 0.3]),
            retreat_quat=identity_quat,
            finger_open=0.04,
            finger_closed=0.0,
        )

        def unstable_open_ik(pos, quat, finger_width, init_qpos=None):
            q = np.zeros(9)
            q[0], q[1], q[2] = pos
            if finger_width > 0.02:
                q[3] = 1.5
            q[7] = q[8] = finger_width
            return q

        params = MotionParams(
            transport_steps=3,
            place_descend_steps=4,
            release_steps=5,
            retreat_steps=6,
        )
        exe = MotionExecutor()
        traj = exe.place(plan, GenesisBackend(unstable_open_ik), np.zeros(9), params)

        # records_trace: the chain is the expected four segments and release is
        # finger-only (no hard-coded frame totals — Cartesian descend/retreat are
        # subdivided by max_joint_step, so an absolute length would be a brittle
        # magic number that nobody could later justify).
        segs = exe.last_motion_trace["segments"]
        labels = [s["label"] for s in segs]
        assert labels == ["transport", "descend", "release", "retreat"]
        release_idx = labels.index("release")
        assert segs[release_idx]["finger_only"] is True

        # locks_arm: slice the release frames using the trace's own per-segment
        # lengths, then assert the arm pose is held constant across the release
        # and stays at descend's pose (joint 3 == 0). If release re-solved IK
        # while the fingers open, unstable_open_ik would whip joint 3 to 1.5.
        bounds = np.cumsum([0] + [s["steps"] + s["hold_after_steps"] for s in segs])
        release_frames = np.asarray(traj)[bounds[release_idx]:bounds[release_idx + 1]]
        release_arm = release_frames[:, :7]
        assert np.allclose(release_arm, release_arm[0])
        assert release_arm[0][3] == pytest.approx(0.0)


class _RecordingPlanner:
    """Collision-aware backend stub: records which segments hit plan_motion.

    ``solve_ik`` maps position→arm joints (like the trivial IK); ``plan_motion``
    records the target xyz it was asked to plan to and returns a 1-step success.
    """

    def __init__(self):
        self.collision_aware = True
        self.plan_motion_targets: list[list[float]] = []

    def solve_ik(self, pos, quat, finger_width, init_qpos=None, **_kw):
        q = np.zeros(9)
        q[0], q[1], q[2] = pos[0], pos[1], pos[2]
        q[7] = q[8] = finger_width
        return q

    def plan_motion(self, q_start, waypoints, *, world=None, control_dt=None, attach=None):
        wp = waypoints[-1]
        self.plan_motion_targets.append([float(v) for v in wp.pos])
        q_to = self.solve_ik(wp.pos, wp.quat, wp.finger_width)
        return PlanResult(success=True, trajectory=[q_to])


class TestPickCollisionWorld:
    """pick(world=...) routes free segments through plan_motion; the final
    approach stays Cartesian (world-blind). See docs/features/pick.md."""

    def _plan(self):
        identity = np.array([1, 0, 0, 0], dtype=np.float64)
        return GraspPlan(
            grasp_pos=np.array([0.5, 0.0, 0.1]),
            grasp_quat=identity,
            pre_grasp_pos=np.array([0.5, 0.0, 0.2]),
            pre_grasp_quat=identity,
            retreat_pos=np.array([0.5, 0.0, 0.3]),
            retreat_quat=identity,
            finger_open=0.04,
            finger_closed=0.0,
            waypoints=[
                Waypoint(pos=np.array([0.5, 0.0, 0.22]), quat=identity, finger_width=0.04),
                Waypoint(pos=np.array([0.5, 0.0, 0.10]), quat=identity, finger_width=0.04),
                Waypoint(pos=np.array([0.5, 0.0, 0.10]), quat=identity, finger_width=0.00),
                Waypoint(pos=np.array([0.5, 0.0, 0.30]), quat=identity, finger_width=0.00),
            ],
        )

    def test_free_segments_use_plan_motion_fine_stays_cartesian(self):
        planner = _RecordingPlanner()
        exe = MotionExecutor()
        traj = exe.pick(self._plan(), planner, np.zeros(9), MotionParams(),
                        world=[{"type": "halfspace"}])
        assert len(traj) > 0
        # Only the two free arm segments (pre-approach wp0, retreat wp3) route
        # through plan_motion; the fine grasp approach (wp1) and finger close (wp2)
        # do not.
        assert planner.plan_motion_targets == [[0.5, 0.0, 0.22], [0.5, 0.0, 0.30]]

    def test_no_world_never_calls_plan_motion(self):
        planner = _RecordingPlanner()
        exe = MotionExecutor()
        exe.pick(self._plan(), planner, np.zeros(9), MotionParams(), world=None)
        assert planner.plan_motion_targets == []


# -- _same_cartesian_target --

class TestSameCartesianTarget:
    def test_same_pose_different_finger(self):
        wp_a = Waypoint(pos=np.array([0.5, 0.0, 0.1]),
                        quat=np.array([1, 0, 0, 0], dtype=np.float64),
                        finger_width=0.04)
        wp_b = Waypoint(pos=np.array([0.5, 0.0, 0.1]),
                        quat=np.array([1, 0, 0, 0], dtype=np.float64),
                        finger_width=0.00)
        assert _same_pose(wp_a, wp_b)

    def test_different_pos(self):
        wp_a = Waypoint(pos=np.array([0.5, 0.0, 0.1]),
                        quat=np.array([1, 0, 0, 0], dtype=np.float64),
                        finger_width=0.04)
        wp_b = Waypoint(pos=np.array([0.5, 0.0, 0.3]),
                        quat=np.array([1, 0, 0, 0], dtype=np.float64),
                        finger_width=0.04)
        assert not _same_pose(wp_a, wp_b)
