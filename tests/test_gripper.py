"""gripper: one owner for finger-only segments; the arm stays where it is."""

import numpy as np

from robotsmith.execution.executor import MotionExecutor
from robotsmith.execution.gate import Holding
from robotsmith.execution.meta_actions import pick_actions, place_actions
from robotsmith.grasp.planner import GraspPlan, Waypoint
from robotsmith.motion import gripper
from robotsmith.motion.chain import evaluate_motion_chain
from robotsmith.motion.gripper import Gripper, expand
from robotsmith.motion.params import MotionParams
from robotsmith.motion.plan_state import PlanState
from robotsmith.motion.planner import GenesisBackend

QUAT = np.array([0.0, 1.0, 0.0, 0.0])


def _ik(pos, quat, finger_width, init_qpos=None, **_kw):
    q = np.full(9, 0.7)
    q[:3] = pos
    q[7] = q[8] = finger_width
    return q


def test_expand_replaces_only_the_finger_joints_with_one_ik_call():
    calls = []

    def ik(pos, quat, finger_width, init_qpos=None):
        calls.append(init_qpos)
        return _ik(pos, quat, finger_width)

    q_start = np.arange(9, dtype=np.float64) / 10
    seg = expand(Gripper(np.array([0.5, 0.0, 0.1]), QUAT, 0.04), q_start, ik)
    assert calls == [None]
    np.testing.assert_array_equal(seg.q_target[:7], q_start[:7])
    np.testing.assert_array_equal(seg.q_target[7:], [0.04, 0.04])


def test_expand_leaves_its_inputs_alone_and_repeats_itself():
    move, q_start = Gripper(np.array([0.5, 0.0, 0.1]), QUAT, 0.0), np.full(9, 0.3)
    before = (move.pos.copy(), q_start.copy())
    a, b = expand(move, q_start, _ik), expand(move, q_start, _ik)
    np.testing.assert_array_equal(move.pos, before[0])
    np.testing.assert_array_equal(q_start, before[1])
    np.testing.assert_array_equal(a.q_target, b.q_target)
    assert a.q_target is not q_start


def test_pick_place_drag_and_scoring_open_and_close_through_gripper(monkeypatch):
    seen = []
    real = gripper.expand

    def spy(move, q_start, solve_ik):
        seen.append(move.finger_width)
        return real(move, q_start, solve_ik)

    monkeypatch.setattr(gripper, "expand", spy)
    planner, params, exe = GenesisBackend(_ik), MotionParams(), MotionExecutor()
    plan = GraspPlan(
        grasp_pos=np.array([0.5, 0.0, 0.1]), grasp_quat=QUAT,
        pre_grasp_pos=np.array([0.5, 0.0, 0.2]), pre_grasp_quat=QUAT,
        retreat_pos=np.array([0.5, 0.0, 0.3]), retreat_quat=QUAT,
        finger_open=0.04, finger_closed=0.0,
        waypoints=[
            Waypoint(pos=np.array([0.5, 0.0, 0.2]), quat=QUAT, finger_width=0.04),
            Waypoint(pos=np.array([0.5, 0.0, 0.1]), quat=QUAT, finger_width=0.04),
            Waypoint(pos=np.array([0.5, 0.0, 0.1]), quat=QUAT, finger_width=0.0),
            Waypoint(pos=np.array([0.5, 0.0, 0.3]), quat=QUAT, finger_width=0.0),
        ],
    )
    q = _ik(plan.waypoints[0].pos, QUAT, 0.0)
    exe.run("pick", pick_actions(plan, q, params, planner.solve_ik, grasp=Holding("die", np.eye(4))), planner,
            PlanState(q), params)
    assert seen == [0.04, 0.04, 0.0]
    seen.clear()
    exe.run("place", place_actions(plan, params), planner, PlanState(np.zeros(9), Holding("die", np.eye(4))), params)
    assert seen == [0.04]
    seen.clear()
    exe.drag_handle(np.array([0.5, 0.0, 0.15]), QUAT, np.array([-0.1, 0.0, 0.0]), planner, np.zeros(9), params)
    assert seen == [0.0, 0.04]
    seen.clear()
    evaluate_motion_chain(plan.waypoints, _ik, np.zeros(9), max_final_approach_joint_step=0.8)
    assert seen == [0.0]
