"""move_linear: one owner for Cartesian straight-line expansion, with the caller's jump policy."""

import numpy as np
import pytest

from robotsmith.execution.executor import MotionExecutor
from robotsmith.execution.gate import Holding
from robotsmith.execution.meta_actions import pick_actions, place_actions
from robotsmith.grasp.planner import GraspPlan
from robotsmith.motion import move_linear
from robotsmith.motion.move_linear import MoveLinear, expand
from robotsmith.motion.params import MotionParams
from robotsmith.motion.plan_state import PlanState
from robotsmith.motion.planner import GenesisBackend

QUAT = np.array([0.0, 1.0, 0.0, 0.0])


def _ik(pos, quat, finger_width, init_qpos=None, **_kw):
    q = np.zeros(9)
    q[:3] = pos
    q[7] = q[8] = finger_width
    return q


def _jumping_ik(pos, quat, finger_width, init_qpos=None, **_kw):
    q = _ik(pos, quat, finger_width)
    if pos[2] < 0.15:
        q[4] = 1.5
    return q


def _move(on_jump, **kw):
    return MoveLinear(np.array([0.5, 0.0, 0.2]), np.array([0.5, 0.0, 0.1]), QUAT, 0.04, on_jump=on_jump, **kw)


def test_expand_starts_at_the_nominal_position_and_ends_at_the_target():
    calls = []

    def ik(pos, quat, finger_width, init_qpos=None):
        calls.append((pos.copy(), None if init_qpos is None else init_qpos.copy()))
        return _ik(pos, quat, finger_width)

    q_start = np.full(9, 0.3)
    seg = expand(_move("refuse"), q_start, ik)
    assert len(seg.path) == len(calls) == 10
    np.testing.assert_allclose(calls[0][0], [0.5, 0.0, 0.19])
    np.testing.assert_array_equal(calls[0][1], q_start)
    np.testing.assert_allclose(seg.path[-1][:3], [0.5, 0.0, 0.1])
    assert seg.continuous


def test_expand_leaves_its_inputs_alone_and_repeats_itself():
    move, q_start = _move("keep"), np.zeros(9)
    before = (move.start_pos.copy(), move.end_pos.copy(), q_start.copy())
    a, b = expand(move, q_start, _jumping_ik), expand(move, q_start, _jumping_ik)
    np.testing.assert_array_equal(move.start_pos, before[0])
    np.testing.assert_array_equal(move.end_pos, before[1])
    np.testing.assert_array_equal(q_start, before[2])
    assert a.max_joint_step == b.max_joint_step
    for qa, qb in zip(a.path, b.path, strict=True):
        np.testing.assert_array_equal(qa, qb)


def test_refuse_raises_on_a_jump_and_keep_returns_the_path():
    with pytest.raises(RuntimeError, match=r"wp2 IK discontinuity: step=1\.500 rad > 0\.800 rad"):
        expand(_move("refuse", label="wp2"), np.zeros(9), _jumping_ik)
    seg = expand(_move("keep"), np.zeros(9), _jumping_ik)
    assert not seg.continuous
    assert seg.max_joint_step == pytest.approx(1.5)
    np.testing.assert_allclose(seg.path[-1][:3], [0.5, 0.0, 0.1])


def test_pick_place_and_drag_expand_their_lines_through_move_linear(monkeypatch):
    seen = []
    real = move_linear.expand

    def spy(move, q_start, solve_ik, **kw):
        seen.append((move.label, move.on_jump))
        return real(move, q_start, solve_ik, **kw)

    monkeypatch.setattr(move_linear, "expand", spy)
    planner, params, exe = GenesisBackend(_ik), MotionParams(), MotionExecutor()
    plan = GraspPlan(
        grasp_pos=np.array([0.5, 0.0, 0.1]), grasp_quat=QUAT,
        pre_grasp_pos=np.array([0.5, 0.0, 0.2]), pre_grasp_quat=QUAT,
        retreat_pos=np.array([0.5, 0.0, 0.3]), retreat_quat=QUAT,
        finger_open=0.04, finger_closed=0.0,
    )
    pre = (plan.pre_grasp_pos, plan.pre_grasp_quat)
    q = _ik(*pre, 0.04)
    exe.run("pick", pick_actions(plan, q, pre, (), params, planner.solve_ik, grasp=Holding("die", np.eye(4))), planner,
            PlanState(q), params)
    assert seen == [("descend", "refuse"), ("lift", "keep")]
    seen.clear()
    exe.run("place", place_actions(plan, params), planner, PlanState(np.zeros(9), Holding("die", np.eye(4))), params)
    assert seen == [("descend", "keep"), ("retreat", "keep")]
    seen.clear()
    exe.drag_handle(np.array([0.5, 0.0, 0.15]), QUAT, np.array([-0.1, 0.0, 0.0]), planner, np.zeros(9), params)
    assert seen and {policy for _, policy in seen} == {"keep"}
