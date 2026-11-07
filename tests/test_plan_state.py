"""PlanState: attach / detach are the only changes to what the hand holds, and chain segments read it."""

import numpy as np
import pytest

from robotsmith.execution.executor import MotionExecutor
from robotsmith.execution.gate import Holding
from robotsmith.grasp.planner import GraspPlan, Waypoint
from robotsmith.grasp.transforms import pose_matrix
from robotsmith.motion.params import MotionParams
from robotsmith.motion.plan_state import PlanState, attach, binding, detach
from robotsmith.motion.planner import GenesisBackend

QUAT = np.array([0.0, 1.0, 0.0, 0.0])
HELD = Holding("die", np.eye(4))


def _ik(pos, quat, finger_width, init_qpos=None, **_kw):
    q = np.zeros(9)
    q[:3] = pos
    q[7] = q[8] = finger_width
    return q


def _plan():
    return GraspPlan(
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


def test_attach_and_detach_change_only_the_holding():
    q = np.arange(9.0)
    empty = PlanState(q)
    held = attach(empty, HELD)
    assert held.holding is HELD and held.q is q and empty.holding is None
    assert detach(held).holding is None and detach(held).q is q
    with pytest.raises(ValueError):
        attach(held, HELD)
    with pytest.raises(ValueError):
        detach(empty)


def test_a_binding_holds_the_object_where_it_is_relative_to_the_given_hand():
    obj = (np.array([0.5, 0.1, 0.05]), np.array([1.0, 0.0, 0.0, 0.0]))
    grasp, measured = (np.array([0.5, 0.1, 0.12]), QUAT), (np.array([0.5, 0.1, 0.13]), QUAT)
    predicted, actual = binding("die", grasp, obj), binding("die", measured, obj)
    for hand, held in ((grasp, predicted), (measured, actual)):
        np.testing.assert_allclose(pose_matrix(*hand) @ held.t_hand_object, pose_matrix(*obj), atol=1e-12)
    assert not np.allclose(predicted.t_hand_object, actual.t_hand_object)


def test_pick_holds_after_its_last_finger_segment_and_place_until_its_release_ends():
    params, exe, planner = MotionParams(), MotionExecutor(), GenesisBackend(_ik)
    pick = exe.pick(_plan(), planner, _ik(_plan().waypoints[0].pos, QUAT, 0.0), params, grasp=HELD)
    assert [(s["label"], s["holding"], s["allow"]) for s in pick.segments] == [
        ("wp0", False, None), ("wp1", False, None), ("wp2", False, "grip"), ("wp3", True, "rest")]
    place = exe.place(_plan(), planner, PlanState(np.zeros(9), HELD), params)
    assert [(s["label"], s["holding"], s["allow"]) for s in place.segments] == [
        ("transport", True, None), ("descend", True, "rest"), ("release", True, "rest"), ("retreat", False, "grip")]
    transport = exe.place(_plan(), planner, PlanState(np.zeros(9), HELD), params, phase="transport")
    assert [(s["label"], s["holding"]) for s in transport.segments] == [("transport", True)]
    with pytest.raises(ValueError):
        exe.place(_plan(), planner, PlanState(np.zeros(9)), params)
