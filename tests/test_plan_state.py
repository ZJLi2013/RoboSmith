"""PlanState: attach / detach are the only changes to what the hand holds, and chain segments read it."""

import numpy as np
import pytest

from robotsmith.execution.executor import MotionExecutor
from robotsmith.execution.gate import Holding
from robotsmith.execution.meta_actions import pick_actions, place_actions
from robotsmith.grasp.planner import GraspPlan
from robotsmith.grasp.transforms import pose_matrix
from robotsmith.motion.chain import RuntimeMotionWaypoint
from robotsmith.motion.params import MotionParams
from robotsmith.motion.plan_state import Attach, Detach, PlanState, attach, binding, detach
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
    )


START = (np.array([0.3, 0.2, 0.4]), QUAT)


def _pick(exe, planner, params, finger=0.0):
    """From the arm at ``START`` with the fingers at ``finger``, along a straight joint path to the pre-grasp."""
    q0, opened, q_pre = _ik(*START, finger), _ik(*START, 0.04), _ik(_plan().pre_grasp_pos, QUAT, 0.04)
    approach = [opened + (q_pre - opened) * t for t in (0.5, 1.0)]
    return exe.run("pick", pick_actions(_plan(), q0, START, approach, params, planner.solve_ik, grasp=HELD), planner,
                   PlanState(q0), params)


def _place(exe, planner, params, start, phase="all"):
    return exe.run("place", place_actions(_plan(), params, phase=phase), planner, start, params)


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


def test_pick_holds_from_its_close_and_place_until_its_descend_ends():
    params, exe, planner = MotionParams(), MotionExecutor(), GenesisBackend(_ik)
    pick = _pick(exe, planner, params)
    assert [(s["label"], s["holding"], s["allow"]) for s in pick.segments] == [
        ("open", False, None), ("approach", False, None), ("descend", False, None), ("close", False, "grip"),
        ("lift", True, "rest")]
    place = _place(exe, planner, params, PlanState(np.zeros(9), HELD))
    assert [(s["label"], s["holding"], s["allow"]) for s in place.segments] == [
        ("transport", True, None), ("descend", True, "rest"), ("release", False, "grip"), ("retreat", False, "grip")]
    transport = _place(exe, planner, params, PlanState(np.zeros(9), HELD), phase="transport")
    assert [(s["label"], s["holding"]) for s in transport.segments] == [("transport", True)]
    with pytest.raises(ValueError):
        _place(exe, planner, params, PlanState(np.zeros(9)))


def test_pick_opens_the_fingers_where_the_arm_starts_before_it_moves():
    params, exe, planner = MotionParams(), MotionExecutor(), GenesisBackend(_ik)
    pick = _pick(exe, planner, params)
    opening, approach = pick.segments[0], pick.segments[1]
    q0 = _ik(*START, 0.0)
    frames = np.asarray(pick.trajectory[slice(*opening["span"])])
    assert len(frames) == params.finger_steps and np.allclose(frames[:, :7], q0[:7])
    np.testing.assert_allclose(frames[-1][7:], 0.04)
    # The approach starts from the opened hand and is the planned path, point for point.
    np.testing.assert_array_equal(pick.trajectory[approach["span"][0]][7:], [0.04, 0.04])
    assert approach["span"][1] - approach["span"][0] == 2
    already = _pick(exe, planner, params, finger=0.04)
    assert already.segments[0]["span"] == [0, 0]


def test_pick_lifts_in_a_straight_line_along_world_z_after_the_attach():
    params, exe, planner = MotionParams(), MotionExecutor(), GenesisBackend(_ik)
    pick = _pick(exe, planner, params)
    lift = pick.segments[-1]
    assert lift["label"] == "lift" and lift["cartesian"] and lift["holding"]
    frames = np.asarray(pick.trajectory[lift["span"][0]:lift["span"][1] - params.lift_hold_steps])[:, :3]
    assert len(frames) > 1
    np.testing.assert_allclose(frames[:, :2], np.tile([0.5, 0.0], (len(frames), 1)), atol=1e-12)
    assert np.all(np.diff(frames[:, 2]) > 0)
    # 0.10 m above the grasp, not the candidate's last waypoint 0.3.
    np.testing.assert_allclose(frames[-1], [0.5, 0.0, 0.2], atol=1e-12)


def test_place_detaches_before_its_fingers_open():
    params, exe, planner = MotionParams(), MotionExecutor(), GenesisBackend(_ik)
    actions = place_actions(_plan(), params)
    kinds = [a.label if isinstance(a, RuntimeMotionWaypoint) else type(a).__name__ for a in actions]
    assert kinds == ["transport", "descend", "Detach", "release", "retreat"]
    place = _place(exe, planner, params, PlanState(np.zeros(9), HELD))
    descend, release = place.segments[1], place.segments[2]
    assert descend["holding"] and not release["holding"] and release["finger_only"]
    assert release["span"][0] == descend["span"][1]


def test_the_chain_advances_q_through_each_segment_and_hands_over_where_it_ends():
    params, exe, planner = MotionParams(), MotionExecutor(), GenesisBackend(_ik)
    pick = _pick(exe, planner, params)
    for seg, q_end in zip(pick.segments, pick.q_waypoints[1:]):
        start, end = seg["span"]
        if end > start:
            np.testing.assert_array_equal(pick.trajectory[end - 1], q_end)
    np.testing.assert_array_equal(pick.state.q, pick.trajectory[-1])
    assert pick.state.holding is HELD
    carried = _place(exe, planner, params, pick.state, phase="transport")
    np.testing.assert_array_equal(carried.q_waypoints[0], pick.state.q)
    assert carried.state.holding is HELD
    placed = _place(exe, planner, params, carried.state)
    np.testing.assert_array_equal(placed.state.q, placed.trajectory[-1])
    assert placed.state.holding is None


def test_an_event_takes_a_list_item_and_moves_nothing():
    params, exe, planner = MotionParams(), MotionExecutor(), GenesisBackend(_ik)
    moves = [RuntimeMotionWaypoint(pos=np.array([0.5, 0.0, 0.2]), quat=QUAT, finger_width=0.0, label="a"),
             RuntimeMotionWaypoint(pos=np.array([0.5, 0.0, 0.1]), quat=QUAT, finger_width=0.0, label="b",
                                   linear="keep")]
    plain = exe.run("moves", moves, planner, PlanState(np.zeros(9)), params)
    evented = exe.run("moves", [Attach(HELD), moves[0], Detach(), moves[1]], planner, PlanState(np.zeros(9)), params)
    np.testing.assert_array_equal(evented.trajectory, plain.trajectory)
    assert [(s["index"], s["label"], s["span"]) for s in evented.segments] == [
        (s["index"], s["label"], s["span"]) for s in plain.segments]
    assert [s["holding"] for s in evented.segments] == [True, False] and evented.state.holding is None
    assert exe.last_motion_trace["action"] == "moves"
