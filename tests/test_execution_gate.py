"""The execution gate's contract, on scenes that report scripted contacts."""

import contextlib
import dataclasses
import itertools
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from robotsmith.execution.executor import MotionExecutor
from robotsmith.execution.gate import (
    CLEARANCE_M,
    ExecutionFailed,
    ExecutionGate,
    Holding,
    MotionRejected,
    Refused,
    Span,
    StaleMotion,
    VerifiedMotion,
    certified,
    permitted,
    poses_of,
    spans_of,
)
from robotsmith.execution.meta_actions import pick_actions, place_actions
from robotsmith.grasp.planner import GraspPlan
from robotsmith.grasp.transforms import pose_matrix
from robotsmith.motion.params import MotionParams
from robotsmith.motion.plan_state import PlanState
from robotsmith.motion.planner import GenesisBackend
from robotsmith.scenario_runtime import runner
from robotsmith.skills import Skill, run_skills

WALL = (("robot", "left_finger"), ("wall", "base"))


class _Scene:
    """Reports ``hits[i]`` at the i-th queried point and records what each query asked. The hand sits at
    ``(0.5 + q[0], 0, 0.9)`` pointing down."""

    joint_limits = (np.full(9, -10.0), np.full(9, 10.0))
    joined_links = frozenset({frozenset(("link7", "hand")), frozenset(("hand", "left_finger"))})

    def __init__(self, hits=None, obj=(np.array([0.5, 0.0, 0.8]), np.array([1.0, 0, 0, 0]))):
        self.hits, self.obj, self.queries, self.writes = hits or {}, obj, [], 0
        self.clearance = []

    def contacts_at(self, qpos, held=None, placed=None):
        self.queries.append((np.asarray(qpos).copy(), held, placed))
        return self.hits.get(len(self.queries) - 1, [])

    def hand_pose(self, qpos):
        return np.array([0.5, 0.0, 0.9]) + [float(qpos[0]), 0, 0], np.array([0.0, 1.0, 0, 0])

    def object_pose(self, name):
        return self.obj

    def lever_arms(self, held=None):
        return np.zeros(9)

    def clearance_at(self, qpos, held=None, within=None):
        return self.clearance

    def state_fingerprint(self):
        return str(self.writes)


Q0 = np.zeros(9)


def _points(*xs):
    return [np.full(9, x) for x in xs]


def _holding_at(scene, q, target="die"):
    """The target held where it is now, by the hand at ``q``."""
    return Holding(target, np.linalg.inv(pose_matrix(*scene.hand_pose(q))) @ pose_matrix(*scene.object_pose(target)))


def test_gate_rejects_a_path_that_crosses_a_wall_between_legal_endpoints():
    scene = _Scene({1: [WALL]})
    with pytest.raises(MotionRejected) as exc:
        ExecutionGate(scene).verify(_points(0, 1, 0), [Span("move", 0, 3)], target="die", q_start=Q0)
    assert exc.value.index == 1 and exc.value.pair == WALL

    motion = ExecutionGate(_Scene()).verify(_points(0, 0), [Span("move", 0, 2)], target="die", q_start=Q0)
    assert motion.checked == 2 and [q[0] for q in motion.trajectory] == [0, 0]


def test_gate_checks_every_trajectory_point():
    scene = _Scene()
    motion = ExecutionGate(scene).verify(_points(*range(7)), [Span("a", 0, 3), Span("b", 3, 7)], target="die", q_start=Q0)
    assert [float(q[0]) for q, _, _ in scene.queries] == list(range(7)) and motion.checked == 7


def test_verify_asks_the_budget_before_and_after_every_point():
    scene, calls = _Scene(), []

    class Budget:
        def spend(self):
            calls.append(("spend", len(scene.queries)))
            if len(scene.queries) == 2:
                raise TimeoutError

        def check(self):
            calls.append(("check", len(scene.queries)))

    gate = ExecutionGate(scene)
    motion = gate.verify(_points(0, 0), [Span("move", 0, 2)], target="die", q_start=Q0, budget=Budget())
    assert calls == [("spend", 0), ("check", 1), ("spend", 1), ("check", 2)] and motion.checked == 2
    scene.queries.clear()
    with pytest.raises(TimeoutError):
        gate.verify(_points(0, 0, 0), [Span("move", 0, 3)], target="die", q_start=Q0, budget=Budget())
    assert len(scene.queries) == 2


def test_gate_refuses_spans_that_leave_points_unchecked():
    with pytest.raises(ValueError):
        ExecutionGate(_Scene()).verify(_points(0, 1, 2), [Span("a", 0, 2)], target="die", q_start=Q0)


def _pick_plan():
    quat = np.array([0.0, 1.0, 0.0, 0.0])
    return GraspPlan(
        grasp_pos=np.array([0.5, 0.0, 0.85]), grasp_quat=quat, pre_grasp_pos=np.array([0.5, 0.0, 0.95]),
        pre_grasp_quat=quat, retreat_pos=np.array([0.5, 0.0, 1.0]), retreat_quat=quat,
        finger_open=0.04, finger_closed=0.0, waypoints=[
            SimpleNamespace(pos=np.array([0.5, 0.0, 0.95]), quat=quat, finger_width=0.04),
            SimpleNamespace(pos=np.array([0.5, 0.0, 0.85]), quat=quat, finger_width=0.04),
            SimpleNamespace(pos=np.array([0.5, 0.0, 0.85]), quat=quat, finger_width=0.0),
            SimpleNamespace(pos=np.array([0.5, 0.0, 1.0]), quat=quat, finger_width=0.0),
        ])


def _ik(pos, quat, finger_pos, *, init_qpos=None):
    q = np.zeros(9)
    q[:3] = pos
    q[7:] = finger_pos
    return q


def _at_first_waypoint(plan, finger):
    return _ik(plan.waypoints[0].pos, plan.waypoints[0].quat, finger)


GRASP = Holding("die", np.eye(4))


def _pick(plan, start, params):
    """From the arm at the pre-grasp, so the approach is empty."""
    planner = GenesisBackend(_ik)
    return MotionExecutor().run("pick", pick_actions(plan, start, (plan.pre_grasp_pos, plan.pre_grasp_quat), (),
                                                     params, planner.solve_ik, grasp=GRASP),
                                planner, PlanState(start), params)


def _pick_motion():
    params = MotionParams()
    plan = _pick_plan()
    result = _pick(plan, _at_first_waypoint(plan, 0.0), params)
    transit = _points(0.1, 0.2)
    return transit + result.trajectory, [Span("transit", 0, 2)] + spans_of(result.segments, 2), params


def test_pick_spans_tile_the_whole_trajectory_with_holds_and_rules():
    traj, spans, params = _pick_motion()
    assert spans[-1].end == len(traj) and params.lift_hold_steps > 0
    assert [(s.label, s.holding, s.allow) for s in spans] == [
        ("transit", False, None), ("open", False, None), ("approach", False, None), ("descend", False, None),
        ("close", False, "grip"), ("lift", True, "rest")]


def test_pick_poses_are_the_waypoints_at_the_points_that_reach_them():
    params = MotionParams()
    plan = _pick_plan()
    result = _pick(plan, _at_first_waypoint(plan, 0.0), params)
    poses = poses_of(result.segments, 2)
    wanted = [plan.pre_grasp_pos, plan.grasp_pos, plan.grasp_pos, plan.grasp_pos + [0.0, 0.0, 0.10]]
    assert len(poses) == len(wanted)
    traj = _points(0.1, 0.2) + result.trajectory
    for (i, (pos, quat)), want in zip(sorted(poses.items()), wanted):
        np.testing.assert_allclose(traj[i][:3], want)
        np.testing.assert_allclose(pos, want)
        np.testing.assert_allclose(quat, plan.grasp_quat)
    # The descend's intermediate points ask for no pose.
    descend = next(s for s in spans_of(result.segments, 2) if s.label == "descend")
    assert [i for i in range(descend.start, descend.end) if i in poses] == [descend.end - 1]


def test_pick_holds_the_arm_at_its_start_while_the_fingers_open():
    plan, params = _pick_plan(), MotionParams()
    start = _at_first_waypoint(plan, 0.0)
    result = _pick(plan, start, params)
    first = spans_of(result.segments)[0]
    assert first.end == params.finger_steps
    assert all(np.allclose(q[:7], start[:7]) for q in result.trajectory[:first.end])
    opened = _pick(plan, _at_first_waypoint(plan, 0.04), params)
    assert spans_of(opened.segments)[0].end == 0


@pytest.mark.parametrize("where", ["transit", "close", "tail"])
def test_gate_rejects_a_violation_in_any_span_including_holds(where):
    traj, spans, _ = _pick_motion()
    by = {s.label: s for s in spans}
    index = {"transit": 1, "close": by["close"].end - 1, "tail": len(traj) - 1}[where]
    palm = (("robot", "hand"), ("wall", "base"))
    scene = _Scene({index: [palm]})
    holding = _holding_at(scene, traj[by["lift"].start - 1])
    with pytest.raises(MotionRejected) as exc:
        ExecutionGate(scene).verify(traj, spans, target="die", q_start=Q0, holding=holding)
    assert exc.value.index == index


@pytest.mark.parametrize("pair, span, ok", [
    ((("robot", "left_finger"), ("die", "base")), Span("close", 0, 1, allow="grip"), True),
    ((("robot", "left_finger"), ("die", "base")), Span("lift", 0, 1, holding=True), True),
    ((("robot", "right_finger"), ("die", "base")), Span("approach", 0, 1), False),
    ((("robot", "hand"), ("die", "base")), Span("close", 0, 1, allow="grip"), False),
    ((("die", "base"), ("env", "table")), Span("descend", 0, 1, holding=True, allow="rest"), True),
    ((("die", "base"), ("mat", "base")), Span("descend", 0, 1, holding=True, allow="rest"), True),
    ((("die", "base"), ("env", "table")), Span("transport", 0, 1, holding=True), False),
    ((("die", "base"), ("wall", "base")), Span("descend", 0, 1, holding=True, allow="rest"), False),
    ((("die", "base"), ("env", "table")), Span("approach", 0, 1), True),
    ((("robot", "hand"), ("robot", "link7")), Span("approach", 0, 1), True),
    ((("robot", "link7"), ("env", "table")), Span("approach", 0, 1), False),
])
def test_gate_contact_rules_per_segment(pair, span, ok):
    assert permitted(pair, span, "die", {"env", "mat"}) is ok


@pytest.mark.parametrize("pair, span, gate_ok, cert_ok", [
    ((("die", "base"), ("wall", "base")), Span("approach", 0, 1), True, False),
    ((("die", "base"), ("env", "table")), Span("approach", 0, 1), True, True),
    ((("die", "base"), ("mat", "base")), Span("retreat", 0, 1, allow="grip"), True, True),
    ((("robot", "left_finger"), ("die", "base")), Span("retreat", 0, 1, allow="grip"), True, True),
    ((("robot", "hand"), ("die", "base")), Span("retreat", 0, 1, allow="grip"), False, False),
    ((("wall", "base"), ("env", "table")), Span("approach", 0, 1), True, True),
])
def test_certification_also_rejects_a_free_target_meeting_an_obstacle(pair, span, gate_ok, cert_ok):
    assert permitted(pair, span, "die", {"env", "mat"}) is gate_ok
    assert certified(pair, span, "die", {"env", "mat"}) is cert_ok


def test_gate_checks_the_held_object_only_while_holding():
    scene = _Scene()
    holding = _holding_at(scene, _points(0)[0])
    ExecutionGate(scene).verify(_points(0, 0.1, 0.2), [Span("close", 0, 1, allow="grip"),
                                                       Span("lift", 1, 3, holding=True)], target="die", q_start=Q0,
                                holding=holding)
    held = [h for _, h, _ in scene.queries]
    assert held[0] is None and held[1][0] == held[2][0] == "die"
    # The hand right over the die, 0.1 m along hand z (top-down quat [0, 1, 0, 0]).
    np.testing.assert_allclose(held[1][1][:3, 3], [0.0, 0.0, 0.1], atol=1e-9)


def test_holding_is_checked_where_it_attaches():
    carry = [Span("close", 0, 1, allow="grip"), Span("lift", 1, 3, holding=True)]
    scene = _Scene()
    good = _holding_at(scene, _points(0)[0])
    ExecutionGate(scene).verify(_points(0, 0.1, 0.2), carry, target="die", q_start=Q0, holding=good)

    off = Holding("die", good.t_hand_object @ pose_matrix(np.array([0.0, 0.0, 0.01])))
    with pytest.raises(MotionRejected) as exc:
        ExecutionGate(_Scene()).verify(_points(0, 0.1, 0.2), carry, target="die", q_start=Q0, holding=off)
    assert exc.value.index == 1 and exc.value.rejection["reason"] == "binding_mismatch"
    assert exc.value.rejection["position_m"] == pytest.approx(0.01)

    # Holding from the first point attaches at the start configuration.
    start = np.full(9, 0.1)
    from_start = [Span("transport", 0, 2, holding=True)]
    ExecutionGate(scene).verify(_points(0.2, 0.3), from_start, target="die", q_start=start,
                                holding=_holding_at(scene, start))
    with pytest.raises(MotionRejected):
        ExecutionGate(scene).verify(_points(0.2, 0.3), from_start, target="die", q_start=start, holding=good)
    with pytest.raises(ValueError):
        ExecutionGate(scene).verify(_points(0.2, 0.3), from_start, target="die", q_start=start)


def test_a_released_object_stays_where_the_hand_left_it():
    hand_on_die = (("robot", "hand"), ("die", "base"))
    scene = _Scene({3: [hand_on_die]})
    spans = [Span("descend", 0, 2, holding=True, allow="rest"), Span("retreat", 2, 4, allow="grip")]
    with pytest.raises(MotionRejected) as exc:
        ExecutionGate(scene).verify(_points(0.1, 0.3, 0.2, 0.0), spans, target="die", q_start=Q0,
                                    holding=_holding_at(scene, Q0))
    assert exc.value.index == 3 and exc.value.span.label == "retreat" and exc.value.pair == hand_on_die
    placed = [p for _, _, p in scene.queries]
    assert placed[0] is placed[1] is None and placed[2][0] == placed[3][0] == "die"
    # Released at point 1, hand 0.3 m along x from where T was taken: the die moved with it.
    np.testing.assert_allclose(placed[2][1][:3, 3], [0.8, 0.0, 0.8], atol=1e-9)


def test_the_release_is_checked_with_the_object_placed():
    params, plan, planner, scene = MotionParams(), _pick_plan(), GenesisBackend(_ik), _Scene()
    start = _ik(plan.pre_grasp_pos, plan.pre_grasp_quat, 0.0)
    holding = _holding_at(scene, start)
    result = MotionExecutor().run("place", place_actions(plan, params), planner, PlanState(start, holding), params)
    spans = spans_of(result.segments)
    release = next(s for s in spans if s.label == "release")
    scene.hits = {release.start: [(("robot", "left_finger"), ("die", "base"))]}
    ExecutionGate(scene).verify(result.trajectory, spans, target="die", q_start=start, holding=holding)
    asked = scene.queries[release.start:release.end]
    assert len(asked) == params.release_steps
    assert all(held is None and placed[0] == "die" for _, held, placed in asked)


def test_stale_verified_motion_is_not_executed():
    scene = _Scene()
    gate = ExecutionGate(scene)
    motion = gate.verify(_points(0), [Span("a", 0, 1)], target="die", q_start=Q0)
    gate.check_current(motion)
    scene.writes += 1
    with pytest.raises(StaleMotion):
        gate.check_current(motion)


def _episode(monkeypatch, plan):
    def no_ik(*a, **k):
        raise AssertionError("the episode re-solved IK")

    env = SimpleNamespace(entity_map={}, planner=None, solve_ik=no_ik, state_fingerprint=lambda: "now", writes=0)
    recorded = []

    def record(env, dataset, seg, task_spec, **kw):
        recorded.append(seg)
        return []

    def judge(env, entry, *a, **k):
        entry.update(done=True, completion={})

    monkeypatch.setattr(runner, "plan_segment", plan)
    monkeypatch.setattr(runner, "_live_scene_state", lambda *a, **k: {"positions": {}, "joint_positions": {}})
    monkeypatch.setattr(runner, "_judge_completion", judge)
    run = runner.EpisodeRun(env, object(), executor=None, motion_params=None, dataset=None, record_fn=record,
                            target_frames={}, camera_names=(), env_state_names=())
    return run, recorded


def test_episode_replays_the_verified_trajectory_without_ik(monkeypatch):
    issued = []

    def plan(*a, gate, **k):
        issued.append(gate.ungated(_points(1, 2)))
        return issued[-1], None, None

    run, recorded = _episode(monkeypatch, plan)
    run.run_steps([Skill("pick", "die")], [])
    assert len(recorded) == 1
    np.testing.assert_array_equal(recorded[0], issued[0].trajectory)


def test_episode_refuses_a_motion_verified_on_another_state(monkeypatch):
    def plan(*a, gate, **k):
        motion = gate.ungated(_points(1))
        gate.env.state_fingerprint = lambda: "after"
        return motion, None, None

    run, recorded = _episode(monkeypatch, plan)
    with pytest.raises(StaleMotion):
        run.run_steps([Skill("pick", "die")], [])
    assert recorded == []


def test_rejected_step_records_zero_frames(monkeypatch):
    calls = []

    def plan(skill, *a, **k):
        calls.append(skill.name)
        raise Refused("search_exhausted", {"reason": "no pick candidate passed the gate and its trial",
                                           "candidates": 2})

    run, recorded = _episode(monkeypatch, plan)
    entry = run.run_steps([Skill("pick", "die"), Skill("place", "mat")], [])
    assert recorded == [] and run.frame_count == 0 and calls == ["pick"]
    assert entry["cause"] == "search_exhausted" and entry["done"] is False and entry["steps"] == 0
    assert entry["evidence"] == {"reason": "no pick candidate passed the gate and its trial", "candidates": 2}
    assert run.result().stopped is entry


def test_check_point_rejects_bad_numbers_limits_fk_residual_and_self_collision():
    scene = _Scene()
    scene.joint_limits = (np.full(9, -1.0), np.full(9, 1.0))
    gate, span, q = ExecutionGate(scene), Span("move", 0, 1), np.full(9, 0.2)
    check = lambda q, **kw: gate.check_point(q, span, target="die", **kw)

    assert check(q) is None
    assert check(np.r_[q[:8], np.nan])["reason"] == "invalid_value"
    assert check(q[:7])["reason"] == "invalid_value"
    assert check(np.r_[1.5, q[1:]]) == {"reason": "joint_limit", "joints": [0]}

    hand = scene.hand_pose(q)
    assert check(q, pose=hand) is None
    assert check(q, pose=(hand[0] + [0.0, 0.01, 0.0], hand[1]))["reason"] == "fk_residual"
    tilted = np.array([0.0, np.cos(0.05), np.sin(0.05), 0.0])
    assert check(q, pose=(hand[0], tilted))["reason"] == "fk_residual"

    joined = (("robot", "link7"), ("robot", "hand"))
    apart = (("robot", "link3"), ("robot", "link5"))
    scene.hits = {len(scene.queries): [joined]}
    assert check(q) is None
    scene.hits = {len(scene.queries): [apart]}
    assert check(q) == {"reason": "self_collision", "pair": apart}

    # Only points that ask for a pose are held to one: a free point between two posed ones is not.
    ends = {0: scene.hand_pose(_points(0)[0]), 2: scene.hand_pose(_points(0.4)[0])}
    ExecutionGate(_Scene()).verify(_points(0, 0.2, 0.4), [Span("move", 0, 3)], target="die", q_start=Q0, poses=ends)
    with pytest.raises(MotionRejected) as exc:
        ExecutionGate(_Scene()).verify(_points(0, 0.2, 0.4), [Span("move", 0, 3)], target="die", q_start=Q0,
                                       poses={1: ends[2]})
    assert exc.value.index == 1 and exc.value.rejection["reason"] == "fk_residual"


class _StateScene(_Scene):
    """Contacts that depend on what is posed: the hand meets a wall at x 0.25, the held die meets it at 0.55,
    and the hand passes through the released die at 0.85."""

    def contacts_at(self, qpos, held=None, placed=None):
        self.queries.append((np.asarray(qpos).copy(), held, placed))
        x = round(float(qpos[0]), 2)
        if x == 0.25:
            return [(("robot", "hand"), ("wall", "base"))]
        if held and x == 0.55:
            return [(("die", "base"), ("wall", "base"))]
        if placed and x == 0.85:
            return [(("robot", "hand"), ("die", "base"))]
        return []


def test_check_point_and_verify_agree_empty_held_and_released():
    spans = [Span("approach", 0, 2), Span("close", 2, 3, allow="grip"), Span("carry", 3, 5, holding=True),
             Span("retreat", 5, 7, allow="grip")]
    clean = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for bad, index in ((None, None), (0.25, 1), (0.55, 4), (0.85, 6)):
        xs = list(clean)
        if index is not None:
            xs[index] = bad
        scene = _StateScene()
        gate = ExecutionGate(scene)
        holding = _holding_at(scene, _points(xs[2])[0])
        if index is None:
            gate.verify(_points(*xs), spans, target="die", q_start=Q0, holding=holding)
        else:
            with pytest.raises(MotionRejected) as exc:
                gate.verify(_points(*xs), spans, target="die", q_start=Q0, holding=holding)
            assert exc.value.index == index
        asked = list(scene.queries)
        assert all((h is not None) == (3 <= i < 5) and (p is not None) == (i >= 5) for i, (_, h, p) in enumerate(asked))
        # The same point asked alone, posed as verify posed it, gets the same answer.
        for i, (q, held, placed) in enumerate(asked):
            span = next(s for s in spans if s.start <= i < s.end)
            why = gate.check_point(q, span, target="die", held=held, placed=placed)
            assert (why is None) == (i != index)
            if i == index:
                assert why == exc.value.rejection


def test_a_verified_motion_cannot_be_changed_after_issue():
    scene = _Scene()
    traj = _points(0, 0.1, 0.2)
    t = _holding_at(scene, traj[0]).t_hand_object.copy()
    motion = ExecutionGate(scene).verify(traj, [Span("close", 0, 1, allow="grip"), Span("lift", 1, 3, holding=True)],
                                         target="die", q_start=Q0, holding=Holding("die", t))
    traj[1][:] = 9.0
    t[:] = 0.0
    assert float(motion.trajectory[1][0]) == 0.1 and motion.holding.t_hand_object[3, 3] == 1.0
    with pytest.raises(ValueError):
        motion.trajectory[0][0] = 5.0
    with pytest.raises(ValueError):
        motion.holding.t_hand_object[0, 0] = 5.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        motion.trajectory = ()
    with pytest.raises(AttributeError):
        motion.trajectory.append(np.zeros(9))


def test_issued_arrays_cannot_be_made_writable_through_flags_views_or_buffers():
    scene = _Scene()
    traj = _points(0, 0.1, 0.2)
    motion = ExecutionGate(scene).verify(traj, [Span("close", 0, 1, allow="grip"), Span("lift", 1, 3, holding=True)],
                                         target="die", q_start=Q0, holding=_holding_at(scene, traj[0]))
    reads = [lambda: motion.trajectory[1], lambda: motion.holding.t_hand_object]
    before = [read().copy() for read in reads]
    for read in reads:
        a = read()
        for target in (a, a.view(), a.reshape(-1), a.base):
            with pytest.raises(ValueError):
                target.setflags(write=True)
        with pytest.raises(TypeError):
            memoryview(a).cast("B")[0] = 1
        with pytest.raises(TypeError):
            memoryview(a.base.base).cast("B")[0] = 1
        # NumPy 2.5 deprecates assigning dtype / shape; a later NumPy raises instead.
        with warnings.catch_warnings(), contextlib.suppress(AttributeError):
            warnings.simplefilter("ignore", DeprecationWarning)
            a.dtype, a.shape = np.int64, (-1,)
    for read, was in zip(reads, before):
        assert read().dtype == np.float64
        np.testing.assert_array_equal(read(), was)


def test_only_the_gate_issues_a_motion_it_accepts():
    scene = _Scene()
    gate = ExecutionGate(scene)
    forged = VerifiedMotion(tuple(_points(1)), (Span("a", 0, 1),), scene.state_fingerprint(), 1)
    with pytest.raises(StaleMotion):
        gate.check_current(forged)
    other = ExecutionGate(scene).verify(_points(0), [Span("a", 0, 1)], target="die", q_start=Q0)
    with pytest.raises(StaleMotion):
        gate.check_current(other)
    mine = gate.verify(_points(0), [Span("a", 0, 1)], target="die", q_start=Q0)
    gate.check_current(mine)
    scene.writes += 1
    with pytest.raises(StaleMotion):
        gate.check_current(mine)


class _Planner:
    def plan(self, *a, **k):
        quat = np.array([0.0, 1.0, 0.0, 0.0])
        return [SimpleNamespace(grasp_pos=np.array([0.5, 0.0, 0.85]), grasp_quat=quat, metadata={"candidate_index": i},
                                pre_grasp_pos=np.array([0.5, 0.0, 0.95]), pre_grasp_quat=quat, finger_open=0.04,
                                finger_closed=0.0)
                for i in range(2)]


class _Executor:
    """The pick list's planned approach, then two fixed points."""

    last_motion_trace = None

    def run(self, name, actions, *a, **k):
        approach = [np.asarray(q) for q in actions[1].path]
        n = len(approach)
        return SimpleNamespace(trajectory=approach + _points(0.1, 0.2), segments=[
            {"label": "approach", "span": [0, n], "holding": False, "allow": None},
            {"label": "seg", "span": [n, n + 2], "holding": False, "allow": None}])


def test_a_refused_step_steps_no_physics():
    with pytest.raises(ValueError):
        Refused("blocked", {})

    scene, trials = _Scene(), []
    scene.contacts_at = lambda q, held=None, placed=None: [(("robot", "hand"), ("cage", "base"))]

    def trial(traj, target):
        trials.append(target)
        scene.writes += len(traj)
        return {"lifted": False, "lift_m": 0.0, "seconds": 0.1}

    state = {"home_qpos": Q0, "positions": {"die": np.array([0.5, 0.0, 0.8])}, "grasp_trial": trial}
    with pytest.raises(Refused) as exc:
        run_skills([Skill("pick", "die")], _Planner(), _Executor(), lambda *a, **k: Q0, state, MotionParams(),
                   gate=ExecutionGate(scene))
    assert exc.value.label == "search_exhausted" and trials == [] and scene.writes == 0
    assert [t["gate"]["reason"] for t in exc.value.evidence["trials"]] == ["collision", "collision"]

    # Trials that stepped physics and lifted nothing are a failed execution, not a zero-step refusal.
    del scene.contacts_at
    with pytest.raises(ExecutionFailed) as failed:
        run_skills([Skill("pick", "die")], _Planner(), _Executor(), lambda *a, **k: Q0, dict(state), MotionParams(),
                   gate=ExecutionGate(scene))
    assert failed.value.label == "execution_failed" and trials == ["die", "die"] and failed.value.steps == 6


def test_a_free_span_point_inside_the_clearance_of_a_fixture_is_refused():
    scene = _Scene()
    scene.clearance = [("hand", "wall", CLEARANCE_M / 2)]
    why = ExecutionGate(scene).check_point(Q0, Span("approach", 0, 1), target="die")
    assert why["reason"] == "clearance" and why["link"] == "hand" and why["entity"] == "wall"
    assert why["distance_m"] == pytest.approx(CLEARANCE_M / 2)


def test_a_free_span_point_beyond_twice_the_clearance_passes():
    scene = _Scene()
    scene.clearance = [("hand", "wall", 2 * CLEARANCE_M)]
    assert ExecutionGate(scene).check_point(Q0, Span("transport", 0, 1), target="die") is None


def test_the_hand_near_a_fixture_in_a_contact_span_is_judged_by_contact_only():
    scene = _Scene()
    scene.clearance = [("hand", "wall", CLEARANCE_M / 2), ("left_finger", "wall", CLEARANCE_M / 2)]
    assert ExecutionGate(scene).check_point(Q0, Span("descend", 0, 1), target="die") is None


def test_an_arm_link_inside_the_clearance_in_a_contact_span_is_refused():
    scene = _Scene()
    scene.clearance = [("link5", "barrier", CLEARANCE_M / 2)]
    why = ExecutionGate(scene).check_point(Q0, Span("lift", 0, 1), target="die")
    assert why["reason"] == "clearance" and why["link"] == "link5" and why["entity"] == "barrier"


def test_the_held_object_is_not_held_to_the_clearance():
    scene = _Scene()
    scene.clearance = [("die", "wall", CLEARANCE_M / 2), ("link5", "wall", 2 * CLEARANCE_M)]
    assert ExecutionGate(scene).check_point(Q0, Span("transport", 0, 1, holding=True), target="die") is None


def test_verify_checks_configurations_between_points_without_adding_any():
    scene = _Scene()
    lever = np.zeros(9)
    lever[3] = 1.0
    scene.lever_arms = lambda held=None: lever
    start, end = np.zeros(9), np.zeros(9)
    end[0], end[3] = 0.0005, 0.04
    hand_move = np.linalg.norm(scene.hand_pose(end)[0] - scene.hand_pose(start)[0])
    assert hand_move < 0.001 and float(lever @ np.abs(end - start)) > 0.02
    motion = ExecutionGate(scene).verify([start, end], [Span("approach", 0, 2)], target="die", q_start=start)
    assert len(motion.trajectory) == 2
    np.testing.assert_allclose(motion.trajectory[0], start)
    np.testing.assert_allclose(motion.trajectory[1], end)
    asked = [q for q, _, _ in scene.queries]
    assert len(asked) > 2
    np.testing.assert_allclose(asked[0], start)
    np.testing.assert_allclose(asked[-1], end)
    for left, right in itertools.pairwise(asked):
        assert float(lever @ np.abs(right - left)) <= 0.005 + 1e-8
