"""The execution gate's contract, on scenes that report scripted contacts."""

from types import SimpleNamespace

import numpy as np
import pytest

from robotsmith.execution.executor import MotionExecutor
from robotsmith.execution.gate import (ExecutionGate, Infeasible, MotionRejected, Span, StaleMotion,
                                       VerifiedMotion, certified, permitted, spans_of)
from robotsmith.grasp.planner import GraspPlan
from robotsmith.motion.params import MotionParams
from robotsmith.motion.planner import GenesisBackend
from robotsmith.scenario_runtime import runner
from robotsmith.skills import Skill

WALL = (("robot", "left_finger"), ("wall", "base"))


class _Scene:
    """Reports ``hits[i]`` at the i-th queried point and records what each query asked."""

    def __init__(self, hits=None, obj=(np.array([0.5, 0.0, 0.8]), np.array([1.0, 0, 0, 0]))):
        self.hits, self.obj, self.queries, self.writes = hits or {}, obj, [], 0

    def contacts_at(self, qpos, held=None, placed=None):
        self.queries.append((np.asarray(qpos).copy(), held, placed))
        return self.hits.get(len(self.queries) - 1, [])

    def hand_pose(self, qpos):
        return np.array([0.5, 0.0, 0.9]) + [float(qpos[0]), 0, 0], np.array([0.0, 1.0, 0, 0])

    def object_pose(self, name):
        return self.obj

    def state_fingerprint(self):
        return str(self.writes)


Q0 = np.zeros(9)


def _points(*xs):
    return [np.full(9, x) for x in xs]


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


def _pick_motion():
    def ik(pos, quat, finger_pos, *, init_qpos=None):
        q = np.zeros(9)
        q[:3] = pos
        q[7:] = finger_pos
        return q

    params = MotionParams()
    result = MotionExecutor().pick(_pick_plan(), GenesisBackend(ik), np.zeros(9), params)
    transit = _points(0.1, 0.2)
    return transit + result.trajectory, [Span("transit", 0, 2)] + spans_of(result.segments, 2), params


def test_pick_spans_tile_the_whole_trajectory_with_holds_and_rules():
    traj, spans, params = _pick_motion()
    assert spans[-1].end == len(traj) and params.lift_hold_steps > 0
    assert [(s.label, s.holding, s.allow) for s in spans] == [
        ("transit", False, None), ("wp0", False, None), ("wp1", False, None), ("wp2", False, "grip"),
        ("wp3", True, "rest")]


@pytest.mark.parametrize("where", ["transit", "wp2", "tail"])
def test_gate_rejects_a_violation_in_any_span_including_holds(where):
    traj, spans, _ = _pick_motion()
    by = {s.label: s for s in spans}
    index = {"transit": 1, "wp2": by["wp2"].end - 1, "tail": len(traj) - 1}[where]
    palm = (("robot", "hand"), ("wall", "base"))
    with pytest.raises(MotionRejected) as exc:
        ExecutionGate(_Scene({index: [palm]})).verify(traj, spans, target="die", q_start=Q0)
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
    ExecutionGate(scene).verify(_points(0, 0.1, 0.2), [Span("close", 0, 1, allow="grip"),
                                                       Span("lift", 1, 3, holding=True)], target="die", q_start=Q0)
    held = [h for _, h, _ in scene.queries]
    assert held[0] is None and held[1][0] == held[2][0] == "die"
    # T is taken where holding starts from, the closed grip at point 0: hand right over the die,
    # 0.1 m along hand z (top-down quat [0, 1, 0, 0]).
    np.testing.assert_allclose(held[1][1][:3, 3], [0.0, 0.0, 0.1], atol=1e-9)


def test_holding_from_the_first_point_takes_t_at_the_start_configuration():
    scene = _Scene()
    ExecutionGate(scene).verify(_points(0.2, 0.3), [Span("transport", 0, 2, holding=True)], target="die",
                                q_start=np.full(9, 0.1))
    np.testing.assert_allclose(scene.queries[0][1][1][:3, 3], [-0.1, 0.0, 0.1], atol=1e-9)


def test_a_released_object_stays_where_the_hand_left_it():
    hand_on_die = (("robot", "hand"), ("die", "base"))
    scene = _Scene({3: [hand_on_die]})
    spans = [Span("descend", 0, 2, holding=True, allow="rest"), Span("retreat", 2, 4, allow="grip")]
    with pytest.raises(MotionRejected) as exc:
        ExecutionGate(scene).verify(_points(0.1, 0.3, 0.2, 0.0), spans, target="die", q_start=Q0)
    assert exc.value.index == 3 and exc.value.span.label == "retreat" and exc.value.pair == hand_on_die
    placed = [p for _, _, p in scene.queries]
    assert placed[0] is placed[1] is None and placed[2][0] == placed[3][0] == "die"
    # Released at point 1, hand 0.3 m along x from where T was taken: the die moved with it.
    np.testing.assert_allclose(placed[2][1][:3, 3], [0.8, 0.0, 0.8], atol=1e-9)


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

    env = SimpleNamespace(entity_map={}, planner=None, solve_ik=no_ik, state_fingerprint=lambda: "now")
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
    motion = VerifiedMotion(_points(1, 2), (Span("a", 0, 2),), "now", 2)
    run, recorded = _episode(monkeypatch, lambda *a, **k: (motion, None, None))
    run.run_steps([Skill("pick", "die")], [])
    assert recorded == [motion.trajectory] and recorded[0] is motion.trajectory


def test_episode_refuses_a_motion_verified_on_another_state(monkeypatch):
    motion = VerifiedMotion(_points(1), (Span("a", 0, 1),), "before", 1)
    run, recorded = _episode(monkeypatch, lambda *a, **k: (motion, None, None))
    with pytest.raises(StaleMotion):
        run.run_steps([Skill("pick", "die")], [])
    assert recorded == []


def test_rejected_step_records_zero_frames(monkeypatch):
    calls = []

    def plan(skill, *a, **k):
        calls.append(skill.name)
        raise Infeasible("no pick candidate passed the gate and its trial", {"candidates": 2})

    run, recorded = _episode(monkeypatch, plan)
    entry = run.run_steps([Skill("pick", "die"), Skill("place", "mat")], [])
    assert recorded == [] and run.frame_count == 0 and calls == ["pick"]
    assert entry["cause"] == "infeasible" and entry["done"] is False
    assert entry["infeasible"] == {"reason": "no pick candidate passed the gate and its trial", "candidates": 2}
