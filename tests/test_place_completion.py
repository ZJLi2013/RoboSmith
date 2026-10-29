import numpy as np
import pytest

from robotsmith.scenario_runtime import runner
from robotsmith.scenario_runtime.runner import evaluate_scenario_task, run_scenario_episode
from robotsmith.sim.world_state import settle_window
from robotsmith.skills import SKILL_DEFS, Skill, bound_leaf
from robotsmith.tasks import evaluate_predicate, failed_leaves
from robotsmith.tasks.task_spec import SuccessNode, TaskSpec

ON = {"object": "die", "parent": "supporter", "box": [[-0.05, -0.05, 0.0], [0.05, 0.05, 0.05]],
      "yaw_deg": 0.0, "mode": "on", "target": "shelf"}
BOUND = {"status": "bound", "predicate": "object_in_box", "params": ON}
UNCHECKED = {"status": "unchecked", "reason": "not migrated"}
# 0.5 s at dt 1/30 is 15 held frames; these are the 10 sampled ones, counted from the hold's start.
MARKS = [2, 3, 4, 6, 8, 9, 10, 12, 14, 15]


class _Ent:
    def __init__(self, pos, robot=False):
        self.pos, self.robot = np.asarray(pos, dtype=float), robot

    def get_pos(self):
        return self.pos

    def get_quat(self):
        return np.array([1.0, 0.0, 0.0, 0.0])

    def get_AABB(self):
        return np.stack([self.pos - 0.01, self.pos + 0.01])

    def get_contacts(self, with_entity=None):
        hit = with_entity is not self and (isinstance(with_entity, _Ent) or self.robot)
        return {"link_a": np.zeros(1 if hit else 0)}


class _Scene:
    dt = 1.0 / 30.0

    def __init__(self):
        self.t = 0

    def step(self):
        self.t += 1


class _Franka:
    def control_dofs_position(self, q, dofs):
        pass


class _Env:
    """The die rests on the supporter's box; ``robot`` makes the arm still touch it."""

    motors_dof = list(range(9))

    def __init__(self, robot=False):
        self.scene, self.franka = _Scene(), _Franka()
        self.entity_map = {"die": _Ent([0.56, 0.0, 0.835], robot), "supporter": _Ent([0.56, 0.0, 0.775])}
        self.placed_map, self.object_heights, self.asset_map, self.object_quats = {}, {}, {}, {}
        self.table_surface_z, self.articulated_joint_dofs, self.planner = 0.775, {}, object()

    def reset(self, obj_xyz_map, *, settle_steps):
        pass

    def get_joint_positions(self):
        return {}


def _run(monkeypatch, steps, bindings, *, robot=False, phases=None):
    """Drive an episode whose every phase records 2 frames; return env, task, result and recorded lengths."""
    monkeypatch.setattr(runner, "plan_segment", lambda skill, *a, **kw: ([np.zeros(9)] * 2, None, None))
    if phases is not None:
        monkeypatch.setattr(runner, "skill_phases", lambda skill, frame: phases.get(skill.name, ["all"]))
    recorded = []

    def record(env, dataset, seg, task_spec, **kw):
        env.scene.t += len(seg)
        recorded.append(len(seg))
        return []

    env = _Env(robot)
    task = TaskSpec(name="t", scene="t", instruction="", success=SuccessNode.leaf("object_in_box", ON),
                    skills=[Skill(name, target) for name, target in steps], step_bindings=bindings)
    result = run_scenario_episode(env, task, executor=object(), motion_params=object(), dataset=object(),
                                  record_fn=record)
    return env, task, result, recorded


PICK_PLACE = [("pick", "die"), ("place", "supporter.shelf")]
PICK_PLACE_CLOSE = PICK_PLACE + [("close", "drawer")]


def test_place_binds_its_leaf_and_unchecked_claims_nothing():
    assert SKILL_DEFS["place"].post is bound_leaf and SKILL_DEFS["place"].settle_s == 0.5
    assert SKILL_DEFS["pick"].post is None
    place = Skill("place", "supporter.shelf")
    assert bound_leaf(place, BOUND).to_dict() == SuccessNode.leaf("object_in_box", ON).to_dict()
    assert bound_leaf(place, UNCHECKED) is None


@pytest.mark.parametrize("robot", [False, True])
def test_place_completion_equals_the_task_leaf_on_the_same_state(monkeypatch, robot):
    _, _, result, _ = _run(monkeypatch, PICK_PLACE, [UNCHECKED, BOUND], robot=robot)
    place = result.handoffs[-1]
    assert place["done"] is (not robot)
    assert evaluate_predicate("object_in_box", place["completion"]["judge_state"], ON) is place["done"]
    assert place.get("cause") == ("release?" if robot else None)


def test_unchecked_place_records_why_and_does_not_settle(monkeypatch):
    env, _, result, recorded = _run(monkeypatch, PICK_PLACE, [UNCHECKED, UNCHECKED])
    place = result.handoffs[-1]
    assert place["done"] is None and place["completion"] == UNCHECKED
    assert recorded == [2, 2] and result.final_window is None


def test_transport_is_not_judged_and_descend_is(monkeypatch):
    _, _, result, _ = _run(monkeypatch, PICK_PLACE, [UNCHECKED, BOUND], phases={"place": ["transport", "descend"]})
    transport, descend = result.handoffs[1:]
    assert transport["done"] is None and "completion" not in transport
    assert descend["completion"]["status"] == "bound" and descend["done"] is True


def test_zero_settle_steps_and_records_nothing():
    env, calls = _Env(), []
    assert settle_window(env, np.zeros(9), settle_s=0.0, step=calls.append)["steps"] == []
    assert calls == [] and env.scene.t == 0


def test_mid_episode_hold_is_recorded_counted_and_numbered(monkeypatch):
    _, _, result, recorded = _run(monkeypatch, PICK_PLACE_CLOSE, [UNCHECKED, BOUND, UNCHECKED])
    c = result.handoffs[1]["completion"]
    assert recorded == [2, 2] + [1] * 15 + [2]
    assert c["settle_frames"] == 15 and result.frame_count == 6 + 15
    assert c["judge_state"]["window"]["steps"] == [4 + m for m in MARKS]
    assert c["judge_state"]["step"] == 19
    assert result.final_window is None


def test_last_step_place_shares_its_window_with_the_task(monkeypatch):
    env, task, result, _ = _run(monkeypatch, PICK_PLACE, [UNCHECKED, BOUND])
    t = env.scene.t
    ok, state = evaluate_scenario_task(env, task, initial_positions={}, hold=result.final_qpos,
                                       window=result.final_window)
    assert env.scene.t == t and ok is True
    assert state["window"] == result.handoffs[-1]["completion"]["judge_state"]["window"]


def test_close_after_place_rejudges_the_task_on_a_new_settle(monkeypatch):
    env, task, result, _ = _run(monkeypatch, PICK_PLACE_CLOSE, [UNCHECKED, BOUND, UNCHECKED])
    t = env.scene.t
    ok, state = evaluate_scenario_task(env, task, initial_positions={}, hold=result.final_qpos,
                                       window=result.final_window)
    assert env.scene.t == t + 15 and ok is True
    assert state["window"]["steps"] == [t + m for m in MARKS]


def test_failed_leaves_name_what_works_against_the_verdict():
    state = {"object_positions": {"die": [0.0, 0.0, 1.0]}, "initial_positions": {"die": [0.0, 0.0, 0.8]}}
    up = SuccessNode.leaf("object_above", {"object": "die", "reference": "table", "z_margin": 0.05})
    high = SuccessNode.leaf("object_above", {"object": "die", "reference": "table", "z_margin": 0.5})
    assert failed_leaves(SuccessNode(op="all", terms=[up, high]), state) == [high.to_dict()]
    assert failed_leaves(SuccessNode(op="all", terms=[up, SuccessNode(op="not", terms=[up])]), state) == [up.to_dict()]
