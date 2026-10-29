"""pick is judged by both fingers on its target and a lift from the step's own start."""

from types import SimpleNamespace

import numpy as np
import pytest

from robotsmith.scenario_runtime import runner
from robotsmith.scenario_runtime.runner import evaluate_scenario_task, run_scenario_episode
from robotsmith.sim.world_state import collect_world_state
from robotsmith.skills import SKILL_DEFS, Skill, bound_leaf
from robotsmith.tasks import evaluate_predicate, evaluate_success
from robotsmith.tasks.predicates import JudgeInputError
from robotsmith.tasks.task_spec import SuccessNode, TaskSpec

GRASPED = {"status": "bound", "predicate": "grasped", "params": {"object": "die", "lift": 0.02}}


def _state(fingers, z, start_z=0.80, other=()):
    return {"finger_contacts": {"die": list(fingers), "mug": list(other)},
            "object_positions": {"die": [0.5, 0.0, z]}, "step_start_positions": {"die": [0.5, 0.0, start_z]}}


@pytest.mark.parametrize("fingers, z, other, expected", [
    (["left_finger", "right_finger"], 0.85, (), True),
    (["left_finger"], 0.85, (), False),
    (["left_finger", "left_finger"], 0.85, (), False),
    (["left_finger", "right_finger"], 0.81, (), False),
    ([], 0.85, ("left_finger", "right_finger"), False),
], ids=["two fingers lifted", "one finger", "one finger twice", "pinched on the table", "fingers on another object"])
def test_grasped(fingers, z, other, expected):
    assert evaluate_predicate("grasped", _state(fingers, z, other=other), GRASPED["params"]) is expected


@pytest.mark.parametrize("drop", ["finger_contacts", "step_start_positions"])
def test_missing_observation_raises_even_under_not(drop):
    state = _state(["left_finger", "right_finger"], 0.85)
    state[drop] = {}
    with pytest.raises(JudgeInputError):
        evaluate_success(SuccessNode(op="not", terms=[SuccessNode.leaf("grasped", GRASPED["params"])]), state)


class _Link:
    def __init__(self, idx, name):
        self.idx, self.name = idx, name


class _Ent:
    def __init__(self, pos, links):
        self.pos, self.links = np.asarray(pos, dtype=float), links

    def get_pos(self):
        return self.pos

    def get_quat(self):
        return np.array([1.0, 0.0, 0.0, 0.0])

    def get_AABB(self):
        return np.stack([self.pos - 0.01, self.pos + 0.01])

    def get_contacts(self, with_entity=None):
        return {"link_a": np.zeros(0)}


class _Franka:
    links = [_Link(0, "hand"), _Link(1, "left_finger"), _Link(2, "right_finger")]

    def __init__(self):
        self.pairs = []

    def get_contacts(self, with_entity=None):
        a, b, ok = zip(*self.pairs) if self.pairs else ((), (), ())
        return {"link_a": np.array(a), "link_b": np.array(b), "valid_mask": np.array(ok, dtype=bool)}

    def control_dofs_position(self, q, dofs):
        pass


class _Env:
    """A die (link 10) and a mug (link 20); ``franka.pairs`` are the raw (link_a, link_b, valid) contacts."""

    motors_dof = list(range(9))

    def __init__(self):
        self.scene = SimpleNamespace(dt=1.0 / 30.0, t=0, step=lambda: setattr(self.scene, "t", self.scene.t + 1))
        self.franka = _Franka()
        self.entity_map = {"die": _Ent([0.5, 0.0, 0.80], [_Link(10, "die")]),
                           "mug": _Ent([0.7, 0.0, 0.80], [_Link(20, "mug")])}
        self.placed_map, self.object_heights, self.asset_map, self.object_quats = {}, {}, {}, {}
        self.table_surface_z, self.articulated_joint_dofs, self.planner = 0.775, {}, object()

    def reset(self, obj_xyz_map, *, settle_steps):
        pass

    def get_joint_positions(self):
        return {}

    def solve_ik(self, *args, **kwargs):
        raise AssertionError("not used by the stubbed plan_segment")


def test_finger_contacts_name_each_finger_once_and_drop_invalid_or_foreign_links():
    env = _Env()
    env.franka.pairs = [(1, 10, True), (10, 1, True), (2, 10, False), (2, 20, True), (0, 10, True), (2, 99, True)]
    fingers = collect_world_state(env)["finger_contacts"]
    assert fingers == {"die": ["left_finger"], "mug": ["right_finger"]}


def _picks(monkeypatch, env, n, lift_per_pick, success, bindings=None):
    """Each pick records 2 frames, closes both fingers on the die and raises it by ``lift_per_pick``."""
    monkeypatch.setattr(runner, "plan_segment", lambda skill, *a, **kw: ([np.zeros(9)] * 2, None, None))

    def record(env, dataset, seg, task_spec, **kw):
        env.scene.t += len(seg)
        env.entity_map["die"].pos = env.entity_map["die"].pos + [0.0, 0.0, lift_per_pick]
        env.franka.pairs = [(1, 10, True), (2, 10, True)]
        return []

    task = TaskSpec(name="t", scene="t", instruction="", success=success,
                    skills=[Skill("pick", "die")] * n, step_bindings=bindings or [GRASPED] * n)
    result = run_scenario_episode(env, task, executor=object(), motion_params=object(), dataset=object(),
                                  record_fn=record)
    return task, result


def test_pick_is_judged_from_its_own_start_at_the_end_of_the_action(monkeypatch):
    env = _Env()
    _, result = _picks(monkeypatch, env, 2, 0.015, SuccessNode.leaf("object_above", {"object": "die", "reference": "table"}))
    first, second = result.handoffs
    assert SKILL_DEFS["pick"].post is bound_leaf and SKILL_DEFS["pick"].settle_s == 0.0
    assert first["done"] is False
    assert second["completion"]["judge_state"]["step_start_positions"]["die"][2] == pytest.approx(0.815)
    assert second["done"] is False
    assert result.frame_count == 4 and env.scene.t == 4 and result.final_window is None


def test_pick_passes_while_the_taller_task_lift_fails(monkeypatch):
    env = _Env()
    task, result = _picks(monkeypatch, env, 1, 0.05,
                          SuccessNode.leaf("object_above", {"object": "die", "reference": "table", "z_margin": 0.10}))
    ok, _ = evaluate_scenario_task(env, task, initial_positions=result.initial_positions,
                                   hold=result.final_qpos, window=result.final_window)
    step = result.handoffs[0]
    assert step["done"] is True and ok is False
    assert evaluate_predicate("grasped", step["completion"]["judge_state"], GRASPED["params"]) is True


def test_unchecked_pick_claims_nothing(monkeypatch):
    unchecked = {"status": "unchecked", "reason": "no step binding"}
    _, result = _picks(monkeypatch, _Env(), 1, 0.05, SuccessNode.leaf("object_above", {"object": "die", "reference": "table"}),
                       bindings=[unchecked])
    assert result.handoffs[0]["done"] is None and result.handoffs[0]["completion"] == unchecked
