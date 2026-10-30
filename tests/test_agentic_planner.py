"""Agent planning: replies are validated against the scene, place binds to its own target, the baseline never
sees the success tree, failures are routed by evidence, and a re-plan continues in the same episode."""

from types import SimpleNamespace

import numpy as np
import pytest

from robotsmith.scenario_runtime import planner, runner
from robotsmith.scenario_runtime.planner import PlanError, parse_steps, render_request, route_failure
from robotsmith.skills import Skill
from robotsmith.tasks.task_spec import SuccessNode, TaskSpec

BOX = [[-0.15, -0.15, 0.0], [0.15, 0.15, 0.05]]


def _asset(fixture=False, targets=()):
    return SimpleNamespace(is_fixture=fixture, movable_joints=[],
                           metadata=SimpleNamespace(size_cm=[3.0, 3.0, 3.0], place_targets=list(targets)))


ASSETS = {"die": _asset(), "mug": _asset(), "shelf": _asset(True, [{"name": "top", "box": BOX}])}
YAW = {"die": 0.0, "mug": 0.0, "shelf": 0.0}
PLAN = '{"steps": [{"pick": "die"}, {"place": {"target": "shelf.top"}}]}'


@pytest.mark.parametrize("reply", ["no json here", '{"steps": [{"pick": "cup"}]}',
                                   '{"steps": [{"place": {"target": "shelf.top"}}]}',
                                   '{"steps": [{"push": "die"}]}'],
                         ids=["no json", "unknown object", "place with nothing held", "unknown skill"])
def test_invalid_replies_are_plan_errors(reply):
    with pytest.raises(PlanError):
        parse_steps(reply, ASSETS, YAW)


def test_place_binds_to_its_own_target_without_the_success_tree():
    _, intents, bindings = parse_steps(PLAN, ASSETS, YAW)
    assert [i.name for i in intents.intents] == ["pick", "place"]
    assert bindings[0]["predicate"] == "grasped"
    params = bindings[1]["params"]
    assert bindings[1]["predicate"] == "object_in_box"
    assert (params["object"], params["parent"], params["target"], params["mode"]) == ("die", "shelf", "top", "on")


def test_baseline_request_has_no_success_tree():
    objects = [{"id": "die", "asset": "die_01", "kind": "movable", "pos": [0, 0, 0], "size": [0.03] * 3,
                "place_targets": [], "joints": []}]
    assert "Success condition" not in render_request("do it", objects)
    assert "Success condition" in render_request("do it", objects, success={"on": "x"})


PLACE_BINDING = {"status": "bound", "predicate": "object_in_box",
                 "params": {"object": "die", "parent": "shelf", "box": BOX, "yaw_deg": 0.0}}


@pytest.mark.parametrize("entry, positions, expected", [
    ({"skill": "pick"}, {}, "physical"),
    ({"skill": "place", "cause": "grip?"}, {}, "physical"),
    ({"skill": "place"}, {"shelf": [0.6, 0, 0.8], "mug": [0.7, 0, 0.8], "die": [0.3, 0, 0.8]}, "plan"),
    ({"skill": "place"}, {"shelf": [0.6, 0, 0.8], "mug": [1.2, 0, 0.8], "die": [0.3, 0, 0.8]}, "unknown"),
], ids=["pick fails", "dropped in transport", "target occupied", "no evidence"])
def test_failures_are_routed_by_evidence(entry, positions, expected):
    assert route_failure(entry, positions, PLACE_BINDING) == expected


class _Ent:
    def __init__(self, pos):
        self.pos = np.asarray(pos, dtype=float)

    def get_pos(self):
        return self.pos


def _episode(monkeypatch, verdicts, replies, role="baseline", mug=(0.7, 0.0, 0.8)):
    """Each step's verdict comes from ``verdicts`` in order; the LLM answers ``replies`` in order."""
    env = SimpleNamespace(entity_map={"die": _Ent([0.3, 0, 0.8]), "mug": _Ent(mug), "shelf": _Ent([0.6, 0, 0.8])},
                          placed_map={}, articulated_joint_dofs={}, planner=object(), solve_ik=None,
                          reset=lambda *a, **k: None, get_joint_positions=lambda: {})
    monkeypatch.setattr(runner, "plan_segment", lambda *a, **k: ([np.zeros(9)], None, None))
    monkeypatch.setattr(runner, "_live_scene_state",
                        lambda *a, **k: {"positions": {}, "joint_positions": {}})
    monkeypatch.setattr(planner, "to_legacy_skills", lambda intents, task: [Skill(i.name, i.target) for i in intents.intents])
    verdicts = iter(verdicts)

    def judge(env, entry, *a, **k):
        entry.update(done=next(verdicts), completion={"settle_frames": 0})

    monkeypatch.setattr(runner, "_judge_completion", judge)
    prompts, replies = [], iter(replies)

    def llm(prompt):
        prompts.append(prompt)
        return next(replies), 100

    objects = [SimpleNamespace(name=n, asset=n, fixed_position=(0, 0, 0), yaw_deg=0.0) for n in ("die", "mug", "shelf")]
    task = TaskSpec(name="t", scene="t", instruction="put the die on the shelf",
                    success=SuccessNode.leaf("object_above", {"object": "die", "reference": "table"}))
    package = SimpleNamespace(legacy_task=task, scenario_scene=SimpleNamespace(objects=objects), scenario_task=None)
    result, record = planner.run_agentic_episode(
        env, package, ASSETS, role=role, executor=object(), motion_params=object(), dataset=object(),
        record_fn=lambda *a, **k: [], settle_steps=0, target_frames={}, camera_names=(), env_state_names=(), llm=llm)
    return result, record, prompts


def test_plan_level_failure_replans_and_continues_in_the_same_episode(monkeypatch):
    result, record, prompts = _episode(monkeypatch, [True, False, True, True], [PLAN, PLAN])
    assert record.stopped == "plan completed" and len(record.plans) == 2 and record.tokens == 200
    assert record.routing[0]["route"] == "plan"
    assert "Held now: nothing" in prompts[1] and '"pick": "die"' in prompts[1]
    assert [h["skill"] for h in result.handoffs] == ["pick", "place", "pick", "place"]
    assert all("Success condition" not in p for p in prompts)


def test_an_invalid_reply_uses_a_replan(monkeypatch):
    _, record, prompts = _episode(monkeypatch, [True, True], ['{"steps": [{"pick": "cup"}]}', PLAN])
    assert record.stopped == "plan completed" and record.plans[0] == [] and len(prompts) == 2


def test_physical_failure_is_reported_without_replanning(monkeypatch):
    _, record, prompts = _episode(monkeypatch, [False], [PLAN])
    assert record.stopped == "physical failure reported" and len(prompts) == 1


def test_unknown_failure_is_reported_without_replanning(monkeypatch):
    _, record, prompts = _episode(monkeypatch, [True, False], [PLAN], mug=(1.2, 0.0, 0.8))
    assert record.stopped == "unknown failure reported" and len(prompts) == 1


def test_certifier_sees_the_success_tree(monkeypatch):
    _, _, prompts = _episode(monkeypatch, [True, True], [PLAN], role="certifier")
    assert "Success condition" in prompts[0]
