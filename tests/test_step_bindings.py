"""Plan steps bind to the completion condition they can be judged by; contradictions fail at load."""

from pathlib import Path

import pytest

from robotsmith.scenario.adapters import to_step_bindings
from robotsmith.scenario.relations import SolveError
from robotsmith.scenario.load import load_scenario
from robotsmith.skills import Skill
from robotsmith.tasks.task_spec import SuccessNode, TaskSpec

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"


def _steps(*pairs):
    return [Skill(n, t) for n, t in pairs]


def _on(obj, parent, target, mode="on"):
    return SuccessNode.leaf("object_in_box", {"object": obj, "parent": parent, "target": target, "mode": mode})


PICK_PLACE = _steps(("pick", "die"), ("place", "supporter.shelf_lower"))


@pytest.mark.parametrize("name", sorted(p.stem for p in SCENARIOS.glob("*.yaml")))
def test_every_scenario_binds_one_entry_per_step(name):
    task = load_scenario(SCENARIOS / f"{name}.yaml", seed=0).task
    assert len(task.step_bindings) == len(task.skills)
    assert TaskSpec.from_dict(task.to_dict()).step_bindings == task.step_bindings


def test_drawer_binds_place_inside_and_the_drawer_joint_for_open_and_close():
    task = load_scenario(SCENARIOS / "pick_place_into_drawer.yaml", seed=0).task
    by_skill = {s.name: b for b, s in zip(task.step_bindings, task.skills)}
    assert {k: by_skill["place"]["params"][k] for k in ("object", "parent", "target", "mode")} == {
        "object": "die", "parent": "drawer", "target": "drawer_opening", "mode": "inside"}
    assert by_skill["pick"] == {"status": "bound", "predicate": "grasped", "params": {"object": "die", "lift": 0.02}}
    opened, closed = by_skill["open"], by_skill["close"]
    assert (opened["predicate"], closed["predicate"]) == ("joint_opened", "joint_closed")
    assert opened["params"]["object"] == closed["params"]["object"] == "drawer"
    assert opened["params"]["joint"] == closed["params"]["joint"]
    assert opened["params"]["threshold"] > closed["params"]["threshold"]


@pytest.mark.parametrize("success, expected", [
    (_on("die", "supporter", "shelf_lower"), "bound"),
    (SuccessNode(op="any", terms=[_on("die", "supporter", "shelf_lower"), _on("die", "supporter", "shelf_upper")]),
     "success tree is not a conjunction"),
    (SuccessNode(op="all", terms=[_on("die", "supporter", "shelf_lower"), _on("die", "supporter", "shelf_lower")]),
     "2 success relations for die"),
    (_on("apple", "supporter", "shelf_lower"), "0 success relations for die"),
], ids=["single on", "any_of", "two candidates", "other object"])
def test_place_binding(success, expected):
    place = to_step_bindings(PICK_PLACE, success)[1]
    assert (place["status"] if place["status"] == "bound" else place["reason"]) == expected


def test_wrong_shelf_fails_at_load():
    with pytest.raises(SolveError, match="shelf_upper"):
        to_step_bindings(_steps(("pick", "die"), ("place", "supporter.shelf_upper")),
                         _on("die", "supporter", "shelf_lower"))


def test_object_placed_twice_stays_unchecked():
    steps = _steps(("pick", "die"), ("place", "supporter.shelf_upper"), ("pick", "die"),
                   ("place", "supporter.shelf_lower"))
    bindings = to_step_bindings(steps, _on("die", "supporter", "shelf_lower"))
    assert [b.get("reason") for b in bindings] == [None, "object placed more than once"] * 2


@pytest.mark.parametrize("first", [False, True], ids=["plan then agent", "agent then plan"])
def test_each_load_owns_its_task(first):
    path = SCENARIOS / "pick_place_block_on_mat.yaml"
    a = load_scenario(path, seed=0, agent_plan=first).task
    a_dict = a.to_dict()
    b = load_scenario(path, seed=0, agent_plan=not first).task
    assert a.to_dict() == a_dict
    assert a is not b and a.success is not b.success
    plan, agent = (b, a) if first else (a, b)
    assert [s.name for s in plan.skills] == ["pick", "place"] and len(plan.step_bindings) == 2
    assert agent.skills == [] and agent.step_bindings == []
