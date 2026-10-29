"""Teacher steps bind to the completion condition they can be judged by; contradictions fail at load."""

from pathlib import Path

import pytest

from robotsmith.scenario.adapters import to_step_bindings
from robotsmith.scenario.intents import SkillIntentSequenceSpec, SkillIntentSpec
from robotsmith.scenario.relations import SolveError
from robotsmith.scenario.specs import AllOf, AnyOf, TaskSuccessCfg
from robotsmith.scenario_runtime import materialize_candidate
from robotsmith.tasks.task_spec import TaskSpec

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"
UNMIGRATED = {"status": "unchecked", "reason": "not migrated"}


def _steps(*pairs):
    return SkillIntentSequenceSpec(intents=tuple(SkillIntentSpec(name=n, target=t) for n, t in pairs))


def _on(obj, parent, target, mode="on"):
    return TaskSuccessCfg(predicate="object_in_box",
                          params={"object": obj, "parent": parent, "target": target, "mode": mode})


PICK_PLACE = _steps(("pick", "die"), ("place", "supporter.shelf_lower"))


@pytest.mark.parametrize("name", sorted(p.stem for p in SCENARIOS.glob("*.yaml")))
def test_every_scenario_binds_one_entry_per_step(name):
    task = materialize_candidate(SCENARIOS / f"{name}.yaml", seed=0).legacy_task
    assert len(task.step_bindings) == len(task.skills)
    assert TaskSpec.from_dict(task.to_dict()).step_bindings == task.step_bindings


def test_drawer_place_binds_inside_and_leaves_the_close_out():
    task = materialize_candidate(SCENARIOS / "pick_place_into_drawer.yaml", seed=0).legacy_task
    place = next(b for b, s in zip(task.step_bindings, task.skills) if s.name == "place")
    assert place["status"] == "bound"
    assert {k: place["params"][k] for k in ("object", "parent", "target", "mode")} == {
        "object": "die", "parent": "drawer", "target": "drawer_opening", "mode": "inside"}
    assert [b for b, s in zip(task.step_bindings, task.skills) if s.name != "place"] == [UNMIGRATED] * 3


@pytest.mark.parametrize("success, expected", [
    (_on("die", "supporter", "shelf_lower"), "bound"),
    (AnyOf(terms=(_on("die", "supporter", "shelf_lower"), _on("die", "supporter", "shelf_upper"))),
     "success tree is not a conjunction"),
    (AllOf(terms=(_on("die", "supporter", "shelf_lower"), _on("die", "supporter", "shelf_lower"))),
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
    assert [b.get("reason") for b in bindings] == ["not migrated", "object placed more than once"] * 2
