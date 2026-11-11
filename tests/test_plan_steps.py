"""Plan files and step lists: the acceptance families run the same steps, a pick may carry its lift, and steps that
cannot run from the hand they start with are refused at load, before any physics step."""

import re
import shutil
from pathlib import Path

import jsonschema
import pytest

from robotsmith.scenario.load import PlanInvalid, load_scenario, steps_to_intents

ACCEPTANCE = Path(__file__).resolve().parents[1] / "scenarios" / "acceptance"
PLANS = ACCEPTANCE / "plans"
FAMILIES = ("f0_open", "f1_wall_gap", "f2_over_barrier", "f4_caged")
ASSETS = {"die": None, "mat": None, "drawer": None}


def _steps(*steps):
    return [(i.name, i.target) for i in steps_to_intents(list(steps), ASSETS)[0].intents]


def _pair(tmp_path, plan_text: str) -> Path:
    shutil.copy(ACCEPTANCE / "f0_open.yaml", tmp_path / "f0_open.yaml")
    (tmp_path / "plans").mkdir(exist_ok=True)
    (tmp_path / "plans" / "f0_open.yaml").write_text(plan_text, encoding="utf-8")
    return tmp_path / "f0_open.yaml"


def _intents(scenario: Path):
    return load_scenario(scenario, seed=100)[2].intents


@pytest.mark.parametrize("family", FAMILIES)
def test_each_development_family_runs_the_same_steps(family):
    other = (PLANS / f"{family}.yaml").read_text(encoding="utf-8").splitlines()[1:]
    assert other == (PLANS / "f0_open.yaml").read_text(encoding="utf-8").splitlines()[1:]


def test_the_public_plan_picks_the_die_and_places_it_on_the_mat_with_the_default_lift():
    intents = _intents(ACCEPTANCE / "f0_open.yaml")
    assert [(i.name, i.target, i.params) for i in intents] == [("pick", "die", {}), ("place", "mat.top", {})]


def test_a_pick_carries_its_lift(tmp_path):
    text = (PLANS / "f0_open.yaml").read_text(encoding="utf-8")
    lifted = text.replace("  - pick: die\n", "  - pick: die\n    lift: {direction: [1, 0, 1], dist: 0.05}\n")
    assert _intents(_pair(tmp_path, lifted))[0].params == {"lift": {"direction": [1, 0, 1], "dist": 0.05}}


@pytest.mark.parametrize("lift", ["{direction: [0, 0, 1], dist: 0}", "{direction: [0, 1], dist: 0.1}",
                                  "{dist: 0.1}"])
def test_a_lift_that_is_not_a_world_line_is_refused_at_load(tmp_path, lift):
    text = (PLANS / "f0_open.yaml").read_text(encoding="utf-8").replace("  - pick: die\n",
                                                                          f"  - pick: die\n    lift: {lift}\n")
    with pytest.raises(jsonschema.ValidationError):
        load_scenario(_pair(tmp_path, text), seed=100)


def test_a_plan_file_with_meta_actions_is_refused(tmp_path):
    text = (PLANS / "f0_open.yaml").read_text(encoding="utf-8") + "actions:\n  - attach: g\n"
    with pytest.raises(jsonschema.ValidationError, match="actions"):
        load_scenario(_pair(tmp_path, text), seed=100)


def test_steps_may_end_holding_and_may_start_holding():
    assert _steps({"pick": "die"}) == [("pick", "die")]
    intents, _ = steps_to_intents([{"place": {"target": "mat"}}], ASSETS, held="die")
    assert [(i.name, i.target) for i in intents.intents] == [("place", "mat")]
    assert _steps({"open": "drawer"}, {"pick": "die"}, {"place": {"target": "mat"}}, {"close": "drawer"}) == [
        ("open", "drawer"), ("pick", "die"), ("place", "mat"), ("close", "drawer")]


@pytest.mark.parametrize("steps, held, why", [
    ([{"place": {"target": "mat"}}], None, "step 0: place mat with nothing held"),
    ([{"pick": "die"}, {"place": {"target": "mat"}}, {"place": {"target": "mat"}}], None,
     "step 2: place mat with nothing held"),
    ([{"pick": "die"}, {"pick": "mat"}], None, "step 1: pick mat while holding die"),
    ([{"pick": "mat"}], "die", "step 0: pick mat while holding die"),
])
def test_steps_that_cannot_run_from_the_hand_they_start_with_are_refused(steps, held, why):
    with pytest.raises(PlanInvalid, match=re.escape(why)):
        steps_to_intents(steps, ASSETS, held=held)


@pytest.mark.parametrize("steps", [[{"pick": "cup"}], [{"pick": "die"}, {"place": {"target": "tray"}}],
                                   [{"open": "cabinet"}], [{"close": "cabinet"}]])
def test_an_unknown_object_is_refused(steps):
    with pytest.raises(PlanInvalid, match="unknown object"):
        steps_to_intents(steps, ASSETS)


def test_an_unknown_place_target_is_refused_at_load(tmp_path):
    text = (PLANS / "f0_open.yaml").read_text(encoding="utf-8").replace("mat.top", "mat.side")
    with pytest.raises(PlanInvalid, match="side"):
        load_scenario(_pair(tmp_path, text), seed=100)


def test_the_usage_example_loads_as_the_public_plan(tmp_path):
    doc = (ACCEPTANCE.parents[0] / "AGENT_PROMPT.md").read_text(encoding="utf-8")
    example = re.search(r"<!-- example: f0_open -->\n```yaml\n(.*?)```", doc, re.S)[1]
    assert _intents(_pair(tmp_path, example))[1:] == _intents(ACCEPTANCE / "f0_open.yaml")[1:]
    assert _intents(_pair(tmp_path, example))[0].params == {"lift": {"direction": [0, 0, 1], "dist": 0.10}}
