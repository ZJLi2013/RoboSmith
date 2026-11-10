"""Plan files: the acceptance families' meta actions load, plans the first version cannot run are rejected rather
than skipped, a file's actions are what its steps run, and a file frozen with the old keys reads the same through
both entries."""

import re
import shutil
from pathlib import Path

import pytest

from robotsmith.scenario.load import (
    OLD_KEYS,
    PlanInvalid,
    check_plan,
    load_plan,
    load_scenario,
)

ACCEPTANCE = Path(__file__).resolve().parents[1] / "scenarios" / "acceptance"
PLANS = ACCEPTANCE / "plans"
FAMILIES = ("f0_open", "f1_wall_gap", "f2_over_barrier", "f4_caged")
OLD = {new: old for old, new in OLD_KEYS.items()}
OLD_NAME, OLD_ACTIONS = OLD["name"], OLD["actions"]


@pytest.mark.parametrize("family", FAMILIES)
def test_each_development_family_has_the_same_composition(family):
    assert load_plan(PLANS / f"{family}.yaml") == load_plan(PLANS / "f0_open.yaml")


@pytest.mark.parametrize("edit, why", [
    (lambda p: [{"push": {"object": "die"}}] + p, "push"),
    (lambda p: [{"move_to": "g.pre"}] + p, "must start with a sample"),
    (lambda p: [{"move_to": "q.pre"} if s == {"move_to": "g.pre"} else s for s in p], "names no candidate"),
    (lambda p: [{"move_to": "g.retreat"} if s == {"move_to": "g.pre"} else s for s in p], "not a pose"),
    (lambda p: p + [{"sample": {"kind": "grasp", "object": "die", "as": "h"}}, {"attach": "h"}],
     "one grasp candidate"),
    (lambda p: [s for s in p if "detach" not in s], "ends holding"),
    (lambda p: [s for s in p if "attach" not in s], "must place what is held"),
])
def test_a_plan_the_first_version_cannot_run_is_rejected(edit, why):
    with pytest.raises(PlanInvalid, match=why):
        check_plan(edit(load_plan(PLANS / "f0_open.yaml")))


def _old_keys(text: str) -> str:
    for old, new in OLD_KEYS.items():
        text = re.sub(rf"^{new}:", f"{old}:", text, flags=re.MULTILINE)
    return text


def _pair(tmp_path, plan_text: str) -> Path:
    shutil.copy(ACCEPTANCE / "f0_open.yaml", tmp_path / "f0_open.yaml")
    (tmp_path / "plans").mkdir(exist_ok=True)
    (tmp_path / "plans" / "f0_open.yaml").write_text(plan_text, encoding="utf-8")
    return tmp_path / "f0_open.yaml"


def test_a_file_with_the_old_keys_reads_the_same_through_both_entries(tmp_path):
    new = (PLANS / "f0_open.yaml").read_text(encoding="utf-8")
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    scenario = _pair(old_dir, _old_keys(new))
    before = (old_dir / "plans" / "f0_open.yaml").read_bytes()
    assert load_plan(old_dir / "plans" / "f0_open.yaml") == load_plan(PLANS / "f0_open.yaml")
    assert load_scenario(scenario, seed=100)[2] == load_scenario(ACCEPTANCE / "f0_open.yaml", seed=100)[2]
    assert (old_dir / "plans" / "f0_open.yaml").read_bytes() == before


def test_the_old_keys_do_not_admit_what_the_new_ones_reject(tmp_path):
    old = _old_keys((PLANS / "f0_open.yaml").read_text(encoding="utf-8"))
    pushed = _pair(tmp_path, old.replace(f"{OLD_ACTIONS}:\n", f"{OLD_ACTIONS}:\n  - push: {{object: die}}\n", 1))
    pushed = pushed.parent / "plans"
    with pytest.raises(PlanInvalid, match="push"):
        load_plan(pushed / "f0_open.yaml")
    unbound = old.replace("move_to: g.pre", "move_to: q.pre", 1)
    (pushed / "f0_open.yaml").write_text(unbound, encoding="utf-8")
    with pytest.raises(PlanInvalid, match="names no candidate"):
        load_plan(pushed / "f0_open.yaml")


def _intents(scenario: Path):
    return load_scenario(scenario, seed=100)[2].intents


def test_a_steps_only_plan_runs_the_same_parts_as_its_actions(tmp_path):
    new = (PLANS / "f0_open.yaml").read_text(encoding="utf-8")
    steps_only = _intents(_pair(tmp_path, new[:new.index("\nactions:")] + "\n"))
    actions = load_plan(PLANS / "f0_open.yaml")
    assert [(i.name, i.target) for i in steps_only] == [("pick", "die"), ("place", "mat.top")]
    assert [i.params["actions"] for i in steps_only] == [actions[:7], actions[7:]]
    assert steps_only == _intents(ACCEPTANCE / "f0_open.yaml")


def test_the_actions_decide_the_lift_when_steps_are_also_given(tmp_path):
    new = (PLANS / "f0_open.yaml").read_text(encoding="utf-8")
    pick = _intents(_pair(tmp_path, new.replace("dist: 0.10", "dist: 0.05", 1)))[0]
    assert pick.params["actions"][-1] == {"move_linear": {"direction": [0, 0, 1], "dist": 0.05}}


def test_a_composition_in_another_order_is_refused_at_load(tmp_path):
    new = (PLANS / "f0_open.yaml").read_text(encoding="utf-8")
    close = "  - gripper: {width: close, target: die}\n"
    attach_first = new.replace(close, "").replace("  - attach: g\n", "  - attach: g\n" + close, 1)
    scenario = _pair(tmp_path, attach_first)
    with pytest.raises(PlanInvalid, match="the first version runs"):
        load_scenario(scenario, seed=100)


@pytest.mark.parametrize("old, new, why", [
    ("  - move_to: p.pre\n", "  - move_to: g.pre\n", "names candidate 'g', not this part's 'p'"),
    ("{width: close, target: die}", "{width: close, target: mat}", "target 'mat' is not this part's object 'die'"),
    ("{width: close, target: die}", "{width: close}", "close names no target"),
    ("{kind: place, object: die,", "{kind: place, object: mat,", "puts down 'mat', but its grasp 'g' holds 'die'"),
])
def test_a_reference_or_object_of_another_part_is_refused_at_load(tmp_path, old, new, why):
    text = (PLANS / "f0_open.yaml").read_text(encoding="utf-8")
    assert old in text
    with pytest.raises(PlanInvalid, match=re.escape(why)):
        load_scenario(_pair(tmp_path, text.replace(old, new, 1)), seed=100)


def test_bindings_renamed_consistently_load_the_same_parts(tmp_path):
    text = (PLANS / "f0_open.yaml").read_text(encoding="utf-8")
    renamed = re.sub(r"\bg\b", "hold", re.sub(r"\bp\b", "drop", text[text.index("\nactions:"):]))
    intents = _intents(_pair(tmp_path, text[:text.index("\nactions:")] + renamed))
    default = _intents(ACCEPTANCE / "f0_open.yaml")
    assert [(i.name, i.target) for i in intents] == [(i.name, i.target) for i in default]
    assert intents[1].params["actions"][0]["sample"] == {**default[1].params["actions"][0]["sample"],
                                                         "grasp": "hold", "as": "drop"}


def _usage_example() -> str:
    doc = (ACCEPTANCE.parents[0] / "AGENT_PROMPT.md").read_text(encoding="utf-8")
    return re.search(r"<!-- example: f0_open -->\n```yaml\n(.*?)```", doc, re.S)[1]


def test_the_usage_example_loads_as_the_public_composition(tmp_path):
    example = _intents(_pair(tmp_path, _usage_example()))
    assert [i.params["actions"] for i in example] == [i.params["actions"] for i in _intents(ACCEPTANCE / "f0_open.yaml")]


def test_a_file_mixing_an_old_key_with_its_new_one_is_rejected(tmp_path):
    new = (PLANS / "f0_open.yaml").read_text(encoding="utf-8")
    scenario = _pair(tmp_path, new.replace("name:", f"{OLD_NAME}: acc_f0_open\nname:", 1))
    with pytest.raises(PlanInvalid, match=f"both '{OLD_NAME}' and 'name'"):
        load_plan(tmp_path / "plans" / "f0_open.yaml")
    with pytest.raises(PlanInvalid, match="both"):
        load_scenario(scenario, seed=100)
