"""Plan files: the acceptance families' meta actions load, plans the first version cannot run are rejected rather
than skipped, and a file frozen with the old keys reads the same through both entries."""

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
    (lambda p: [{"move_to": "g.pre"}] + p, "names no candidate"),
    (lambda p: [{"move_to": "g.retreat"} if s == {"move_to": "g.pre"} else s for s in p], "not a pose"),
    (lambda p: p + [{"attach": "g"}], "one grasp candidate"),
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


def test_a_file_mixing_an_old_key_with_its_new_one_is_rejected(tmp_path):
    new = (PLANS / "f0_open.yaml").read_text(encoding="utf-8")
    scenario = _pair(tmp_path, new.replace("name:", f"{OLD_NAME}: acc_f0_open\nname:", 1))
    with pytest.raises(PlanInvalid, match=f"both '{OLD_NAME}' and 'name'"):
        load_plan(tmp_path / "plans" / "f0_open.yaml")
    with pytest.raises(PlanInvalid, match="both"):
        load_scenario(scenario, seed=100)
