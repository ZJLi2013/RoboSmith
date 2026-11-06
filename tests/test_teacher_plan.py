"""Teacher meta action plans: the acceptance families' plans load, and plans the first version cannot run are
rejected rather than skipped."""

from pathlib import Path

import pytest

from robotsmith.scenario.load import PlanInvalid, check_plan, load_plan

TEACHERS = Path(__file__).resolve().parents[1] / "scenarios" / "acceptance" / "teacher"
FAMILIES = ("f0_open", "f1_wall_gap", "f2_over_barrier", "f4_caged")


@pytest.mark.parametrize("family", FAMILIES)
def test_each_development_family_has_the_same_composition(family):
    assert load_plan(TEACHERS / f"{family}.yaml") == load_plan(TEACHERS / "f0_open.yaml")


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
        check_plan(edit(load_plan(TEACHERS / "f0_open.yaml")))
