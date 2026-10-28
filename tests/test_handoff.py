import numpy as np
import pytest

from robotsmith.scenario_runtime import handoff
from robotsmith.skills import Skill


def _place(n=20, release=15, loss=None, jump=0.0, end_xy=(0.5, 0.0)):
    """Hand moves along x; the object rides 0.1 m below until ``loss`` (or release)."""
    seg = [np.r_[np.zeros(7), 0.01, 0.01] if k < release else np.r_[np.zeros(7), 0.04, 0.04]
           for k in range(n)]
    samples = []
    for k in range(n):
        hand = np.array([0.3 + 0.01 * k, 0.0, 1.0])
        lost = (loss is not None and k >= loss) or k >= release
        obj = np.array([*end_xy, 0.8]) if lost else hand - [0, 0, 0.1]
        q = np.zeros(7)
        q[0] = jump * k if loss is not None and loss - 5 <= k <= loss else 0.0
        samples.append({"hand": hand, "obj": obj, "q": q})
    return seg, samples


@pytest.mark.parametrize("kw, done, cause", [
    ({}, True, None),
    ({"loss": 8, "jump": 0.07}, False, "motion?"),
    ({"loss": 8, "jump": 0.0}, False, "grip?"),
    ({"end_xy": (0.9, 0.3)}, False, "release?"),
])
def test_place_handoff(kw, done, cause):
    seg, samples = _place(**kw)
    h = handoff.check(Skill("place", "goal"), "all", seg, samples, held="die", target=[0.5, 0.0, 0.8])
    assert h["done"] is done
    assert h.get("cause") == cause
    assert h["release_frame"] == 15


def test_pick_and_open_handoff_and_first_failure():
    hand = np.array([0.5, 0.0, 1.0])
    samples = [{"hand": hand, "obj": hand - [0, 0, 0.1], "q": np.zeros(7)}]
    pick = handoff.check(Skill("pick", "die"), "all", [np.zeros(9)], samples, held="die",
                         held_start=[0.5, 0.0, 0.8])
    assert pick["done"] is True and pick["lift_m"] == pytest.approx(0.1)

    opened = handoff.check(Skill("open", "drawer"), "all", [], [], held=None,
                           joints_before={"drawer": {"slide": 0.0}},
                           joints_after={"drawer": {"slide": -0.05}},
                           trace={"move_vec": [-0.245, 0.0, 0.0]})
    assert opened["done"] is False
    assert handoff.first_failure([pick, opened]) is opened
