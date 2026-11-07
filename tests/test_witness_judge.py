"""The solvability witness judges the task's settle like every other step, and the held-out families stay out of
the development entry points."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE = ROOT / "scenarios" / "acceptance"
_spec = importlib.util.spec_from_file_location("witness", ROOT / "scripts" / "acceptance" / "witness.py")
witness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(witness)

LINKS = {"robot": [(0, "link0", -1), (1, "hand", 0), (2, "left_finger", 1), (3, "right_finger", 1)],
         "die": [(10, "base", -1)], "wall": [(20, "base", -1)]}


class _Ent:
    def __init__(self, env, name):
        self.env, self.links = env, [SimpleNamespace(idx=i, name=n, parent_idx=p) for i, n, p in LINKS[name]]

    def get_contacts(self):
        own = {lk.idx for lk in self.links}
        mine = [p for p in self.env.pairs if own & set(p)]
        a, b = zip(*mine) if mine else ((), ())
        return {"link_a": np.array(a, dtype=int), "link_b": np.array(b, dtype=int)}


class _Env:
    on_step = None

    def __init__(self, settle_pairs):
        self.pairs, self.settle_pairs = [], settle_pairs
        self.franka = _Ent(self, "robot")
        self.entity_map = {"die": _Ent(self, "die"), "wall": _Ent(self, "wall")}

    def step(self):
        self.pairs = self.settle_pairs
        self.on_step(self)


def _settle(env, task, *, initial_positions, hold):
    for _ in range(5):
        env.step()
    return True, {}


def test_a_contact_only_in_the_settle_fails_the_witness(monkeypatch):
    monkeypatch.setattr(witness, "evaluate_scenario_task", _settle)
    env = _Env([(10, 20)])
    judge = witness.Judge(witness.Contacts(env), ["retreat", "settle"])
    assert witness.judge_task(env, None, {}, None, judge)
    assert judge.first == "settle: die-wall" and judge.steps["settle"] == 5
    assert env.on_step is None


def test_joined_links_may_touch_and_others_may_not():
    env = _Env([])
    contacts = witness.Contacts(env)
    env.pairs = [(2, 3), (1, 2)]
    assert contacts.illegal("carry") == []
    env.pairs = [(0, 2)]
    assert contacts.illegal("carry") == ["robot.link0-robot.left_finger"]


def test_no_development_entry_names_or_resolves_a_heldout_family():
    heldout = [p.stem for p in (ACCEPTANCE / "heldout").glob("*.yaml")]
    assert heldout
    entries = [ROOT / "scripts" / "acceptance" / f for f in ("witness.py", "certify.py")]
    for name in heldout:
        assert not (ACCEPTANCE / f"{name}.yaml").exists()
        assert all(name not in p.read_text(encoding="utf-8") for p in entries)
