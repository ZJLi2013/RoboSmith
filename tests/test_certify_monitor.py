"""The acceptance contact monitor: every physics step judged by the gate's rule for the span it executes."""

import importlib.util
import json
from pathlib import Path

from robotsmith.execution.gate import Span

_spec = importlib.util.spec_from_file_location(
    "certify", Path(__file__).resolve().parents[1] / "scripts" / "acceptance" / "certify.py")
certify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(certify)

FINGER_DIE = (("die", "base"), ("robot", "left_finger"))
HAND_TABLE = (("env", "?"), ("robot", "hand"))


class _Env:
    on_step = None
    _writes = 0

    def __init__(self, contacts):
        self.contacts = contacts

    def contacts_now(self):
        return self.contacts.pop(0) if self.contacts else []

    def step(self):
        self._writes += 1
        self.on_step(self)


PICK = ([Span("approach", 0, 2), Span("close", 2, 3, allow="grip"), Span("lift", 3, 4, holding=True)],
        "die", {"env"}, "pick")


def test_each_step_is_judged_by_the_span_it_executes():
    env = _Env([[], [FINGER_DIE], [FINGER_DIE], [FINGER_DIE]])
    mon = certify.Monitor(env)
    mon.enter("replay", PICK)
    for _ in range(4):
        env.step()
    assert mon.steps["replay"] == 4 and mon.violating["replay"] == 1
    assert mon.first == {"stage": "replay", "skill": "pick", "index": 1, "span": "approach",
                         "pair": [["die", "base"], ["robot", "left_finger"]], "physics_step": 2}


def test_reset_steps_are_counted_not_judged_and_unmapped_steps_are_violations():
    env = _Env([[HAND_TABLE], [HAND_TABLE]])
    mon = certify.Monitor(env)
    env.step()
    assert mon.steps["init"] == 1 and not mon.violating
    mon.enter("between")
    env.step()
    assert mon.violating["between"] == 1 and mon.first["span"] == "unmapped"


def test_a_settle_keeps_the_last_span_rule():
    spans, _, _, _ = certify._settle_ctx(PICK)
    assert spans[0].holding and spans[0].allow is None
    env = _Env([[(("die", "base"), ("env", "?"))]])
    mon = certify.Monitor(env)
    mon.enter("settle", certify._settle_ctx(PICK))
    env.step()
    assert mon.violating["settle"] == 1 and mon.first["span"] == "settle"


def test_a_free_die_pushed_into_a_wall_is_a_violation():
    env = _Env([[(("die", "base"), ("wall", "base"))]])
    mon = certify.Monitor(env)
    mon.enter("replay", PICK)
    env.step()
    assert mon.first["pair"] == [["die", "base"], ["wall", "base"]]


def test_a_trial_restore_does_not_undo_its_steps():
    env = _Env([[FINGER_DIE], [], []])
    mon = certify.Monitor(env)
    mon.enter("trial", PICK)
    for _ in range(3):
        env.step()
    env._writes = 0
    mon.enter("replay", PICK)
    env.step()
    assert mon.steps == {"trial": 3, "replay": 1} and mon.violating["trial"] == 1


def test_a_crashed_instance_is_retried_once_and_both_rows_are_kept(tmp_path):
    calls = []

    def one(seed):
        calls.append(seed)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return {"outcome": "certified", "certified": True}

    with (tmp_path / "rows.jsonl").open("w", encoding="utf-8") as fh:
        rows = certify.run_batch([7], one, fh)
    written = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["attempt"], r["outcome"]) for r in written] == [(1, "error"), (2, "certified")]
    assert "boom" in written[0]["why"] and rows == [written[1]]
