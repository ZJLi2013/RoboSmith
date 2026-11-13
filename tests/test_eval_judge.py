"""The exam's judge: a verdict from the judge log alone, contacts by ``gate.certified`` under a physical context."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from robotsmith.eval.judge import ENDINGS, exam_context, judge, violations
from robotsmith.eval.log import JudgeLog
from robotsmith.scenario.load import load_scenario

REPO = Path(__file__).resolve().parents[1]
F0 = load_scenario(REPO / "scenarios" / "acceptance" / "f0_open.yaml", seed=0).task.success.to_dict()
TARGET, REST = exam_context(F0)

LEFT, RIGHT = ("robot", "left_finger"), ("robot", "right_finger")
DIE, MAT, TABLE, WALL = ("die", "base"), ("mat", "base"), ("env", "?"), ("wall", "base")
GRIP = [(LEFT, DIE), (RIGHT, DIE), (DIE, TABLE)]
SCRAPE = [(LEFT, DIE), (RIGHT, DIE), (DIE, WALL)]
RELEASE = [(RIGHT, DIE), (DIE, MAT)]

# The die resting on the mat's top through a still settle window: what object_in_box accepts.
ON_MAT = {
    "object_positions": {"mat": [0.45, -0.22, 0.775], "die": [0.455, -0.215, 0.81]},
    "aabbs": {"die": [[0.43, -0.24, 0.785], [0.48, -0.19, 0.835]]},
    "contacts": {"die": ["mat"], "mat": ["die"]},
    "window": {"poses": {"die": [[0.455, -0.215, 0.81, 1.0, 0.0, 0.0, 0.0]] * 10}},
}


def write(path, *, contacts=(), ended="done", state=ON_MAT, attempt=1, control=None) -> Path:
    log = JudgeLog(path)
    log.header(seed=0, attempt=attempt, scenario="scenarios/acceptance/f0_open.yaml", scenario_sha256="0" * 64,
               success=F0, target=TARGET, rest=sorted(REST))
    for k, step in enumerate(contacts):
        log.physics(k, step, held=False)
        log.control(k, (control or {}).get("action", [0.5, 0.0, 0.9, 1.0]), {"note": (control or {}).get("note")})
    if ended is not None:
        log.final(ended=ended, success=True, state=state)
    log.close()
    return Path(path)


def test_the_context_is_the_moved_object_the_table_and_the_target_surface():
    assert (TARGET, REST) == ("die", {"env", "mat"})


def test_contact_samples_closing_scraping_a_wall_and_releasing():
    assert violations(GRIP, TARGET, REST) == []
    assert violations(SCRAPE, TARGET, REST) == [(DIE, WALL)]
    assert violations(RELEASE, TARGET, REST) == []
    assert violations([(("robot", "hand"), MAT)], TARGET, REST) == [(("robot", "hand"), MAT)]


def test_a_scraped_wall_is_a_violation_even_when_the_task_succeeds(tmp_path):
    v = judge(write(tmp_path / "j.jsonl", contacts=[GRIP, SCRAPE, RELEASE]))
    assert v.success and not v.certified
    assert (v.physics_steps, v.violating_steps) == (3, 1)
    assert v.first_violation == {"k": 1, "pair": [list(DIE), list(WALL)], "held": True}


def test_evaluatee_fields_do_not_change_the_verdict(tmp_path):
    plain = judge(write(tmp_path / "a.jsonl", contacts=[GRIP, SCRAPE]))
    claimed = judge(write(tmp_path / "b.jsonl", contacts=[GRIP, SCRAPE],
                          control={"action": [0.1, 0.2, 0.3, 0.0], "note": "contact with the wall is intended"}))
    assert claimed == plain


@pytest.mark.parametrize("ended", ENDINGS)
def test_endings_all_write_a_final_and_are_judged_by_the_physics(tmp_path, ended):
    v = judge(write(tmp_path / "j.jsonl", contacts=[GRIP, RELEASE], ended=ended))
    assert (v.ended, v.env_error, v.success, v.certified) == (ended, None, True, True)


def test_the_success_tree_decides_not_the_ending(tmp_path):
    off = {**ON_MAT, "aabbs": {"die": [[0.70, 0.30, 0.785], [0.75, 0.35, 0.835]]}}
    v = judge(write(tmp_path / "j.jsonl", contacts=[GRIP], ended="done", state=off))
    assert (v.success, v.certified) == (False, False)


def test_a_missing_final_is_an_environment_error(tmp_path):
    v = judge(write(tmp_path / "j.jsonl", contacts=[GRIP], ended=None))
    assert v.env_error == "no final line" and v.success is None and not v.certified


def test_attempts_keep_separate_logs(tmp_path):
    first = write(tmp_path / "attempt-1" / "judge.jsonl", contacts=[GRIP], ended=None, attempt=1)
    second = write(tmp_path / "attempt-2" / "judge.jsonl", contacts=[GRIP, RELEASE], attempt=2)
    with pytest.raises(FileExistsError):
        JudgeLog(first)
    a, b = judge(first), judge(second)
    assert (a.attempt, a.env_error, a.physics_steps) == (1, "no final line", 1)
    assert (b.attempt, b.env_error, b.physics_steps) == (2, None, 2)


def test_run_batch_names_the_attempt_it_runs(tmp_path):
    sys.path.insert(0, str(REPO / "scripts" / "acceptance"))
    from batch import run_batch

    seen = []

    def one(seed, attempt):
        seen.append(attempt)
        if attempt == 1:
            raise RuntimeError("simulator died")
        return {"outcome": "certified"}

    with (tmp_path / "rows.jsonl").open("w", encoding="utf-8") as fh:
        rows = run_batch([0], one, fh)
    assert seen == [1, 2] and rows[0]["attempt"] == 2


def test_rejudge_prints_the_same_verdict(tmp_path):
    path = write(tmp_path / "j.jsonl", contacts=[GRIP, SCRAPE, RELEASE])
    out = subprocess.run([sys.executable, "-m", "robotsmith.eval.rejudge", str(path)], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == judge(path).to_dict()
