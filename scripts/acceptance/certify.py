"""Certified readings of the manipulation chain on the acceptance families.

  python scripts/acceptance/certify.py --family f0_open --split test --out out/f0_open_test

Each instance runs the family's teacher steps once through ``EpisodeRun`` (skills, execution gate, executor)
on the seed's layout. Every physics step after the reset is read for the contacts it resolved and judged by
``gate.certified`` for the span being executed: trial and replay steps by the verified motion's spans, step
and task settles by the last replayed span. A positive instance is certified when the task predicate holds
and no step of any stage broke the rule; a negative one when its first step is infeasible and nothing ran
after the reset. Writes ``instances.jsonl`` (one row per instance, as it runs), the inputs and ``summary.json``.
"""

import argparse
import dataclasses
import json
import os
import shutil
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robotsmith.execution.gate import SUPPORT, Span, certified  # noqa: E402

SETTLE_STEPS = 30
NEGATIVE = {"f4_caged"}
INSTANCE_TIMEOUT_S = 600
HUNG = 3


class Monitor:
    """Counts the physics steps of each stage and checks each one's contacts against its span's rule.

    The only reader of ``SimEnv.on_step``. Callers name the stage and its ``(spans, target, rest, skill)`` with
    ``enter``; a step outside any named stage is a violation, so an instance it touches is not certified.
    Steps of the ``init`` stage (the reset and its settle) are counted, not judged. Counts live here, so a
    trial's snapshot restore does not undo them; ``start`` clears them for the next instance."""

    def __init__(self, env):
        self.env = env
        self.stage, self.ctx, self.k = "init", None, 0
        self.start()
        env.on_step = self._on_step

    def start(self):
        self.steps, self.violating = Counter(), Counter()
        self.first = None

    def enter(self, stage, ctx=None):
        self.stage, self.ctx, self.k = stage, ctx, 0

    def _on_step(self, env):
        k, self.k = self.k, self.k + 1
        self.steps[self.stage] += 1
        if self.stage == "init":
            return
        if self.ctx is None:
            self._violate(k, "unmapped", None)
            return
        spans, target, rest, skill = self.ctx
        span = next((s for s in spans if s.start <= k < s.end), spans[-1])
        bad = [p for p in env.contacts_now() if not certified(p, span, target, rest)]
        if bad:
            self._violate(k, span.label, bad[0], skill)

    def _violate(self, k, label, pair, skill=None):
        self.violating[self.stage] += 1
        if self.first is None:
            self.first = {"stage": self.stage, "skill": skill, "index": k, "span": label,
                          "pair": [list(x) for x in pair] if pair else None, "physics_step": self.env._writes}


def _settle_ctx(ctx):
    spans, target, rest, skill = ctx
    last = spans[-1]
    return [Span("settle", 0, 1, holding=last.holding, allow=last.allow)], target, rest, skill


def _tolist(x):
    return np.asarray(x, dtype=float).round(5).tolist()


def run_instance(env, pkg, family, mon, record_episode):
    from robotsmith.execution import MotionExecutor
    from robotsmith.motion import MotionParams
    from robotsmith.scenario.adapters import resolved_target_positions, target_frames
    from robotsmith.scenario_runtime.runner import (EpisodeRun, evaluate_scenario_task, reset_scene,
                                                    resolve_static_target_positions)
    from robotsmith.sim.sim_env import SceneInvalid
    from witness import caged, difficulty

    task = pkg.legacy_task
    placed = {o.name_override: np.asarray(o.fixed_position, dtype=float) for o in pkg.legacy_scene.objects}
    die = env.placed_map["die"]
    env.placed_map["die"] = dataclasses.replace(die, position=list(placed["die"]))
    row = {"family": family, "layout": {k: _tolist(v) for k, v in placed.items()}}
    mon.enter("init")
    try:
        reset_scene(env, SETTLE_STEPS)
    except SceneInvalid as exc:
        return {**row, "settle_ok": False, "outcome": "init_failed", "why": str(exc), "steps": dict(mon.steps)}
    settled = {n: _tolist(p) for n, p in env.sync_object_states().items()}
    row.update(settle_ok=True, settled=settled, **difficulty(family, placed, np.asarray(settled["die"])))
    if family in NEGATIVE:
        why = caged(placed, np.asarray(settled["die"]))
        row.update(caged=not why, caged_why=why)

    meta, last = {}, [None]
    run = EpisodeRun(env, task, executor=MotionExecutor(), motion_params=MotionParams(), dataset=_NoDataset(),
                     record_fn=None, target_frames=target_frames(pkg.scenario_scene), camera_names=(),
                     env_state_names=())
    verify = run.gate.verify

    def verified(trajectory, spans, *, target, q_start, rest=()):
        m = verify(trajectory, spans, target=target, q_start=q_start, rest=rest)
        meta[id(m.trajectory)] = (m.spans, target, {SUPPORT, *rest}, run.skills[-1].name)
        return m

    def record(env_, dataset, seg, task_spec, **kw):
        ctx = meta.get(id(seg))
        if ctx is not None:
            last[0] = ctx
            mon.enter("replay", ctx)
        else:
            mon.enter("settle", _settle_ctx(last[0]) if last[0] else None)
        try:
            return record_episode(env_, dataset, seg, task_spec, **kw)
        finally:
            mon.enter("between")

    trial = env.trial_from_here

    def trialled(trajectory, target, **kw):
        mon.enter("trial", meta.get(id(trajectory)))
        try:
            return trial(trajectory, target, **kw)
        finally:
            mon.enter("between")

    run.gate.verify, run.record_fn, env.trial_from_here = verified, record, trialled
    try:
        mon.enter("between")
        run.run_steps(list(task.skills), list(task.step_bindings))
        if run.infeasible is not None:
            success, judged = False, None
        else:
            mon.enter("judge", _settle_ctx(last[0]) if last[0] else None)
            targets = {n: np.asarray(x, dtype=np.float32)
                       for n, x in resolved_target_positions(pkg.scenario_scene).items()}
            targets.update(resolve_static_target_positions(env, target_frames(pkg.scenario_scene)))
            success, judged = evaluate_scenario_task(env, task, initial_positions=run.initial_positions,
                                                     reference_positions=targets, hold=run.qpos, window=run.window)
    finally:
        del env.trial_from_here
        mon.enter("init")

    steps = {k: v for k, v in mon.steps.items() if k != "init"}
    row.update(
        steps=steps, violating_steps=dict(mon.violating), first_violation=mon.first,
        success=bool(success), frame_count=run.frame_count, infeasible=run.infeasible,
        handoffs=[{k: h.get(k) for k in ("skill", "phase", "done", "cause")} for h in run.handoffs],
        trials=[t.get("trials") for t in run.skill_traces if "trials" in t],
        carry_tries=[t.get("carry_tries") for t in run.skill_traces if "carry_tries" in t],
        task_judge={"object_positions": judged.get("object_positions")} if judged else None,
    )
    clean = not mon.violating
    if family in NEGATIVE:
        executed = sum(steps.values())
        first = run.handoffs[0] if run.handoffs else {}
        row["certified"] = bool(first.get("cause") == "infeasible" and executed == 0)
        row["outcome"] = "negative_ok" if row["certified"] else ("executed" if executed else "not_infeasible")
    else:
        row["certified"] = bool(success and clean)
        row["outcome"] = ("certified" if row["certified"] else "infeasible" if run.infeasible
                          else "violation" if not clean else "task_failed")
    return row


class _NoDataset:
    def add_frame(self, frame):
        pass


def run_batch(seeds, one, fh, prior=(), timeout=None, on_hang=lambda: os._exit(HUNG)) -> list[dict]:
    """``one(seed)`` for each seed, its row appended to ``fh`` as it finishes. An instance that raises is retried
    once; both attempts stay in ``fh`` and the last one is the instance's reading. ``prior`` are the rows of an
    earlier run of the batch: a seed they settle is not run again. An attempt still running after ``timeout``
    seconds is written as an error and ``on_hang`` ends the process, since a hung simulator step cannot be
    interrupted from Python; rerunning with the rows so far retries it."""
    lock = threading.Lock()

    def write(row):
        with lock:
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
        print(json.dumps({k: row.get(k) for k in ("seed", "attempt", "outcome", "first_violation", "seconds")},
                         default=str), flush=True)

    def hang(seed, attempt):
        write({"seed": seed, "attempt": attempt, "outcome": "error", "certified": False,
               "why": f"no result after {timeout} s", "seconds": timeout})
        on_hang()

    rows = []
    for seed in seeds:
        past = [r for r in prior if r["seed"] == seed]
        attempt = len(past)
        row = past[-1] if past else None
        while attempt < 2 and (row is None or row["outcome"] == "error"):
            attempt += 1
            watchdog = threading.Timer(timeout, hang, (seed, attempt)) if timeout else None
            if watchdog:
                watchdog.daemon = True
                watchdog.start()
            t0 = time.perf_counter()
            try:
                row = one(seed)
            except Exception:  # noqa: BLE001 -- an instance's crash is a reading, the batch goes on
                row = {"outcome": "error", "certified": False, "why": traceback.format_exc()}
            if watchdog:
                watchdog.cancel()
            row = {"seed": seed, "attempt": attempt, **row, "seconds": round(time.perf_counter() - t0, 1)}
            write(row)
        rows.append(row)
    return rows


def seeds_for(path: Path, split: str, limit: int | None) -> list[int]:
    lo, hi = yaml.safe_load(path.read_text(encoding="utf-8"))["split"][split]
    return list(range(lo, hi + 1))[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--split", choices=["playground", "test"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true", help=f"continue the rows in --out (after exit code {HUNG})")
    args = ap.parse_args()

    from robotsmith.scenario_runtime import materialize_candidate
    from robotsmith.sim.recorder import record_episode
    from robotsmith.sim.sim_env import SimEnv

    path = REPO / "scenarios" / "acceptance" / f"{args.family}.yaml"
    teacher = path.parent / "teacher" / path.name
    seeds = seeds_for(path, args.split, args.limit)
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, args.out / path.name)
    shutil.copy(teacher, args.out / f"teacher_{path.name}")
    packages = {s: materialize_candidate(path, seed=s) for s in seeds}
    first = packages[seeds[0]].legacy_scene
    env = SimEnv.build(first, seed=seeds[0])
    fixed = {o.name_override: o.fixed_position for o in first.objects if o.name_override != "die"}
    for pkg in packages.values():
        assert all(np.allclose(o.fixed_position, fixed[o.name_override])
                   for o in pkg.legacy_scene.objects if o.name_override != "die"), "fixtures move between seeds"
    mon = Monitor(env)

    def one(seed):
        mon.start()
        return {"split": args.split, **run_instance(env, packages[seed], args.family, mon, record_episode)}

    log = args.out / "instances.jsonl"
    prior = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if args.resume else []
    with log.open("a" if args.resume else "w", encoding="utf-8") as fh:
        rows = run_batch(seeds, one, fh, prior=prior, timeout=INSTANCE_TIMEOUT_S)
    outcomes = Counter(r["outcome"] for r in rows)
    summary = {"family": args.family, "split": args.split, "seeds": seeds,
               "certified": sum(r["certified"] for r in rows), "n": len(rows), "outcomes": dict(outcomes)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"{args.family} {args.split}: {summary['certified']}/{summary['n']} certified {dict(outcomes)}")


if __name__ == "__main__":
    main()
