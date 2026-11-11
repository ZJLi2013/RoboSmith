"""Certified readings of the manipulation chain on a scenario's instances.

  python scripts/acceptance/certify.py --scenario scenarios/acceptance/f0_open.yaml --split test --out out/f0_open_test

Each instance runs the scenario's plan file (its ``actions`` when present) once through ``EpisodeRun``
(skills, execution gate, executor) on the seed's layout: one scene is built per group of seeds whose scenes differ
only in where the picked objects start (``scene_groups``), and each instance moves them to its seed's start. Every physics step after the reset is read for the contacts it resolved and judged by
``gate.certified`` for the span being executed: trial and replay steps by the verified motion's spans, step
and task settles by the last replayed span. A positive instance is certified when the task predicate holds
and no step of any stage broke the rule; a negative one when its first step is refused and nothing ran
after the reset. Writes ``instances.jsonl`` (one row per instance, as it runs), the inputs and ``summary.json``.
"""

import argparse
import dataclasses
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch import run_batch, supervise  # noqa: E402
from robotsmith.execution.gate import REFUSALS, SUPPORT, Span, certified  # noqa: E402

SETTLE_STEPS = 30
NEGATIVE = {"f4_caged"}


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


def _content(trajectory) -> bytes:
    # Each read of VerifiedMotion.trajectory is a new tuple, so replays are matched to their motion by value.
    return b"".join(np.asarray(q, dtype=np.float64).tobytes() for q in trajectory)


def run_instance(env, pkg, family, mon, record_episode):
    from robotsmith.execution import MotionExecutor
    from robotsmith.motion import MotionParams
    from robotsmith.scenario.adapters import resolved_target_positions, target_frames
    from robotsmith.scenario_runtime.runner import (EpisodeRun, evaluate_scenario_task, reset_scene,
                                                    resolve_static_target_positions)
    from robotsmith.sim.sim_env import SceneInvalid
    from witness import FAMILIES, caged, difficulty

    task = pkg.legacy_task
    placed = {o.name_override: np.asarray(o.fixed_position, dtype=float) for o in pkg.legacy_scene.objects}
    picked = movers(pkg)
    for name in picked:
        env.placed_map[name] = dataclasses.replace(env.placed_map[name], position=list(placed[name]))
    row = {"family": family, "layout": {k: _tolist(v) for k, v in placed.items()}}
    mon.enter("init")
    try:
        reset_scene(env, SETTLE_STEPS)
    except SceneInvalid as exc:
        return {**row, "settle_ok": False, "outcome": "init_failed", "why": str(exc), "steps": dict(mon.steps)}
    settled = {n: _tolist(p) for n, p in env.sync_object_states().items()}
    row.update(settle_ok=True, settled=settled)
    if family in FAMILIES:
        (obj,) = picked
        row.update(difficulty(family, placed, np.asarray(settled[obj])))
        if family in NEGATIVE:
            why = caged(placed, np.asarray(settled[obj]))
            row.update(caged=not why, caged_why=why)

    meta, last = {}, [None]
    run = EpisodeRun(env, task, executor=MotionExecutor(), motion_params=MotionParams(), dataset=_NoDataset(),
                     record_fn=None, target_frames=target_frames(pkg.scenario_scene), camera_names=(),
                     env_state_names=())
    verify = run.gate.verify

    def verified(trajectory, spans, *, target, rest=(), **kw):
        m = verify(trajectory, spans, target=target, rest=rest, **kw)
        meta[_content(m.trajectory)] = (m.spans, target, {SUPPORT, *rest}, run.skills[-1].name)
        return m

    def record(env_, dataset, seg, task_spec, **kw):
        ctx = meta.get(_content(seg))
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
        mon.enter("trial", meta.get(_content(trajectory)))
        try:
            return trial(trajectory, target, **kw)
        finally:
            mon.enter("between")

    run.gate.verify, run.record_fn, env.trial_from_here = verified, record, trialled
    try:
        mon.enter("between")
        run.run_steps(list(task.skills), list(task.step_bindings))
        if run.stopped is not None:
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
        success=bool(success), frame_count=run.frame_count, stopped=run.stopped,
        handoffs=[{k: h.get(k) for k in ("skill", "phase", "done", "cause")} for h in run.handoffs],
        trials=[t.get("trials") for t in run.skill_traces if "trials" in t],
        transports=[t["transport"] for t in run.skill_traces if t.get("transport")],
        task_judge={"object_positions": judged.get("object_positions")} if judged else None,
    )
    clean = not mon.violating
    if family in NEGATIVE:
        executed = sum(steps.values())
        first = run.handoffs[0] if run.handoffs else {}
        row["certified"] = bool(first.get("cause") in REFUSALS and executed == 0)
        row["outcome"] = "negative_ok" if row["certified"] else ("executed" if executed else "not_refused")
    else:
        row["certified"] = bool(success and clean)
        row["outcome"] = ("certified" if row["certified"] else run.stopped["cause"] if run.stopped
                          else "violation" if not clean else "task_failed")
    return row


class _NoDataset:
    def add_frame(self, frame):
        pass


def movers(pkg) -> set[str]:
    """The objects the plan picks: the ones whose start differs between seeds of one built scene."""
    return {s.target for s in pkg.legacy_task.skills if s.name == "pick"}


def scene_groups(packages: dict) -> list[list[int]]:
    """The seeds in groups one built scene serves: within a group the scene differs only in where the picked
    objects start, the one thing ``run_instance`` changes between instances."""
    groups: dict[str, list[int]] = {}
    for seed, pkg in packages.items():
        scene, picked = pkg.legacy_scene, movers(pkg)
        objects = [(o.name_override, o.asset_query, o.yaw_deg, sorted((o.joint_init or {}).items()),
                    None if o.name_override in picked else _tolist(o.fixed_position)) for o in scene.objects]
        groups.setdefault(repr((objects, _tolist(scene.table_size), scene.table_height)), []).append(seed)
    return list(groups.values())


def seeds_for(path: Path, split: str, limit: int | None) -> list[int]:
    lo, hi = yaml.safe_load(path.read_text(encoding="utf-8"))["split"][split]
    return list(range(lo, hi + 1))[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=Path, required=True, help="scenario YAML; its plan file is plans/<name>.yaml")
    ap.add_argument("--split", choices=["playground", "test"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true", help="continue the rows in --out")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    log = args.out / "instances.jsonl"
    if not args.worker:
        args.out.mkdir(parents=True, exist_ok=True)
        sys.exit(supervise([sys.executable, __file__, *sys.argv[1:], "--worker"], log, failed={"certified": False}))

    from robotsmith.scenario_runtime import materialize_candidate
    from robotsmith.sim.recorder import record_episode
    from robotsmith.sim.sim_env import SimEnv

    path = args.scenario.resolve()
    family = path.stem
    plan_file = path.parent / "plans" / path.name
    seeds = seeds_for(path, args.split, args.limit)
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, args.out / path.name)
    shutil.copy(plan_file, args.out / f"plan_{path.name}")
    packages = {s: materialize_candidate(path, seed=s) for s in seeds}
    groups = scene_groups(packages)
    built = {}

    def one(seed):
        group = next(g for g in groups if seed in g)
        if built.get("group") is not group:
            if built:
                import genesis as gs  # one Genesis session per process: a new scene needs the old one torn down
                gs.destroy()
                built.clear()
            env = SimEnv.build(packages[group[0]].legacy_scene, seed=group[0])
            built.update(group=group, env=env, mon=Monitor(env))
        built["mon"].start()
        return {"split": args.split,
                **run_instance(built["env"], packages[seed], family, built["mon"], record_episode)}

    prior = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if args.resume else []
    with log.open("a" if args.resume else "w", encoding="utf-8") as fh:
        rows = run_batch([s for g in groups for s in g], one, fh, prior=prior, failed={"certified": False},
                         show=("seed", "attempt", "outcome", "first_violation", "seconds"),
                         running=log.with_suffix(".running"))
    outcomes = Counter(r["outcome"] for r in rows)
    summary = {"family": family, "split": args.split, "seeds": seeds,
               "certified": sum(r["certified"] for r in rows), "n": len(rows), "outcomes": dict(outcomes)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"{family} {args.split}: {summary['certified']}/{summary['n']} certified {dict(outcomes)}")


if __name__ == "__main__":
    main()
