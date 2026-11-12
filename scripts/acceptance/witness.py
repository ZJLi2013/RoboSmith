"""Solvability witness for the acceptance families, independent of the manipulation chain under test.

  python scripts/acceptance/witness.py --family f1_wall_gap --seeds 100 119 --out out/f1.json

A positive instance passes when a scripted top-down pick and place (family-specific grasp yaw and carry
height, Cartesian waypoints through Genesis IK, executed with physics) ends with the task predicate true and
no illegal contact (``PHASES``): the robot touches only the die, and only with its fingers between closing and
opening; its links touch each other only where they are joined; the die touches the table until it is lifted,
nothing while carried, and the mat from the lowering on, through the task judge's settle. A negative instance
passes when the die settles inside its cage.

``--heldout`` reads the family and its witness parameters from ``scenarios/acceptance/heldout/`` (``witness.json``);
this file names no held-out family. Rows go to ``<out>.jsonl`` as they finish (``batch.run_batch``), the readings
to ``--out``.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch import run_batch, supervise

from robotsmith.scenario.load import load_scenario
from robotsmith.scenario_runtime.runner import evaluate_scenario_task

TCP = 0.1034
FINGERS = {"left_finger", "right_finger"}
OPEN, CLOSED = 0.04, 0.0
RULE = "phase-v2"
# grasp yaw (fingers close along world y at 0 deg), carry height of the TCP above the table
FAMILIES = {
    "f0_open": {"yaw": 0.0, "carry": 0.20},
    "f1_wall_gap": {"yaw": 0.0, "carry": 0.20},
    "f2_over_barrier": {"yaw": 90.0, "carry": 0.34},
    "f4_caged": None,
}
# --control: the same witness ignoring the family's constraint, which must then fail.
CONTROLS = {
    "f1_wall_gap": {"yaw": 90.0, "carry": 0.20},
    "f2_over_barrier": {"yaw": 90.0, "carry": 0.20},
}
ACCEPTANCE = REPO / "scenarios" / "acceptance"
# phase -> (fingers may touch the die, what else the die may touch)
PHASES = {
    "approach": (False, {"env"}),
    "descend": (False, {"env"}),
    "close": (True, {"env"}),
    "lift": (True, {"env"}),
    "carry": (True, set()),
    "lower": (True, {"mat"}),
    "open": (True, {"mat"}),
    "retreat": (False, {"mat"}),
    "settle": (False, {"mat"}),
}
TABLE_Z = 0.775
DIE_HALF = 0.025
MAT_TOP = 0.01


def _np(t):
    return np.asarray(t.cpu().numpy() if hasattr(t, "cpu") else t)


def _quat(yaw_deg: float) -> np.ndarray:
    h = math.radians(yaw_deg) / 2
    return np.array([0.0, math.cos(h), math.sin(h), 0.0], dtype=np.float32)


class Contacts:
    def __init__(self, env):
        self.owner = {}
        for name, ent in [("robot", env.franka), *env.entity_map.items()]:
            for link in ent.links:
                self.owner[link.idx] = (name, link.name)
        parent = {lk.name: self.owner.get(lk.parent_idx, (None, None))[1] for lk in env.franka.links}
        self.joined = {frozenset((a, b)) for a in parent for b in parent
                       if a != b and (parent[a] == b or parent[b] == a or parent[a] == parent[b])}
        self.env = env

    def pairs(self, ent):
        c = ent.get_contacts()
        a, b = _np(c["link_a"]).reshape(-1), _np(c["link_b"]).reshape(-1)
        if "valid_mask" in c:
            m = _np(c["valid_mask"]).reshape(-1).astype(bool)
            a, b = a[m], b[m]
        return [(self.owner.get(int(i), ("env", "?")), self.owner.get(int(j), ("env", "?"))) for i, j in zip(a, b)]

    def illegal(self, phase: str) -> list[str]:
        gripping, die_may = PHASES[phase]
        out = []
        for x, y in self.pairs(self.env.franka):
            if x[0] == "robot" and y[0] == "robot":
                if frozenset((x[1], y[1])) not in self.joined:
                    out.append(f"robot.{x[1]}-robot.{y[1]}")
                continue
            robot, other = (x, y) if x[0] == "robot" else (y, x)
            if not (gripping and other[0] == "die" and robot[1] in FINGERS):
                out.append(f"robot.{robot[1]}-{other[0]}")
        for x, y in self.pairs(self.env.entity_map["die"]):
            other = y if x[0] == "die" else x
            if other[0] not in ("die", "robot", *die_may):
                out.append(f"die-{other[0]}")
        return out


def _cartesian(env, q, start, end, quat, fingers, n=30):
    """IK path for the TCP from ``start`` to ``end``; the IK target is the hand link, TCP above it."""
    path = []
    for t in np.linspace(0, 1, n + 1)[1:]:
        hand = start + t * (end - start) + [0, 0, TCP]
        q = env.solve_ik(hand, quat=quat, finger_pos=fingers, init_qpos=q)
        path.append(q)
    return path


def _hold(q, fingers, n=40):
    q = q.copy()
    q[7:] = fingers
    return [q] * n


def difficulty(family: str, placed: dict, die: np.ndarray) -> dict:
    """The family's continuous difficulty parameter for the settled die and the placed fixtures."""
    if family == "f1_wall_gap":
        return {"gap_m": round(float(placed["wall"][0] - 0.01 - (die[0] + 0.023)), 4)}
    if family == "f2_over_barrier":
        return {"dist_m": round(float(die[1] - placed["barrier"][1] - 0.01 - DIE_HALF), 4)}
    if family == "f0_open":
        return {"reach_m": round(float(np.hypot(die[0], die[1])), 4)}
    return {}


def wall_gap(placed: dict, die: np.ndarray, of: str, yaw_deg: float) -> dict:
    """Gap across the thickness (local x) of the 20 mm wall ``of`` yawed ``yaw_deg``, to a die yawed with it."""
    d = die[:2] - placed[of][:2]
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return {"gap_m": round(float(abs(c * d[0] + s * d[1]) - 0.01 - DIE_HALF), 4)}


class Judge:
    """Counts every physics step by phase and keeps the first illegal contact."""

    def __init__(self, contacts: Contacts, phases):
        self.contacts, self.steps, self.first = contacts, {p: 0 for p in phases}, None

    def __call__(self, phase: str) -> bool:
        self.steps[phase] = self.steps.get(phase, 0) + 1
        bad = self.contacts.illegal(phase)
        if bad and self.first is None:
            self.first = f"{phase}: {bad[0]}"
        return bool(bad)


def judge_task(env, task, initial: dict, hold, judge: Judge) -> bool:
    """The task predicate after the judge's settle, every settle step judged as ``settle``."""
    env.on_step = lambda _env: judge("settle")
    try:
        ok, _ = evaluate_scenario_task(env, task, initial_positions=initial, hold=hold)
    finally:
        env.on_step = None
    return bool(ok)


def caged(placed: dict, die: np.ndarray) -> str:
    """Why the settled die is not inside its cage; empty when it is."""
    off = np.abs(die[:2] - placed["cage"][:2])
    inside = bool(np.all(off < 0.05 - 0.025) and die[2] < TABLE_Z + 0.06)
    return "" if inside else f"die not inside cage: {off}"


def witness(env, task, fam: dict, die: np.ndarray, mat: np.ndarray) -> dict:
    quat = _quat(fam["yaw"])
    carry = TABLE_Z + fam["carry"]
    above = die + [0, 0, 0.12]
    grasp = die.copy()
    lift = np.array([die[0], die[1], carry])
    over = np.array([mat[0], mat[1], carry])
    drop = np.array([mat[0], mat[1], mat[2] + MAT_TOP + DIE_HALF + 0.01])
    up = drop + [0, 0, 0.10]

    q = _np(env.franka.get_dofs_position(env.motors_dof)).astype(np.float32)
    here = _np(env.end_effector.get_pos()).reshape(-1) - [0, 0, TCP]
    plan = []
    segs = [("approach", here, above, OPEN), ("descend", above, grasp, OPEN)]
    for name, a, b, f in segs:
        plan.append((name, _cartesian(env, q, a, b, quat, f)))
        q = plan[-1][1][-1]
    plan.append(("close", _hold(q, CLOSED)))
    q = plan[-1][1][-1]
    for name, a, b in [("lift", grasp, lift), ("carry", lift, over), ("lower", over, drop)]:
        plan.append((name, _cartesian(env, q, a, b, quat, CLOSED)))
        q = plan[-1][1][-1]
    plan.append(("open", _hold(q, OPEN)))
    q = plan[-1][1][-1]
    plan.append(("retreat", _cartesian(env, q, drop, up, quat, OPEN, n=15)))

    initial = {n: _np(e.get_pos()).reshape(-1).copy() for n, e in env.entity_map.items()}
    judge = Judge(Contacts(env), [name for name, _ in plan] + ["settle"])
    # Stops at the first illegal contact: the verdict is already decided, and penetrating contacts slow the solver.
    for name, path in plan:
        for qq in path:
            target = np.asarray(qq, dtype=np.float32)
            for _ in range(3):
                env.franka.control_dofs_position(target, env.motors_dof)
                env.scene.step()
                if judge(name):
                    return {"success": False, "first_illegal": judge.first, "phase_steps": judge.steps}
    ok = judge_task(env, task, initial, np.asarray(plan[-1][1][-1]), judge)
    return {"success": ok, "first_illegal": judge.first, "phase_steps": judge.steps}


def main() -> None:
    from robotsmith.sim.sim_env import SceneInvalid, SimEnv

    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--seeds", type=int, nargs=2, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue the rows in --out")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    head = {"family": args.family, "rule": RULE}
    failed = {**head, "settle_ok": None, "witness_ok": False}
    log = args.out.with_suffix(".jsonl")
    if not args.worker:
        log.parent.mkdir(parents=True, exist_ok=True)
        sys.exit(supervise([sys.executable, __file__, *sys.argv[1:], "--worker"], log, failed=failed))
    if args.heldout:
        path = ACCEPTANCE / "heldout" / f"{args.family}.yaml"
        spec = json.loads((path.parent / "witness.json").read_text(encoding="utf-8"))[args.family]
        fam = spec["control"] if args.control else spec["witness"]
        gap = spec["gap"]
    else:
        path = ACCEPTANCE / f"{args.family}.yaml"
        fam = CONTROLS[args.family] if args.control else FAMILIES[args.family]
        gap = None
    seeds = list(range(args.seeds[0], args.seeds[1] + 1))
    packages = {s: load_scenario(path, seed=s) for s in seeds}
    first = packages[seeds[0]].scene
    env = SimEnv.build(first, seed=seeds[0])
    fixed = {o.name_override: o.fixed_position for o in first.objects if o.name_override != "die"}
    for pkg in packages.values():
        scene = pkg.scene
        assert all(np.allclose(o.fixed_position, fixed[o.name_override])
                   for o in scene.objects if o.name_override != "die"), "fixtures move between seeds"
        assert tuple(scene.table_size) == tuple(first.table_size), "table size changes between seeds"

    def one(seed):
        pkg = packages[seed]
        placed = {o.name_override: np.asarray(o.fixed_position, dtype=float) for o in pkg.scene.objects}
        row = {**head, "die_xyz": placed["die"].round(4).tolist()}
        try:
            env.reset({"die": tuple(placed["die"])}, settle_steps=30)
        except SceneInvalid as e:
            return {**row, "settle_ok": False, "witness_ok": False, "why": str(e)}
        row["settle_ok"] = True
        die = _np(env.entity_map["die"].get_pos()).reshape(-1)
        row.update(wall_gap(placed, die, gap["of"], gap["yaw_deg"]) if gap else difficulty(args.family, placed, die))
        if fam is None:
            why = caged(placed, die)
            return {**row, "witness_ok": not why, "why": why}
        w = witness(env, pkg.task, fam, die, placed["mat"])
        ok = w["success"] and w["first_illegal"] is None
        return {**row, **w, "witness_ok": ok, "why": "" if ok else str(w["first_illegal"] or "task failed")}

    prior = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if args.resume else []
    with log.open("a" if args.resume else "w", encoding="utf-8") as fh:
        rows = run_batch(seeds, one, fh, prior=prior, failed=failed, running=log.with_suffix(".running"))
    args.out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{args.family}: {sum(r['witness_ok'] for r in rows)}/{len(rows)} witnessed")


if __name__ == "__main__":
    main()
