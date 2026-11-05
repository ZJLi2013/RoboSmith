"""Solvability witness for the acceptance families, independent of the manipulation chain under test.

  python scripts/acceptance/witness.py --family f1_wall_gap --seeds 100 119 --out out/f1.json

A positive instance passes when a scripted top-down pick and place (family-specific grasp yaw and carry
height, Cartesian waypoints through Genesis IK, executed with physics) ends with the task predicate true and
no illegal contact: the robot touches only the die, and only with its fingers between closing and opening;
the die touches nothing but the table, the mat and the fingers. A negative instance passes when the die
settles inside its cage.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from robotsmith.scenario_runtime import materialize_candidate
from robotsmith.scenario_runtime.runner import evaluate_scenario_task
from robotsmith.sim.sim_env import SceneInvalid, SimEnv

TCP = 0.1034
FINGERS = {"left_finger", "right_finger"}
OPEN, CLOSED = 0.04, 0.0
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
        self.env = env

    def pairs(self, ent):
        c = ent.get_contacts()
        a, b = _np(c["link_a"]).reshape(-1), _np(c["link_b"]).reshape(-1)
        if "valid_mask" in c:
            m = _np(c["valid_mask"]).reshape(-1).astype(bool)
            a, b = a[m], b[m]
        return [(self.owner.get(int(i), ("env", "?")), self.owner.get(int(j), ("env", "?"))) for i, j in zip(a, b)]

    def illegal(self, gripping: bool) -> list[str]:
        out = []
        for x, y in self.pairs(self.env.franka):
            if x[0] == "robot" and y[0] == "robot":
                continue
            robot, other = (x, y) if x[0] == "robot" else (y, x)
            if not (gripping and other[0] == "die" and robot[1] in FINGERS):
                out.append(f"robot.{robot[1]}-{other[0]}")
        for x, y in self.pairs(self.env.entity_map["die"]):
            other = y if x[0] == "die" else x
            if other[0] not in ("die", "robot", "env", "mat"):
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
        plan.append((name, False, _cartesian(env, q, a, b, quat, f)))
        q = plan[-1][2][-1]
    plan.append(("close", True, _hold(q, CLOSED)))
    q = plan[-1][2][-1]
    for name, a, b in [("lift", grasp, lift), ("carry", lift, over), ("lower", over, drop)]:
        plan.append((name, True, _cartesian(env, q, a, b, quat, CLOSED)))
        q = plan[-1][2][-1]
    plan.append(("open", True, _hold(q, OPEN)))
    q = plan[-1][2][-1]
    plan.append(("retreat", False, _cartesian(env, q, drop, up, quat, OPEN, n=15)))

    initial = {n: _np(e.get_pos()).reshape(-1).copy() for n, e in env.entity_map.items()}
    contacts = Contacts(env)
    # Stops at the first illegal contact: the verdict is already decided, and penetrating contacts slow the solver.
    for name, gripping, path in plan:
        for qq in path:
            target = np.asarray(qq, dtype=np.float32)
            for _ in range(3):
                env.franka.control_dofs_position(target, env.motors_dof)
                env.scene.step()
                bad = contacts.illegal(gripping)
                if bad:
                    return {"success": False, "first_illegal": f"{name}: {bad[0]}"}
    ok, _ = evaluate_scenario_task(env, task, initial_positions=initial, hold=np.asarray(plan[-1][2][-1]))
    return {"success": bool(ok), "first_illegal": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(FAMILIES))
    ap.add_argument("--seeds", type=int, nargs=2, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--control", action="store_true")
    args = ap.parse_args()
    fam = CONTROLS[args.family] if args.control else FAMILIES[args.family]
    path = REPO / "scenarios" / "acceptance" / f"{args.family}.yaml"
    seeds = list(range(args.seeds[0], args.seeds[1] + 1))
    packages = {s: materialize_candidate(path, seed=s) for s in seeds}
    first = packages[seeds[0]].legacy_scene
    env = SimEnv.build(first, seed=seeds[0])
    fixed = {o.name_override: o.fixed_position for o in first.objects if o.name_override != "die"}
    rows = []
    for seed in seeds:
        pkg = packages[seed]
        scene = pkg.legacy_scene
        placed = {o.name_override: np.asarray(o.fixed_position, dtype=float) for o in scene.objects}
        assert all(np.allclose(placed[k], v) for k, v in fixed.items()), "fixtures move between seeds"
        assert tuple(scene.table_size) == tuple(first.table_size), "table size changes between seeds"
        row = {"family": args.family, "seed": seed, "die_xyz": placed["die"].round(4).tolist()}
        try:
            env.reset({"die": tuple(placed["die"])}, settle_steps=30)
            row["settle_ok"] = True
        except SceneInvalid as e:
            rows.append({**row, "settle_ok": False, "witness_ok": False, "why": str(e)})
            continue
        die = _np(env.entity_map["die"].get_pos()).reshape(-1)
        row.update(difficulty(args.family, placed, die))
        if fam is None:
            why = caged(placed, die)
            rows.append({**row, "witness_ok": not why, "why": why})
            continue
        w = witness(env, pkg.legacy_task, fam, die, placed["mat"])
        ok = w["success"] and w["first_illegal"] is None
        rows.append({**row, **w, "witness_ok": ok, "why": "" if ok else str(w)})
        print(json.dumps(rows[-1]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{args.family}: {sum(r['witness_ok'] for r in rows)}/{len(rows)} witnessed")


if __name__ == "__main__":
    main()
