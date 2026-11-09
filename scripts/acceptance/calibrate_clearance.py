"""The execution deviation the gate's clearance has to cover, read on a scene with only the robot and its support.

  python scripts/acceptance/calibrate_clearance.py --out out/calibration --commit <sha>

Two cases, each in its own process (Genesis initialises once per process). ``empty``: ``default_rng(0)`` draws
hand poses, the position uniform over ``DEFAULT_WORKSPACE_XY`` and 0.05-0.35 m above the table, pointing down at a
uniform yaw; IK starts from home with the base joint aimed at the position, as the pick's does, and a pose is kept
when the gate passes the solution at that pose. From the settled home pose, ``JointPlanner.connect`` (its own budget)
reaches each in turn and a 0.10 m ``move_linear`` (``keep``) follows in a direction drawn uniformly on the unit
sphere. A draw the gate or the planner refuses is replaced by the next; each draw takes four uniforms (x, y, z,
yaw) or three normals (direction) from the generator, kept or not. ``held``: ``die_01`` at (0.50, 0.00) on the
table, picked with the planner's first top-down candidate, then the same with ``default_rng(1)``. Every trajectory
point is one ``control_dofs_position`` and one physics step, as the recorder runs them; the deviation of a step is
``lever_arms(held) @ |q_actual - q_cmd|``. A case that has not finished in ``CASE_SECONDS`` stops.

Writes ``segments.jsonl`` (one row per segment) and ``calibration.json``: ``e_m`` (the largest deviation),
``s_m`` (the gate's check spacing), ``r_m`` (the surface sampling radius) and ``d_m = ceil_mm(e + s / 2)``.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CASES = {"empty": 0, "held": 1}
N_GOALS = 20
LINEAR_M = 0.10
HAND_ABOVE_TABLE_M = (0.05, 0.35)
DIE_XY = (0.50, 0.00)
MAX_DRAWS = 400
CASE_SECONDS = 900


def _scene(case: str):
    from robotsmith.scenes.config import ObjectPlacement, SceneConfig

    objects = []
    if case == "held":
        x, y = DIE_XY
        objects = [ObjectPlacement("die_01", position_range=[[x, y, 0.0], [x, y, 0.0]], name_override="die")]
    return SceneConfig(name=f"clearance_calibration_{case}", objects=objects)


class _Case:
    def __init__(self, case: str):
        from robotsmith.execution import MotionExecutor
        from robotsmith.execution.gate import ExecutionGate
        from robotsmith.motion import MotionParams
        from robotsmith.motion.planner import GenesisBackend
        from robotsmith.sim.sim_env import SimEnv

        self.case, self.seed = case, CASES[case]
        self.env = SimEnv.build(_scene(case), seed=self.seed)
        self.gate = ExecutionGate(self.env)
        self.backend = GenesisBackend(self.env.solve_ik)
        self.executor, self.params = MotionExecutor(), MotionParams()
        self.q = self._actual()
        self.holding = None
        self.rows: list[dict] = []
        self.steps: list[float] = []
        self.t0 = time.perf_counter()

    def _in_time(self) -> None:
        if time.perf_counter() - self.t0 > CASE_SECONDS:
            raise RuntimeError(f"{self.case} did not finish in {CASE_SECONDS} s ({len(self.rows)} segments)")

    # ---- execution ----

    def _actual(self) -> np.ndarray:
        from robotsmith.sim.franka import to_numpy

        return to_numpy(self.env.franka.get_dofs_position(self.env.motors_dof)).reshape(-1).astype(np.float64)

    def _execute(self, trajectory, lever) -> dict:
        env = self.env
        devs, arm_devs, terms = [], [], []
        for q in trajectory:
            q = np.asarray(q, dtype=np.float64)
            env.franka.control_dofs_position(q, env.motors_dof)
            env.step()
            term = lever * np.abs(self._actual() - q)
            devs.append(float(term.sum()))
            arm_devs.append(float(term[:7].sum()))
            terms.append(term)
        self.q = np.asarray(trajectory[-1], dtype=np.float64)
        k = int(np.argmax(devs))
        return {"points": len(devs), "max_dev_m": round(max(devs), 6), "arm_max_dev_m": round(max(arm_devs), 6),
                "max_at": k, "max_joint": int(np.argmax(terms[k])), "devs": devs}

    def _held(self):
        return None if self.holding is None else (self.holding.target, self.holding.t_hand_object)

    def _rebind(self) -> None:
        from robotsmith.motion.plan_state import binding

        if self.holding is None:
            return
        z = float(self.env.object_pose("die")[0][2])
        if z < self.env.table_surface_z + 0.03:
            raise RuntimeError(f"the die left the hand (z {z:.3f})")
        self.holding = binding("die", self.env.hand_pose(self.q), self.env.object_pose("die"))

    def _record(self, kind: str, q_from, trajectory, lever, extra: dict) -> None:
        out = self._execute(trajectory, lever)
        self.steps += out.pop("devs")
        row = {"case": self.case, "rng_seed": self.seed, "kind": kind, "q_from": np.round(q_from, 5).tolist(),
               "q_to": np.round(np.asarray(trajectory[-1]), 5).tolist(), **out, **extra}
        self.rows.append(row)
        print(json.dumps({k: row[k] for k in ("case", "kind", "points", "max_dev_m", "arm_max_dev_m")}), flush=True)

    # ---- the pick ----

    def pick(self) -> dict:
        from robotsmith.execution.gate import poses_of, spans_of
        from robotsmith.execution.meta_actions import pick_actions
        from robotsmith.motion import gripper
        from robotsmith.motion.joint_planner import JointPlanner
        from robotsmith.motion.plan_state import PlanState, binding
        from robotsmith.sim.franka import HOME_QPOS

        env = self.env
        positions = env.sync_object_states()
        plans = env.planner.plan(np.asarray(positions["die"]), object_quat=env.object_quats.get("die"),
                                 category="die", asset=env.asset_map["die"], object_height=env.object_heights["die"])
        plan = next(p for p in plans if p.metadata.get("kind") == "top_down")
        q0, solve_ik = self.q.copy(), self.backend.solve_ik
        hand = env.hand_pose(q0)
        opened = gripper.expand(gripper.Gripper(*hand, plan.finger_open), q0, solve_ik).q_target
        init = np.asarray(HOME_QPOS, dtype=np.float64).copy()
        init[0] = float(np.arctan2(float(plan.grasp_pos[1]), float(plan.grasp_pos[0])))
        q_pre = np.asarray(solve_ik(plan.pre_grasp_pos, plan.pre_grasp_quat, plan.finger_open, init_qpos=init),
                           dtype=np.float64)
        q_pre[7:] = opened[7:]
        holding = binding("die", (plan.grasp_pos, plan.grasp_quat), env.object_pose("die"))
        approach = JointPlanner(env, self.gate).connect(opened, q_pre, target="die", label="approach")
        result = self.executor.run("pick", pick_actions(plan, q0, hand, approach.motion.trajectory, self.params,
                                                        solve_ik, grasp=holding),
                                   self.backend, PlanState(q0), self.params)
        motion = self.gate.verify(result.trajectory, spans_of(result.segments), target="die", q_start=q0,
                                  holding=holding, poses=poses_of(result.segments))
        z0 = float(env.object_pose("die")[0][2])
        out = self._execute(motion.trajectory, env.lever_arms())
        out.pop("devs")
        lift = float(env.object_pose("die")[0][2]) - z0
        self.holding = holding
        self._rebind()
        return {"candidate_index": plan.metadata.get("candidate_index"), "lift_m": round(lift, 4), **out}

    # ---- the segments ----

    def _legal_goal(self, rng, span) -> np.ndarray | None:
        from robotsmith.grasp.transforms import (
            TOP_DOWN_QUAT,
            quat_wxyz_to_matrix,
            rotmat_to_quat_wxyz,
        )
        from robotsmith.scenes.config import DEFAULT_WORKSPACE_XY
        from robotsmith.sim.franka import HOME_QPOS

        (x0, y0), (x1, y1) = DEFAULT_WORKSPACE_XY
        (z0, z1), table = HAND_ABOVE_TABLE_M, self.env.table_surface_z
        x, y, z, yaw = rng.uniform([x0, y0, table + z0, -np.pi], [x1, y1, table + z1, np.pi])
        c, s = np.cos(yaw), np.sin(yaw)
        quat = rotmat_to_quat_wxyz(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                                   @ quat_wxyz_to_matrix(np.asarray(TOP_DOWN_QUAT, dtype=np.float64)))
        init = np.asarray(HOME_QPOS, dtype=np.float64).copy()
        init[0] = float(np.arctan2(y, x))
        q = np.asarray(self.backend.solve_ik(np.array([x, y, z]), quat, float(self.q[7]), init_qpos=init),
                       dtype=np.float64)
        q[7:] = self.q[7:]
        why = self.gate.check_point(q, span, target="die", held=self._held(), pose=(np.array([x, y, z]), quat))
        return q if why is None else None

    def connect(self, rng) -> None:
        from robotsmith.execution.gate import Refused, Span
        from robotsmith.motion.joint_planner import JointPlanner

        span = Span("move_to", 0, 1, holding=self.holding is not None)
        for draw in range(1, MAX_DRAWS + 1):
            self._in_time()
            goal = self._legal_goal(rng, span)
            if goal is None:
                continue
            self._rebind()
            try:
                c = JointPlanner(self.env, self.gate).connect(self.q, goal, target="die", holding=self.holding)
            except Refused as exc:
                print(json.dumps({"case": self.case, "kind": "connect", "draw": draw, "refused": exc.evidence.get(
                    "reason"), "queries": exc.evidence.get("queries"), "seconds": exc.evidence.get("seconds")}),
                      flush=True)
                continue
            q_from = self.q.copy()
            self._record("connect", q_from, c.motion.trajectory, self.env.lever_arms(self._held()),
                         {"draws": draw, "queries": c.queries, "direct": c.direct})
            return
        raise RuntimeError(f"no connection in {MAX_DRAWS} draws")

    def linear(self, rng) -> None:
        from robotsmith.execution.gate import MotionRejected, poses_of, spans_of
        from robotsmith.motion.chain import RuntimeMotionWaypoint
        from robotsmith.motion.plan_state import PlanState

        for draw in range(1, MAX_DRAWS + 1):
            self._in_time()
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            self._rebind()
            pos, quat = self.env.hand_pose(self.q)
            width = float(self.q[7])
            here = RuntimeMotionWaypoint(pos=pos, quat=quat, finger_width=width, label="move", path=())
            wp = RuntimeMotionWaypoint(pos=pos + LINEAR_M * u, quat=quat, finger_width=width, label="move",
                                       linear="keep")
            try:
                result = self.executor.run("move", [here, wp], self.backend, PlanState(self.q, self.holding),
                                           self.params)
                motion = self.gate.verify(result.trajectory, spans_of(result.segments), target="die", q_start=self.q,
                                          holding=self.holding, poses=poses_of(result.segments))
            except MotionRejected:
                continue
            q_from = self.q.copy()
            self._record("linear", q_from, motion.trajectory, self.env.lever_arms(self._held()),
                         {"draws": draw, "direction": np.round(u, 5).tolist()})
            return
        raise RuntimeError(f"no legal move_linear in {MAX_DRAWS} draws")


def run_case(case: str, out: Path) -> None:
    c = _Case(case)
    pick = c.pick() if case == "held" else None
    rng = np.random.default_rng(c.seed)
    for _ in range(N_GOALS):
        c.connect(rng)
        c.linear(rng)
    with (out / f"segments_{case}.jsonl").open("w", encoding="utf-8") as fh:
        for row in c.rows:
            fh.write(json.dumps(row) + "\n")
    (out / f"steps_{case}.json").write_text(json.dumps({"pick": pick, "devs": c.steps}) + "\n", encoding="utf-8")


def summarize(out: Path, commit: str) -> dict:
    rows, devs = [], []
    for case in CASES:
        rows += [json.loads(line) for line in (out / f"segments_{case}.jsonl").read_text(encoding="utf-8").splitlines()]
        devs += json.loads((out / f"steps_{case}.json").read_text(encoding="utf-8"))["devs"]
    worst = max(range(len(rows)), key=lambda i: rows[i]["max_dev_m"])
    e = rows[worst]["max_dev_m"]
    s, r = 0.005, 0.001
    pick = json.loads((out / "steps_held.json").read_text(encoding="utf-8"))["pick"]
    cal = {
        "e_m": e, "s_m": s, "r_m": r, "d_m": math.ceil(round((e + s / 2) * 1000, 6)) / 1000,
        "segments": {case: sum(row["case"] == case for row in rows) for case in CASES},
        "rng_seeds": dict(CASES), "commit": commit,
        "p99_m": round(float(np.percentile(devs, 99)), 6), "steps": len(devs),
        "e_at": {k: rows[worst][k] for k in ("case", "kind", "max_at", "max_joint")} | {"segment": worst},
        "arm_e_m": max(row["arm_max_dev_m"] for row in rows),
        "per_case_max_m": {case: max(row["max_dev_m"] for row in rows if row["case"] == case) for case in CASES},
        "pick": pick,
    }
    with (out / "segments.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    (out / "calibration.json").write_text(json.dumps(cal, indent=1) + "\n", encoding="utf-8")
    return cal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--case", choices=sorted(CASES), help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.case:
        run_case(args.case, args.out)
        return
    for case in CASES:
        subprocess.run([sys.executable, __file__, "--out", str(args.out), "--commit", args.commit, "--case", case],
                       check=True)
    cal = summarize(args.out, args.commit)
    print(json.dumps({k: cal[k] for k in ("e_m", "d_m", "p99_m", "arm_e_m", "e_at")}))


if __name__ == "__main__":
    main()
