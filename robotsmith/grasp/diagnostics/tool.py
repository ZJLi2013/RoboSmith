"""The diagnosis agent's ``try_grasp``: trial up to 8 proposed grasps from the failed run's start state.

python -m robotsmith.grasp.diagnostics.tool <context.json> <candidates.json> [result.json]

``context.json`` is written by ``request.write_request``. Each candidate is ``{"tip_xyz", "approach", "finger_axis",
"ik_seed"}`` in the world frame (metres): ``tip_xyz`` is the TCP between the finger pads, ``approach`` points from the
hand toward the object, ``finger_axis`` is the closing direction (made orthogonal to ``approach``), ``ik_seed`` is
7 joint angles or null.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

MAX_CANDIDATES = 8


def _plans(planner, candidates: list[dict]):
    plans = []
    for i, c in enumerate(candidates[:MAX_CANDIDATES]):
        a = np.asarray(c["approach"], dtype=np.float64)
        a /= np.linalg.norm(a)
        f = np.asarray(c["finger_axis"], dtype=np.float64)
        f = f - a * (f @ a)
        f /= np.linalg.norm(f)
        p = planner._plan("agent", np.asarray(c["tip_xyz"], dtype=np.float64), a, f, 0.0, "")
        p.metadata["candidate_index"] = i
        if c.get("ik_seed"):
            p.metadata["ik_seed"] = [float(v) for v in c["ik_seed"]][:7]
        plans.append(p)
    return plans


def try_candidates(context: dict, candidates: list[dict], out_dir: Path) -> dict:
    """Run one validation episode whose only candidates are ``candidates``; per-candidate lift and the label."""
    from robotsmith.grasp import geometric
    from robotsmith.rollout.asset_pick_validation import AssetPickValidationConfig, run_asset_pick
    from robotsmith.sim.sim_env import ensure_display
    from robotsmith.skills import pick

    plans = _plans(geometric.GeometricGraspPlanner(), candidates)
    plan_geometric, seed_transit = geometric.GeometricGraspPlanner.plan, pick._pick_seed_transit

    def seeded(ctx, plan):
        seed, transit = seed_transit(ctx, plan)
        if plan.metadata.get("ik_seed"):
            q0 = np.asarray(ctx.qpos, dtype=np.float64)
            seed = seed.copy()
            seed[:7] = plan.metadata["ik_seed"]
            steps = max(ctx.params.approach_steps, ctx.params.min_steps)
            transit = [q0 + (seed - q0) * (k + 1) / steps for k in range(steps)]
        return seed, transit

    geometric.GeometricGraspPlanner.plan = lambda self, *a, **k: plans
    pick._pick_seed_transit = seeded
    try:
        ensure_display()
        asset = context["asset"]
        run_asset_pick(AssetPickValidationConfig(
            assets=[asset], output_dir=out_dir, n_episodes=1, seed=context["seed"], no_videos=True, clean=True,
            settle_steps=context["settle_steps"], grasp_planner="geometric",
            object_xy=tuple(context["object_xy"]), grasp_selection="trial_all"), asset)
    finally:
        geometric.GeometricGraspPlanner.plan, pick._pick_seed_transit = plan_geometric, seed_transit
    label = json.loads((out_dir / asset / f"franka_gen_pick_{asset}" / "episode_labels.json").read_text())[0]
    results = [{"candidate": t["candidate_index"], "lifted": t["lifted"], "lift_m": t.get("lift_m"),
                "motion_error": t.get("motion_error"), "hand_pos": [round(float(v), 4) for v in plans[t["candidate_index"]].grasp_pos]}
               for t in label.get("trials") or []]
    return {"results": results, "label": label}


def main() -> None:
    context = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    candidates = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        out = try_candidates(context, candidates, Path(tmp))
    text = json.dumps({"results": out["results"]}, indent=1)
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
