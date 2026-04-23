"""RoboSmith — open-loop data collection driven by TaskSpec.

Generates expert demonstrations for any registered task.
All episode logic is derived from task_spec.skills — no task-type flags.

Usage:
  python scripts/part2/collect_data.py --task pick_cube --n-episodes 100
  python scripts/part2/collect_data.py --task line_bowls --n-episodes 10
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

from robotsmith.gen.sim_env import ensure_display, SimEnv, TARGET_MARKER_SIZE
from robotsmith.gen.recorder import (
    create_dataset, record_episode, evaluate_episode, save_summary,
)
from robotsmith.gen.franka import HOME_QPOS, to_numpy
from robotsmith.motion import MotionExecutor, MotionParams
from robotsmith.orchestration import run_skills
from robotsmith.tasks import TASK_PRESETS
from robotsmith.tasks.task_spec import TaskSpec
from robotsmith.scenes.presets import SCENE_PRESETS


def parse_args():
    ap = argparse.ArgumentParser(description="TaskSpec-driven data collection")
    ap.add_argument("--task", default="pick_cube",
                    help=f"Task name ({list(TASK_PRESETS.keys())})")
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--repo-id", default="local/franka-genesis-pick")
    ap.add_argument("--save", default="/output")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-videos", action="store_true")
    ap.add_argument("--assets-root", default=None)
    ap.add_argument("--dart-sigma", type=float, default=None,
                    help="Override TaskSpec.dart_sigma")
    return ap.parse_args()


# ---- Episode sampling helpers (workspace-aware) ----

def _sample_xy(rng: random.Random, x_range, y_range):
    return (rng.uniform(*x_range), rng.uniform(*y_range))


def _sample_spaced(rng, n, x_range, y_range, min_dist=0.10):
    pts = []
    for _ in range(n):
        for _try in range(200):
            x, y = _sample_xy(rng, x_range, y_range)
            if all(np.hypot(x - px, y - py) >= min_dist for px, py in pts):
                pts.append((x, y))
                break
        else:
            pts.append(_sample_xy(rng, x_range, y_range))
    return pts


def _sample_place_target(rng, cx, cy, x_range, y_range, min_dist=0.15):
    for _ in range(100):
        px, py = _sample_xy(rng, x_range, y_range)
        if np.hypot(px - cx, py - cy) >= min_dist:
            return (px, py)
    return (cx + min_dist, cy)


def derive_skill_info(task_spec: TaskSpec):
    """Extract pick/place object names from skill sequence."""
    pick_names: list[str] = []
    place_names: list[str] = []
    for sk in task_spec.skills:
        if sk.name == "pick" and sk.target not in pick_names:
            pick_names.append(sk.target)
        elif sk.name == "place" and sk.target not in place_names:
            place_names.append(sk.target)
    return pick_names, place_names


def build_episode_positions(
    env: SimEnv,
    rng: random.Random,
    pick_names: list[str],
    place_names: list[str],
) -> dict[str, np.ndarray]:
    """Sample random positions for all pick objects and place targets.

    Returns name → np.array([x, y, z]) for every key needed by run_skills.
    """
    x_range = env.x_range
    y_range = env.y_range
    positions: dict[str, np.ndarray] = {}

    n_pick = len(pick_names)
    pick_xys = _sample_spaced(rng, n_pick, x_range, y_range, min_dist=0.10)

    for name, (x, y) in zip(pick_names, pick_xys):
        z = env.get_initial_z(name)
        positions[name] = np.array([x, y, z])

    if place_names:
        ref_x, ref_y = pick_xys[0]
        place_center = _sample_place_target(
            rng, ref_x, ref_y, x_range, y_range, min_dist=0.15)

        if len(place_names) == 1:
            z = env.get_initial_z(pick_names[0])
            positions[place_names[0]] = np.array([place_center[0],
                                                   place_center[1], z])
        else:
            spacing = 0.12
            cx, cy = place_center
            n = len(place_names)
            offsets = np.linspace(-(n - 1) / 2 * spacing,
                                  (n - 1) / 2 * spacing, n)
            for name, dy in zip(place_names, offsets):
                z = env.get_initial_z(pick_names[0])
                positions[name] = np.array([cx, cy + dy, z])

    return positions


def main():
    args = parse_args()

    if args.task not in TASK_PRESETS:
        print(f"[error] Unknown task '{args.task}'. "
              f"Available: {list(TASK_PRESETS.keys())}")
        sys.exit(1)

    task_spec = TASK_PRESETS[args.task]
    if args.dart_sigma is not None:
        task_spec = TaskSpec(**{**task_spec.to_dict(),
                                "dart_sigma": args.dart_sigma})

    pick_names, place_names = derive_skill_info(task_spec)
    print(f"[task] {task_spec.name}: {task_spec.instruction}")
    print(f"[task] pick={pick_names} place={place_names}")

    # ---- Build simulation ----
    ensure_display()

    scene_name = task_spec.scene
    if scene_name not in SCENE_PRESETS:
        print(f"[error] Unknown scene '{scene_name}'")
        sys.exit(1)
    scene_config = SCENE_PRESETS[scene_name]

    env = SimEnv.build(
        scene_config,
        assets_root=args.assets_root,
        seed=args.seed,
        fps=args.fps,
        cpu=args.cpu,
        use_videos=not args.no_videos,
    )

    executor = MotionExecutor()
    motion_params = MotionParams()

    # Optional place-target marker (single pick+place only)
    target_marker = None
    if len(place_names) == 1 and len(pick_names) == 1:
        import genesis as gs
        target_marker = env.scene.add_entity(
            morph=gs.morphs.Box(
                size=TARGET_MARKER_SIZE,
                pos=(0.55, 0.2, TARGET_MARKER_SIZE[2] / 2),
            ),
            material=gs.materials.Rigid(friction=0.5),
            surface=gs.surfaces.Default(color=(0.3, 0.8, 0.3, 0.8)),
        )

    dataset = create_dataset(
        repo_id=args.repo_id,
        fps=args.fps,
        use_videos=not args.no_videos,
    )

    # ---- Episode loop ----
    rng = random.Random(args.seed)
    out_dir = Path(args.save) / f"franka_gen_{task_spec.name}"
    episode_labels: list[dict] = []
    frames_per_episode = None
    primary_name = pick_names[0] if pick_names else None
    primary_entity = env.entity_map.get(primary_name)

    print(f"[gen] workspace x={env.x_range} y={env.y_range}")
    print(f"[gen] object_heights={env.object_heights}")
    print(f"[gen] {args.n_episodes} episodes to generate")

    for ep in range(args.n_episodes):
        positions = build_episode_positions(
            env, rng, pick_names, place_names)

        obj_xy_map = {
            name: (pos[0], pos[1])
            for name, pos in positions.items()
            if name in env.entity_map
        }
        marker_xy = None
        if target_marker is not None and place_names:
            p = positions[place_names[0]]
            marker_xy = (p[0], p[1])

        env.reset(obj_xy_map, marker_xy=marker_xy,
                  target_marker=target_marker)

        scene_state = {
            "home_qpos": HOME_QPOS.copy(),
            "positions": positions,
            "object_heights": env.object_heights,
        }

        traj = run_skills(
            task_spec.skills, env.planner, executor,
            env.solve_ik, scene_state, motion_params,
        )

        if frames_per_episode is None:
            frames_per_episode = len(traj)
            print(f"[gen] trajectory: {frames_per_episode} frames/episode")

        obj_z_hist = record_episode(
            env, dataset, traj, task_spec,
            primary_entity=primary_entity,
        )

        initial_positions = {
            name: pos.copy() for name, pos in positions.items()
            if name in env.entity_map
        }
        success = evaluate_episode(
            env, task_spec, pick_names, initial_positions)

        label = {
            "episode_index": ep,
            "task": task_spec.name,
            "success": bool(success),
        }
        for name in pick_names:
            ent = env.entity_map.get(name)
            if ent:
                fpos = to_numpy(ent.get_pos())
                label[f"{name}_final_pos"] = [float(v) for v in fpos]

        episode_labels.append(label)
        dataset.save_episode()

        status = "OK" if success else "FAIL"
        print(f"[gen] ep {ep+1}/{args.n_episodes} [{status}] "
              f"positions={{{', '.join(f'{k}=({v[0]:.3f},{v[1]:.3f})' for k, v in positions.items() if k in env.entity_map)}}}")

    if hasattr(dataset, "consolidate"):
        dataset.consolidate(run_compute_stats=True)
    print(f"[gen] dataset saved: {dataset.root}")

    ws = env.scene_config.workspace_xy
    save_summary(
        out_dir, task_spec, args.repo_id, args.n_episodes,
        frames_per_episode, args.fps, episode_labels,
        workspace_xy=((ws[0][0], ws[0][1]), (ws[1][0], ws[1][1])),
    )


if __name__ == "__main__":
    main()
