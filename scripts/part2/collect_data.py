"""RoboSmith — open-loop data collection driven by TaskSpec.

Generates expert demonstrations via IK solver for any registered task.
Supports DART noise augmentation (--dart-sigma) and scene-based setup.

Usage:
  python scripts/part2/collect_data.py --task pick_cube --n-episodes 100
  python scripts/part2/collect_data.py --task pick_cube --n-episodes 100 --dart-sigma 0.005
  python scripts/part2/collect_data.py --task pick_cube --n-episodes 2 --scene tabletop_simple
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


# ---- constants ----
JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "joint5", "joint6", "joint7",
    "finger_joint1", "finger_joint2",
]
MOTOR_NAMES = [f"{j}.pos" for j in JOINT_NAMES]

HOME_QPOS = np.array([0, -0.3, 0, -2.2, 0, 2.0, 0.79, 0.04, 0.04], dtype=np.float32)
KP = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100], dtype=np.float32)
KV = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10], dtype=np.float32)
FORCE_LOWER = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100], dtype=np.float32)
FORCE_UPPER = np.array([87, 87, 87, 87, 12, 12, 12, 100, 100], dtype=np.float32)

CUBE_SIZE = (0.04, 0.04, 0.04)


def ensure_display():
    if os.environ.get("DISPLAY"):
        return
    xvfb = subprocess.run(["which", "Xvfb"], capture_output=True)
    if xvfb.returncode != 0:
        return
    proc = subprocess.Popen(
        ["Xvfb", ":99", "-screen", "0", "1280x1024x24", "-ac", "+extension", "GLX"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = ":99"
    time.sleep(2)
    if proc.poll() is None:
        print(f"[display] Xvfb started (PID={proc.pid})")


def to_numpy(t):
    arr = t.cpu().numpy() if hasattr(t, "cpu") else np.array(t)
    return arr[0] if arr.ndim > 1 else arr


def render_cam(cam):
    rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
    arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.array(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.uint8)


def main():
    from robotsmith.tasks import TASK_PRESETS, IK_STRATEGIES, TrajectoryParams
    from robotsmith.tasks import evaluate_predicate

    ap = argparse.ArgumentParser(description="TaskSpec-driven data collection")
    ap.add_argument("--task", default="pick_cube",
                    help=f"Task name. Available: {list(TASK_PRESETS.keys())}")
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--repo-id", default="local/franka-genesis-pick")
    ap.add_argument("--save", default="/output")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cube-x-min", type=float, default=0.4)
    ap.add_argument("--cube-x-max", type=float, default=0.7)
    ap.add_argument("--cube-y-min", type=float, default=-0.2)
    ap.add_argument("--cube-y-max", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cube-friction", type=float, default=1.5)
    ap.add_argument("--dart-sigma", type=float, default=None,
                    help="DART noise sigma (overrides TaskSpec.dart_sigma)")
    # trajectory params (defaults match legacy pick-cube)
    ap.add_argument("--hover-z", type=float, default=0.25)
    ap.add_argument("--grasp-z", type=float, default=0.135)
    ap.add_argument("--lift-z", type=float, default=0.30)
    ap.add_argument("--settle-steps", type=int, default=30)
    ap.add_argument("--approach-steps", type=int, default=40)
    ap.add_argument("--descend-steps", type=int, default=30)
    ap.add_argument("--grasp-hold-steps", type=int, default=20)
    ap.add_argument("--lift-steps", type=int, default=30)
    ap.add_argument("--lift-hold-steps", type=int, default=15)
    # success detection params (used for logging, predicate is authoritative)
    ap.add_argument("--success-lift-delta", type=float, default=0.02)
    ap.add_argument("--success-sustain-frames", type=int, default=8)
    ap.add_argument("--success-final-delta", type=float, default=0.01)
    # observation extras
    ap.add_argument("--add-phase", action="store_true")
    ap.add_argument("--add-goal", action="store_true")
    ap.add_argument("--no-bbox-detection", action="store_true")
    ap.add_argument("--no-videos", action="store_true")
    # scene mode
    ap.add_argument("--scene", default=None,
                    help="Scene preset or JSON path. If not set, uses legacy cube.")
    ap.add_argument("--assets-root", default=None)
    args = ap.parse_args()

    # ---- Resolve TaskSpec ----
    if args.task not in TASK_PRESETS:
        print(f"[error] Unknown task '{args.task}'. Available: {list(TASK_PRESETS.keys())}")
        sys.exit(1)
    task_spec = TASK_PRESETS[args.task]
    if args.dart_sigma is not None:
        task_spec = TaskSpec(**{**task_spec.to_dict(), "dart_sigma": args.dart_sigma})
    print(f"[task] {task_spec.name}: {task_spec.instruction}")
    print(f"[task] success_fn={task_spec.success_fn}, ik_strategy={task_spec.ik_strategy}")

    # ---- Resolve IK strategy ----
    if task_spec.ik_strategy not in IK_STRATEGIES:
        print(f"[error] Unknown ik_strategy '{task_spec.ik_strategy}'. "
              f"Available: {list(IK_STRATEGIES.keys())}")
        sys.exit(1)
    strategy = IK_STRATEGIES[task_spec.ik_strategy]

    traj_params = TrajectoryParams(
        hover_z=args.hover_z,
        grasp_z=args.grasp_z,
        lift_z=args.lift_z,
        approach_steps=args.approach_steps,
        descend_steps=args.descend_steps,
        grasp_hold_steps=args.grasp_hold_steps,
        lift_steps=args.lift_steps,
        lift_hold_steps=args.lift_hold_steps,
    )

    # ---- Genesis setup ----
    ensure_display()
    import genesis as gs
    import torch

    from robotsmith.tasks.task_spec import TaskSpec

    gs.init(backend=(gs.cpu if args.cpu else gs.gpu), logging_level="warning")

    use_scene = args.scene is not None

    if use_scene:
        from robotsmith.assets.library import AssetLibrary
        from robotsmith.scenes.backend import ProgrammaticSceneBackend
        from robotsmith.scenes.genesis_loader import load_resolved_scene

        assets_root = args.assets_root or str(
            Path(__file__).resolve().parent.parent.parent / "assets"
        )
        library = AssetLibrary(assets_root)

        if args.scene == "tabletop_simple":
            from robotsmith.scenes.presets.tabletop_simple import tabletop_simple
            scene_config = tabletop_simple
        else:
            raise ValueError(f"Unknown scene preset: {args.scene}")

        backend = ProgrammaticSceneBackend(seed=args.seed)
        resolved = backend.resolve(scene_config, library)
        print(f"[scene] {resolved.summary()}")

        handle = load_resolved_scene(
            resolved,
            gs_module=gs,
            fps=args.fps,
            box_box_detection=(not args.no_bbox_detection),
        )
        scene = handle.scene
        franka = handle.franka
        target_obj = handle.objects[0] if handle.objects else None

        table_z = scene_config.table_height + scene_config.table_size[2] / 2.0
        cam_up = scene.add_camera(
            res=(640, 480), pos=(0.55, 0.55, table_z + 0.55),
            lookat=(0.55, 0.0, table_z + 0.10), fov=45, GUI=False,
        )
        cam_side = scene.add_camera(
            res=(640, 480), pos=(0.55, -0.55, table_z + 0.30),
            lookat=(0.55, 0.0, table_z + 0.15), fov=50, GUI=False,
        )
        scene.build()

        cube = target_obj
        cube_z = resolved.placed_objects[0].position[2] if resolved.placed_objects else table_z
    else:
        cube_z = CUBE_SIZE[2] / 2.0

        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1.0 / args.fps, substeps=4),
            rigid_options=gs.options.RigidOptions(
                enable_collision=True, enable_joint_limit=True,
                box_box_detection=(not args.no_bbox_detection),
            ),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        cube = scene.add_entity(
            morph=gs.morphs.Box(size=CUBE_SIZE, pos=(0.55, 0.0, cube_z)),
            material=gs.materials.Rigid(friction=args.cube_friction),
            surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
        )
        franka = scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        )

        cam_up = scene.add_camera(
            res=(640, 480), pos=(0.55, 0.55, 0.55),
            lookat=(0.55, 0.0, 0.10), fov=45, GUI=False,
        )
        cam_side = scene.add_camera(
            res=(640, 480), pos=(0.55, -0.55, cube_z + 0.25),
            lookat=(0.55, 0.0, cube_z + 0.10), fov=50, GUI=False,
        )
        scene.build()

    motors_dof = [franka.get_joint(name).dofs_idx_local[0] for name in JOINT_NAMES]
    arm_dof = motors_dof[:7]
    finger_dof = motors_dof[7:]

    franka.set_dofs_kp(KP, motors_dof)
    franka.set_dofs_kv(KV, motors_dof)
    franka.set_dofs_force_range(FORCE_LOWER, FORCE_UPPER, motors_dof)

    n_dofs = len(JOINT_NAMES)
    end_effector = franka.get_link("hand")

    def reset_scene(cx, cy):
        franka.set_dofs_position(HOME_QPOS, motors_dof)
        franka.control_dofs_position(HOME_QPOS, motors_dof)
        franka.zero_all_dofs_velocity()
        cube.set_pos(
            torch.tensor([cx, cy, cube_z], dtype=torch.float32,
                         device=gs.device).unsqueeze(0))
        cube.set_quat(
            torch.tensor([1, 0, 0, 0], dtype=torch.float32,
                         device=gs.device).unsqueeze(0))
        cube.zero_all_dofs_velocity()
        for _ in range(args.settle_steps):
            scene.step()

    def solve_ik(pos, quat=traj_params.grasp_quat, finger_pos=traj_params.finger_open):
        qpos = to_numpy(franka.inverse_kinematics(
            link=end_effector,
            pos=np.array(pos, dtype=np.float32),
            quat=np.array(quat, dtype=np.float32),
        ))
        target = np.zeros(n_dofs, dtype=np.float32)
        target[:7] = qpos[:7]
        target[7] = finger_pos
        target[8] = finger_pos
        return target

    z_offset = (scene_config.table_height + scene_config.table_size[2] / 2.0) if use_scene else 0.0

    # ---- create LeRobot dataset ----
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

    extra_dims = 0
    extra_names = []
    if args.add_goal:
        extra_dims += 2
        extra_names += ["cube_x", "cube_y"]
    if args.add_phase:
        extra_dims += 1
        extra_names += ["phase_t_over_T"]
    obs_dim = n_dofs + extra_dims
    obs_names = MOTOR_NAMES + extra_names

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (obs_dim,),
            "names": obs_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (n_dofs,),
            "names": MOTOR_NAMES,
        },
        "observation.images.up": {
            "dtype": "image" if args.no_videos else "video",
            "shape": (3, 480, 640),
            "names": ["channel", "height", "width"],
        },
        "observation.images.side": {
            "dtype": "image" if args.no_videos else "video",
            "shape": (3, 480, 640),
            "names": ["channel", "height", "width"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        robot_type="franka",
        use_videos=(not args.no_videos),
    )
    print(f"[gen] LeRobot dataset created: {args.repo_id}")

    # ---- generate episodes ----
    rng = random.Random(args.seed)
    x_range = (args.cube_x_min, args.cube_x_max)
    y_range = (args.cube_y_min, args.cube_y_max)

    episode_points = []
    for _ in range(args.n_episodes):
        cx = rng.uniform(x_range[0], x_range[1])
        cy = rng.uniform(y_range[0], y_range[1])
        episode_points.append((cx, cy))

    out_dir = Path(args.save) / "franka_gen_pick"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[gen] task={task_spec.name}, ik_strategy={task_spec.ik_strategy}")
    print(f"[gen] cube x_range={x_range}, y_range={y_range}")
    print(f"[gen] {args.n_episodes} episodes to generate")

    frames_per_episode = None
    episode_labels = []

    for ep in range(args.n_episodes):
        cx, cy = episode_points[ep]
        reset_scene(cx, cy)

        target_pos = np.array([cx, cy, cube_z])
        traj = strategy.plan(target_pos, solve_ik, HOME_QPOS, traj_params, z_offset)

        if frames_per_episode is None:
            frames_per_episode = len(traj)
            print(f"[gen] trajectory: {frames_per_episode} frames/episode")

        initial_cube_z = cube_z
        cube_z_hist = []
        total_steps = len(traj)
        for step_idx, target in enumerate(traj):
            joints = to_numpy(franka.get_dofs_position(motors_dof)).astype(np.float32)
            parts = [joints]
            if args.add_goal:
                cube_xy = to_numpy(cube.get_pos())[:2].astype(np.float32)
                parts.append(cube_xy)
            if args.add_phase:
                t_norm = np.array([(step_idx + 1) / total_steps], dtype=np.float32)
                parts.append(t_norm)
            state = np.concatenate(parts) if len(parts) > 1 else joints
            img_up = render_cam(cam_up)
            img_side = render_cam(cam_side)

            franka.control_dofs_position(target, motors_dof)
            scene.step()

            cz = float(to_numpy(cube.get_pos())[2])
            cube_z_hist.append(cz)

            dataset.add_frame({
                "observation.state": state,
                "action": np.array(target, dtype=np.float32),
                "observation.images.up": img_up,
                "observation.images.side": img_side,
                "task": task_spec.instruction,
            })

        # ---- Evaluate success via predicate ----
        env_state = {
            "object_positions": {
                "cube": np.array([cx, cy, cube_z_hist[-1] if cube_z_hist else cube_z]),
            },
            "initial_positions": {
                "cube": np.array([cx, cy, initial_cube_z]),
            },
        }
        predicate_success = evaluate_predicate(
            task_spec.success_fn, env_state, task_spec.success_params
        )

        # Legacy heuristic (kept for backward-compatible logging)
        base_z = initial_cube_z
        cz_max = max(cube_z_hist) if cube_z_hist else base_z
        cz_end = cube_z_hist[-1] if cube_z_hist else base_z
        lifted = [z >= base_z + args.success_lift_delta for z in cube_z_hist]
        sustain_max = 0
        sustain = 0
        for ok in lifted:
            sustain = sustain + 1 if ok else 0
            sustain_max = max(sustain, sustain_max)

        heuristic_success = (
            (cz_max - base_z) >= args.success_lift_delta
            and sustain_max >= args.success_sustain_frames
            and (cz_end - base_z) >= args.success_final_delta
        )

        success = predicate_success

        episode_labels.append({
            "episode_index": ep,
            "task": task_spec.name,
            "cube_xy": [float(cx), float(cy)],
            "base_z": float(base_z),
            "cube_z_max": float(cz_max),
            "cube_z_end": float(cz_end),
            "max_lift_delta": float(cz_max - base_z),
            "end_lift_delta": float(cz_end - base_z),
            "sustain_frames": int(sustain_max),
            "success_predicate": bool(predicate_success),
            "success_heuristic": bool(heuristic_success),
            "success": bool(success),
        })

        dataset.save_episode()
        status = "OK" if success else "FAIL"
        pred_match = "✓" if predicate_success == heuristic_success else "≠"
        print(f"[gen] ep {ep+1}/{args.n_episodes} [{status}] "
              f"cube=({cx:.3f},{cy:.3f}) lift={cz_max-base_z:.4f}m "
              f"sustain={sustain_max} pred={pred_match}")

    if hasattr(dataset, "consolidate"):
        dataset.consolidate(run_compute_stats=True)
    print(f"[gen] dataset saved: {dataset.root}")

    success_ids = [e["episode_index"] for e in episode_labels if e["success"]]
    failure_ids = [e["episode_index"] for e in episode_labels if not e["success"]]
    sr = len(success_ids) / max(len(episode_labels), 1)

    summary = {
        "task": task_spec.to_dict(),
        "repo_id": args.repo_id,
        "n_episodes": args.n_episodes,
        "frames_per_episode": frames_per_episode,
        "total_frames": args.n_episodes * int(frames_per_episode or 0),
        "fps": args.fps,
        "robot": "franka_panda",
        "n_dofs": n_dofs,
        "action_space": "joint_position (rad)",
        "cube_xy_range": {"x": list(x_range), "y": list(y_range)},
        "success_episode_ids": success_ids,
        "failure_episode_ids": failure_ids,
        "success_rate": sr,
    }

    (out_dir / "gen_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "episode_labels.json").write_text(json.dumps(episode_labels, indent=2), encoding="utf-8")
    (out_dir / "success_episode_ids.json").write_text(
        json.dumps({"success_episode_ids": success_ids}, indent=2), encoding="utf-8")
    print(f"\n[gen] success_rate: {len(success_ids)}/{args.n_episodes} = {sr:.0%}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
