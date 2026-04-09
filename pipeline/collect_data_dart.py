"""
RoboSmith pipeline — Stage 1 closed-loop DART data collection.

P6: Closed-loop IK replanning data generation for Franka Panda pick-cube.

Key differences from P3/P3.2 (open-loop DART):
  1. NO pre-planned trajectory -- only task-space waypoints
  2. Per-step IK: every step calls solve_ik() from CURRENT joint config
  3. Task-space interpolation: lerp in Cartesian space, not joint space
  4. Action-space perturbation: noise added to the EXECUTED command (not robot state),
     while the CLEAN IK solution is recorded as the label. Robot naturally drifts
     off-trajectory, and subsequent IK re-solves provide recovery actions.
  5. 100% DART episodes: every frame is naturally a recovery frame
  6. Phase progression based on step count (same timing as baseline)

Usage:
  python collect_data_dart.py \
    --n-episodes 200 --perturb-sigma 0.005 \
    --repo-id local/franka-closed-loop-pick \
    --save /output
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np


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
GRASP_QUAT = np.array([0, 1, 0, 0], dtype=np.float32)
FINGER_OPEN = 0.04
FINGER_CLOSED = 0.01
N_ARM_JOINTS = 7


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
    ap = argparse.ArgumentParser(description="P6: Closed-loop IK data for Franka")
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--perturb-sigma", type=float, default=0.005,
                    help="Per-step joint noise sigma (rad). 0 = no perturbation (pure closed-loop)")
    ap.add_argument("--perturb-phase-end", type=int, default=70,
                    help="Only perturb within first N frames (approach+descend)")
    ap.add_argument("--repo-id", default="local/franka-closed-loop-pick")
    ap.add_argument("--save", default="/output")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cube-x-min", type=float, default=0.4)
    ap.add_argument("--cube-x-max", type=float, default=0.7)
    ap.add_argument("--cube-y-min", type=float, default=-0.2)
    ap.add_argument("--cube-y-max", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cube-friction", type=float, default=1.5)
    ap.add_argument("--hover-z", type=float, default=0.25)
    ap.add_argument("--grasp-z", type=float, default=0.135)
    ap.add_argument("--lift-z", type=float, default=0.30)
    ap.add_argument("--settle-steps", type=int, default=30)
    ap.add_argument("--approach-steps", type=int, default=40)
    ap.add_argument("--descend-steps", type=int, default=30)
    ap.add_argument("--grasp-hold-steps", type=int, default=20)
    ap.add_argument("--lift-steps", type=int, default=30)
    ap.add_argument("--lift-hold-steps", type=int, default=15)
    ap.add_argument("--success-lift-delta", type=float, default=0.02)
    ap.add_argument("--success-sustain-frames", type=int, default=8)
    ap.add_argument("--success-final-delta", type=float, default=0.01)
    ap.add_argument("--task", default="Pick up the cube.")
    args = ap.parse_args()

    ensure_display()
    import genesis as gs
    import torch

    gs.init(backend=(gs.cpu if args.cpu else gs.gpu), logging_level="warning")

    cube_z = CUBE_SIZE[2] / 2.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / args.fps, substeps=4),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True, enable_joint_limit=True, box_box_detection=True,
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
    n_dofs = len(JOINT_NAMES)
    end_effector = franka.get_link("hand")

    franka.set_dofs_kp(KP, motors_dof)
    franka.set_dofs_kv(KV, motors_dof)
    franka.set_dofs_force_range(FORCE_LOWER, FORCE_UPPER, motors_dof)

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

    def solve_ik_from_current(target_pos, target_quat=GRASP_QUAT, finger_pos=FINGER_OPEN):
        """Solve IK from the robot's CURRENT joint configuration."""
        qpos = to_numpy(franka.inverse_kinematics(
            link=end_effector,
            pos=np.array(target_pos, dtype=np.float32),
            quat=np.array(target_quat, dtype=np.float32),
        ))
        target = np.zeros(n_dofs, dtype=np.float32)
        target[:7] = qpos[:7]
        target[7] = finger_pos
        target[8] = finger_pos
        return target

    def get_ee_pos():
        """Get current end-effector position in world frame."""
        pos = to_numpy(end_effector.get_pos())
        return np.array(pos, dtype=np.float64)

    def build_phase_schedule(cx, cy):
        """Build (target_pos, finger_pos, n_steps) for each phase."""
        return [
            (np.array([cx, cy, args.hover_z], dtype=np.float64), FINGER_OPEN, args.approach_steps),
            (np.array([cx, cy, args.grasp_z], dtype=np.float64), FINGER_OPEN, args.descend_steps),
            (np.array([cx, cy, args.grasp_z], dtype=np.float64), FINGER_CLOSED, args.grasp_hold_steps),
            (np.array([cx, cy, args.lift_z], dtype=np.float64), FINGER_CLOSED, args.lift_steps),
            (np.array([cx, cy, args.lift_z], dtype=np.float64), FINGER_CLOSED, args.lift_hold_steps),
        ]

    # ---- create LeRobot dataset ----
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (n_dofs,),
            "names": MOTOR_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (n_dofs,),
            "names": MOTOR_NAMES,
        },
        "observation.images.up": {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["channel", "height", "width"],
        },
        "observation.images.side": {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["channel", "height", "width"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        robot_type="franka",
        use_videos=True,
    )

    # ---- generate episodes ----
    rng = random.Random(args.seed)
    np_rng = np.random.RandomState(args.seed)

    episode_points = []
    for _ in range(args.n_episodes):
        cx = rng.uniform(args.cube_x_min, args.cube_x_max)
        cy = rng.uniform(args.cube_y_min, args.cube_y_max)
        episode_points.append((cx, cy))

    repo_name = args.repo_id.split("/")[-1]
    out_dir = Path(args.save) / repo_name
    out_dir.mkdir(parents=True, exist_ok=True)

    total_steps_per_ep = (args.approach_steps + args.descend_steps +
                          args.grasp_hold_steps + args.lift_steps +
                          args.lift_hold_steps)

    print(f"[closed-loop] {args.n_episodes} episodes, {total_steps_per_ep} frames/ep")
    print(f"[closed-loop] perturb σ={args.perturb_sigma} rad, "
          f"phase_end={args.perturb_phase_end}")

    episode_labels = []
    perturb_count_total = 0

    for ep in range(args.n_episodes):
        cx, cy = episode_points[ep]
        reset_scene(cx, cy)

        phases = build_phase_schedule(cx, cy)
        cube_z_hist = []
        n_perturbed = 0
        global_step = 0

        for phase_idx, (wp_pos, wp_finger, n_steps) in enumerate(phases):
            phase_start_ee = get_ee_pos()

            for sub_step in range(n_steps):
                should_perturb = (
                    args.perturb_sigma > 0
                    and global_step < args.perturb_phase_end
                )

                alpha = (sub_step + 1) / max(n_steps, 1)
                target_pos = phase_start_ee * (1.0 - alpha) + wp_pos * alpha

                ik_target = solve_ik_from_current(
                    target_pos, finger_pos=wp_finger
                )

                if should_perturb:
                    action_noise = np_rng.normal(
                        0, args.perturb_sigma, size=N_ARM_JOINTS
                    ).astype(np.float32)
                    noisy_command = ik_target.copy()
                    noisy_command[:N_ARM_JOINTS] += action_noise
                    franka.control_dofs_position(noisy_command, motors_dof)
                    n_perturbed += 1
                else:
                    franka.control_dofs_position(ik_target, motors_dof)

                scene.step()

                state = to_numpy(
                    franka.get_dofs_position(motors_dof)
                ).astype(np.float32)
                img_up = render_cam(cam_up)
                img_side = render_cam(cam_side)
                cz = float(to_numpy(cube.get_pos())[2])
                cube_z_hist.append(cz)

                dataset.add_frame({
                    "observation.state": state,
                    "action": np.array(ik_target, dtype=np.float32),
                    "observation.images.up": img_up,
                    "observation.images.side": img_side,
                    "task": args.task,
                })

                global_step += 1

        perturb_count_total += n_perturbed

        base_z = cube_z
        cz_max = max(cube_z_hist) if cube_z_hist else base_z
        cz_end = cube_z_hist[-1] if cube_z_hist else base_z
        lifted = [z >= base_z + args.success_lift_delta for z in cube_z_hist]
        sustain = sustain_max = 0
        for ok in lifted:
            sustain = sustain + 1 if ok else 0
            sustain_max = max(sustain, sustain_max)

        success = (
            (cz_max - base_z) >= args.success_lift_delta
            and sustain_max >= args.success_sustain_frames
            and (cz_end - base_z) >= args.success_final_delta
        )

        episode_labels.append({
            "episode_index": ep,
            "cube_xy": [float(cx), float(cy)],
            "base_z": float(base_z),
            "cube_z_max": float(cz_max),
            "cube_z_end": float(cz_end),
            "max_lift_delta": float(cz_max - base_z),
            "end_lift_delta": float(cz_end - base_z),
            "sustain_frames": int(sustain_max),
            "success": bool(success),
            "n_perturbed": n_perturbed,
        })

        dataset.save_episode()
        status = "OK" if success else "FAIL"
        print(f"[closed-loop] ep {ep+1}/{args.n_episodes} [{status}] "
              f"cube=({cx:.3f},{cy:.3f}) lift={cz_max-base_z:.4f}m "
              f"perturbed={n_perturbed}")

    if hasattr(dataset, "consolidate"):
        dataset.consolidate(run_compute_stats=True)
    print(f"[closed-loop] dataset saved: {dataset.root}")

    success_ids = [e["episode_index"] for e in episode_labels if e["success"]]
    sr = len(success_ids) / max(len(episode_labels), 1)

    summary = {
        "repo_id": args.repo_id,
        "n_episodes": args.n_episodes,
        "perturb_sigma": args.perturb_sigma,
        "perturb_phase_end": args.perturb_phase_end,
        "frames_per_episode": total_steps_per_ep,
        "total_frames": args.n_episodes * total_steps_per_ep,
        "success_rate": sr,
        "success_episode_ids": success_ids,
        "total_perturbations": perturb_count_total,
        "avg_perturbations_per_ep": perturb_count_total / max(args.n_episodes, 1),
    }

    (out_dir / "gen_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "episode_labels.json").write_text(
        json.dumps(episode_labels, indent=2), encoding="utf-8")
    (out_dir / "success_episode_ids.json").write_text(
        json.dumps({"success_episode_ids": success_ids}, indent=2), encoding="utf-8")

    print(f"\n[closed-loop] success: {len(success_ids)}/{args.n_episodes} = {sr:.0%}")
    print(f"  total perturbations: {perturb_count_total}")
    print(f"  avg perturbations/ep: {perturb_count_total / max(args.n_episodes, 1):.1f}")


if __name__ == "__main__":
    main()
