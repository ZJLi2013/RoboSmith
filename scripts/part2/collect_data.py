"""RoboSmith — open-loop data collection driven by TaskSpec.

Generates expert demonstrations via IK solver for any registered task.
Records EE-delta actions (7D) and EE state observations (8D).

Action: [delta_pos(3), delta_axangle(3), gripper(1)]
State:  [ee_pos(3), ee_axangle(3), gripper_widths(2)]

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
from scipy.spatial.transform import Rotation


# ---- constants ----
JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "joint5", "joint6", "joint7",
    "finger_joint1", "finger_joint2",
]
MOTOR_NAMES = [f"{j}.pos" for j in JOINT_NAMES]

ACTION_NAMES = [
    "delta_x", "delta_y", "delta_z",
    "delta_ax", "delta_ay", "delta_az",
    "gripper",
]
STATE_NAMES = [
    "ee_x", "ee_y", "ee_z",
    "ee_ax", "ee_ay", "ee_az",
    "gripper_left", "gripper_right",
]

HOME_QPOS = np.array([0, -0.3, 0, -2.2, 0, 2.0, 0.79, 0.04, 0.04], dtype=np.float32)
KP = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100], dtype=np.float32)
KV = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10], dtype=np.float32)
FORCE_LOWER = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100], dtype=np.float32)
FORCE_UPPER = np.array([87, 87, 87, 87, 12, 12, 12, 100, 100], dtype=np.float32)

CUBE_SIZE = (0.04, 0.04, 0.04)
BOWL_RADIUS = 0.030   # 6cm diameter (scaled bowl)
BOWL_HEIGHT = 0.03    # 3cm tall (scaled bowl)


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


def quat_to_axangle(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to axis-angle (3D compact rotvec)."""
    r = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return r.as_rotvec().astype(np.float32)


def get_ee_state(ee_link, finger_dof_vals: np.ndarray) -> np.ndarray:
    """Return 8D EE state: [pos(3), axangle(3), gripper(2)]."""
    pos = to_numpy(ee_link.get_pos()).astype(np.float32)
    quat = to_numpy(ee_link.get_quat()).astype(np.float32)
    axangle = quat_to_axangle(quat)
    gripper = finger_dof_vals[:2].astype(np.float32)
    return np.concatenate([pos, axangle, gripper])


def compute_ee_delta(prev_state: np.ndarray, curr_state: np.ndarray,
                     gripper_cmd: float) -> np.ndarray:
    """Compute 7D EE delta action: [delta_pos(3), delta_axangle(3), gripper(1)]."""
    delta_pos = curr_state[:3] - prev_state[:3]
    r_prev = Rotation.from_rotvec(prev_state[3:6])
    r_curr = Rotation.from_rotvec(curr_state[3:6])
    delta_rot = (r_curr * r_prev.inv()).as_rotvec()
    return np.concatenate([
        delta_pos.astype(np.float32),
        delta_rot.astype(np.float32),
        np.array([gripper_cmd], dtype=np.float32),
    ])


def main():
    from robotsmith.tasks import TASK_PRESETS
    from robotsmith.tasks import evaluate_predicate
    from robotsmith.tasks.task_spec import TaskSpec
    from robotsmith.grasp import TemplateGraspPlanner
    from robotsmith.motion import MotionExecutor, MotionParams
    from robotsmith.orchestration import run_skills

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
    ap.add_argument("--min-place-dist", type=float, default=0.15,
                    help="Min XY distance between pick and place for pick_and_place tasks")
    ap.add_argument("--dart-sigma", type=float, default=None,
                    help="DART noise sigma (overrides TaskSpec.dart_sigma)")
    # motion timing (passed to MotionParams)
    ap.add_argument("--settle-steps", type=int, default=30)
    ap.add_argument("--approach-steps", type=int, default=40)
    ap.add_argument("--descend-steps", type=int, default=30)
    ap.add_argument("--grasp-hold-steps", type=int, default=20)
    ap.add_argument("--lift-steps", type=int, default=30)
    ap.add_argument("--lift-hold-steps", type=int, default=15)
    ap.add_argument("--transport-steps", type=int, default=40)
    ap.add_argument("--place-descend-steps", type=int, default=25)
    ap.add_argument("--release-steps", type=int, default=15)
    ap.add_argument("--retreat-steps", type=int, default=25)
    # success detection params
    ap.add_argument("--success-lift-delta", type=float, default=0.02)
    ap.add_argument("--success-sustain-frames", type=int, default=8)
    ap.add_argument("--success-final-delta", type=float, default=0.01)
    # observation extras
    ap.add_argument("--add-phase", action="store_true")
    ap.add_argument("--add-goal", action="store_true")
    ap.add_argument("--no-bbox-detection", action="store_true")
    ap.add_argument("--no-videos", action="store_true")
    # grasp planner override
    ap.add_argument("--grasp-planner", default=None,
                    help="Grasp planner backend (default: from TaskSpec)")
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
    print(f"[task] motion_type={task_spec.motion_type}, "
          f"grasp_planner={task_spec.grasp_planner}")

    is_place_task = task_spec.motion_type == "pick_and_place"
    is_stack_task = task_spec.is_stack
    is_bowl_task = task_spec.name == "pick_bowl"
    N_BLOCKS = task_spec.n_stack if is_stack_task else 1
    BLOCK_COLORS = [(1.0, 0.3, 0.3, 1.0), (0.3, 0.8, 0.3, 1.0), (0.3, 0.3, 1.0, 1.0)]
    BLOCK_NAMES = ["block_red", "block_green", "block_blue"]
    pick_obj_name = task_spec.skills[0].target if task_spec.skills else "cube"

    # ---- MotionParams from CLI ----
    motion_params = MotionParams(
        approach_steps=args.approach_steps,
        descend_steps=args.descend_steps,
        grasp_hold_steps=args.grasp_hold_steps,
        lift_steps=args.lift_steps,
        lift_hold_steps=args.lift_hold_steps,
        transport_steps=args.transport_steps,
        place_descend_steps=args.place_descend_steps,
        release_steps=args.release_steps,
        retreat_steps=args.retreat_steps,
    )
    executor = MotionExecutor()

    # ---- Genesis setup ----
    ensure_display()
    import genesis as gs
    import torch

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
        cam_wrist = scene.add_camera(
            res=(640, 480),
            pos=(0.05, 0.0, -0.08),
            lookat=(0.0, 0.0, 0.10),
            fov=65, GUI=False,
        )
        scene.build()

        cube = target_obj
        target_marker = None
        cube_z = resolved.placed_objects[0].position[2] if resolved.placed_objects else table_z
    else:
        if is_bowl_task:
            obj_z = BOWL_HEIGHT / 2.0
        else:
            obj_z = CUBE_SIZE[2] / 2.0
        cube_z = obj_z

        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1.0 / args.fps, substeps=4),
            rigid_options=gs.options.RigidOptions(
                enable_collision=True, enable_joint_limit=True,
                box_box_detection=(not args.no_bbox_detection),
            ),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        blocks = []
        if is_stack_task:
            for bi in range(N_BLOCKS):
                b = scene.add_entity(
                    morph=gs.morphs.Box(size=CUBE_SIZE,
                                        pos=(0.55 + 0.08 * bi, 0.0, cube_z)),
                    material=gs.materials.Rigid(friction=args.cube_friction),
                    surface=gs.surfaces.Default(color=BLOCK_COLORS[bi]),
                )
                blocks.append(b)
            cube = blocks[0]
        elif is_bowl_task:
            cube = scene.add_entity(
                morph=gs.morphs.Cylinder(
                    radius=BOWL_RADIUS, height=BOWL_HEIGHT,
                    pos=(0.55, 0.0, cube_z),
                ),
                material=gs.materials.Rigid(friction=args.cube_friction),
                surface=gs.surfaces.Default(color=(0.6, 0.4, 0.2, 1.0)),
            )
        else:
            cube = scene.add_entity(
                morph=gs.morphs.Box(size=CUBE_SIZE, pos=(0.55, 0.0, cube_z)),
                material=gs.materials.Rigid(friction=args.cube_friction),
                surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
            )
        target_marker = None
        if is_place_task and not is_stack_task:
            TARGET_SIZE = (0.06, 0.06, 0.005)
            target_marker = scene.add_entity(
                morph=gs.morphs.Box(size=TARGET_SIZE, pos=(0.55, 0.2, TARGET_SIZE[2] / 2)),
                material=gs.materials.Rigid(friction=0.5),
                surface=gs.surfaces.Default(color=(0.3, 0.8, 0.3, 0.8)),
            )
        franka = scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        )

        cam_up = scene.add_camera(
            res=(640, 480), pos=(0.55, 0.55, 0.55),
            lookat=(0.55, 0.0, 0.10), fov=45, GUI=False,
        )
        cam_wrist = scene.add_camera(
            res=(640, 480),
            pos=(0.05, 0.0, -0.08),
            lookat=(0.0, 0.0, 0.10),
            fov=65, GUI=False,
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

    from genesis.utils.geom import pos_lookat_up_to_T
    wrist_offset_T = pos_lookat_up_to_T(
        torch.tensor([0.05, 0.0, -0.08], dtype=gs.tc_float, device=gs.device),
        torch.tensor([0.0, 0.0, 0.10], dtype=gs.tc_float, device=gs.device),
        torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device),
    )
    try:
        cam_wrist.attach(rigid_link=end_effector, offset_T=wrist_offset_T)
    except TypeError:
        cam_wrist.attach(end_effector, wrist_offset_T)
    print("[cam] wrist camera attached to franka hand link")

    def reset_scene(cx, cy, place_xy=None, block_positions=None):
        franka.set_dofs_position(HOME_QPOS, motors_dof)
        franka.control_dofs_position(HOME_QPOS, motors_dof)
        franka.zero_all_dofs_velocity()
        if is_stack_task and block_positions is not None:
            for bi, bpos in enumerate(block_positions):
                blocks[bi].set_pos(
                    torch.tensor([bpos[0], bpos[1], cube_z], dtype=torch.float32,
                                 device=gs.device).unsqueeze(0))
                blocks[bi].set_quat(
                    torch.tensor([1, 0, 0, 0], dtype=torch.float32,
                                 device=gs.device).unsqueeze(0))
                blocks[bi].zero_all_dofs_velocity()
        else:
            cube.set_pos(
                torch.tensor([cx, cy, cube_z], dtype=torch.float32,
                             device=gs.device).unsqueeze(0))
            cube.set_quat(
                torch.tensor([1, 0, 0, 0], dtype=torch.float32,
                             device=gs.device).unsqueeze(0))
            cube.zero_all_dofs_velocity()
        if target_marker is not None and place_xy is not None:
            target_marker.set_pos(
                torch.tensor([place_xy[0], place_xy[1], 0.0025], dtype=torch.float32,
                             device=gs.device).unsqueeze(0))
        for _ in range(args.settle_steps):
            scene.step()

    # ---- GraspPlanner setup ----
    z_offset = (scene_config.table_height + scene_config.table_size[2] / 2.0) if use_scene else 0.0
    planner = TemplateGraspPlanner(z_offset=z_offset)

    # IK solver uses the planner's grasp_quat as default for backward compat
    default_template = planner._templates.get("block") or next(
        iter(planner._templates.values()))
    _default_quat = default_template.ee_quat
    _default_finger_open = default_template.finger_open

    def solve_ik(pos, quat=_default_quat, finger_pos=_default_finger_open):
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

    # ---- create LeRobot dataset ----
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

    state_dim = 8
    action_dim = 7

    extra_dims = 0
    extra_names = []
    if args.add_goal:
        extra_dims += 2
        extra_names += ["cube_x", "cube_y"]
    if args.add_phase:
        extra_dims += 1
        extra_names += ["phase_t_over_T"]
    obs_dim = state_dim + extra_dims
    obs_names = STATE_NAMES + extra_names

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (obs_dim,),
            "names": obs_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ACTION_NAMES,
        },
        "observation.images.up": {
            "dtype": "image" if args.no_videos else "video",
            "shape": (3, 480, 640),
            "names": ["channel", "height", "width"],
        },
        "observation.images.wrist": {
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

    def sample_place_target(cx, cy, min_dist):
        for _ in range(100):
            px = rng.uniform(x_range[0], x_range[1])
            py = rng.uniform(y_range[0], y_range[1])
            if np.hypot(px - cx, py - cy) >= min_dist:
                return (px, py)
        return (cx + min_dist, cy)

    def sample_spaced_positions(n, min_dist=0.10):
        pts = []
        for _ in range(n):
            for _try in range(200):
                x = rng.uniform(x_range[0], x_range[1])
                y = rng.uniform(y_range[0], y_range[1])
                if all(np.hypot(x - px, y - py) >= min_dist for px, py in pts):
                    pts.append((x, y))
                    break
            else:
                pts.append((x, y))
        return pts

    episode_points = []
    for _ in range(args.n_episodes):
        if is_stack_task:
            block_xys = sample_spaced_positions(N_BLOCKS, min_dist=0.10)
            sx, sy = sample_place_target(
                block_xys[0][0], block_xys[0][1], args.min_place_dist)
            episode_points.append({
                "block_xys": block_xys,
                "stack_xy": (sx, sy),
            })
        elif is_place_task:
            cx = rng.uniform(x_range[0], x_range[1])
            cy = rng.uniform(y_range[0], y_range[1])
            px, py = sample_place_target(cx, cy, args.min_place_dist)
            episode_points.append((cx, cy, px, py))
        else:
            cx = rng.uniform(x_range[0], x_range[1])
            cy = rng.uniform(y_range[0], y_range[1])
            episode_points.append((cx, cy, None, None))

    out_dir = Path(args.save) / f"franka_gen_{task_spec.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[gen] task={task_spec.name}, motion_type={task_spec.motion_type}")
    print(f"[gen] cube x_range={x_range}, y_range={y_range}")
    if is_place_task:
        print(f"[gen] min_place_dist={args.min_place_dist}")
    if is_stack_task:
        print(f"[gen] N_BLOCKS={N_BLOCKS}")
    print(f"[gen] {args.n_episodes} episodes to generate")

    frames_per_episode = None
    episode_labels = []

    for ep in range(args.n_episodes):
        if is_stack_task:
            ep_data = episode_points[ep]
            block_xys = ep_data["block_xys"]
            sx, sy = ep_data["stack_xy"]
            cx, cy = block_xys[0]
            px, py = sx, sy
            block_positions_3d = [[bx, by, cube_z] for bx, by in block_xys]
            reset_scene(cx, cy, block_positions=block_positions_3d)
        else:
            cx, cy, px, py = episode_points[ep]
            place_xy = (px, py) if px is not None else None
            reset_scene(cx, cy, place_xy=place_xy)

        # Build scene_state positions for run_skills()
        positions: dict[str, np.ndarray] = {}
        if is_stack_task:
            for bi, (bx, by) in enumerate(block_xys):
                positions[BLOCK_NAMES[bi]] = np.array([bx, by, cube_z])
            positions["stack_center"] = np.array([sx, sy, cube_z])
        else:
            positions[pick_obj_name] = np.array([cx, cy, cube_z])
            if px is not None:
                positions["target"] = np.array([px, py, cube_z])

        scene_state = {"home_qpos": HOME_QPOS, "positions": positions}
        traj = run_skills(
            task_spec.skills, planner, executor, solve_ik,
            scene_state, motion_params,
        )

        if frames_per_episode is None:
            frames_per_episode = len(traj)
            print(f"[gen] trajectory: {frames_per_episode} frames/episode")

        initial_cube_z = cube_z
        cube_z_hist = []
        total_steps = len(traj)

        finger_vals = to_numpy(franka.get_dofs_position(finger_dof)).astype(np.float32)
        prev_ee_state = get_ee_state(end_effector, finger_vals)

        for step_idx, target in enumerate(traj):
            finger_vals = to_numpy(franka.get_dofs_position(finger_dof)).astype(np.float32)
            ee_state = get_ee_state(end_effector, finger_vals)
            parts = [ee_state]
            if args.add_goal:
                cube_xy = to_numpy(cube.get_pos())[:2].astype(np.float32)
                parts.append(cube_xy)
            if args.add_phase:
                t_norm = np.array([(step_idx + 1) / total_steps], dtype=np.float32)
                parts.append(t_norm)
            state = np.concatenate(parts) if len(parts) > 1 else ee_state
            img_up = render_cam(cam_up)
            img_wrist = render_cam(cam_wrist)

            gripper_cmd = float(target[7])
            franka.control_dofs_position(target, motors_dof)
            scene.step()

            finger_vals_next = to_numpy(franka.get_dofs_position(finger_dof)).astype(np.float32)
            next_ee_state = get_ee_state(end_effector, finger_vals_next)
            action = compute_ee_delta(prev_ee_state, next_ee_state, gripper_cmd)
            prev_ee_state = next_ee_state

            cz = float(to_numpy(cube.get_pos())[2])
            cube_z_hist.append(cz)

            dataset.add_frame({
                "observation.state": state,
                "action": action,
                "observation.images.up": img_up,
                "observation.images.wrist": img_wrist,
                "task": task_spec.instruction,
            })

        # ---- Evaluate success via predicate ----
        if is_stack_task:
            env_state = {"object_positions": {}, "initial_positions": {}}
            for bi in range(N_BLOCKS):
                bname = BLOCK_NAMES[bi]
                bpos = to_numpy(blocks[bi].get_pos())
                env_state["object_positions"][bname] = bpos.copy()
                env_state["initial_positions"][bname] = np.array(
                    [block_xys[bi][0], block_xys[bi][1], cube_z])
        else:
            cube_final_pos = to_numpy(cube.get_pos())
            env_state = {
                "object_positions": {pick_obj_name: cube_final_pos.copy()},
                "initial_positions": {pick_obj_name: np.array([cx, cy, initial_cube_z])},
            }
            if is_place_task and px is not None:
                env_state["object_positions"]["target"] = np.array([px, py, 0.0025])
        predicate_success = evaluate_predicate(
            task_spec.success_fn, env_state, task_spec.success_params
        )

        base_z = initial_cube_z
        cz_max = max(cube_z_hist) if cube_z_hist else base_z
        cz_end = cube_z_hist[-1] if cube_z_hist else base_z

        success = predicate_success

        label = {
            "episode_index": ep,
            "task": task_spec.name,
            "success_predicate": bool(predicate_success),
            "success": bool(success),
        }
        if is_stack_task:
            block_final_zs = [float(to_numpy(blocks[bi].get_pos())[2])
                              for bi in range(N_BLOCKS)]
            label["stack_xy"] = [float(sx), float(sy)]
            label["block_xys"] = [[float(bx), float(by)] for bx, by in block_xys]
            label["block_final_zs"] = block_final_zs
        elif is_place_task:
            cube_final_pos = to_numpy(cube.get_pos())
            label["cube_xy"] = [float(cx), float(cy)]
            label["place_xy"] = [float(px), float(py)]
            label["place_xy_error"] = float(
                np.linalg.norm(cube_final_pos[:2] - np.array([px, py])))
        else:
            label["cube_xy"] = [float(cx), float(cy)]
            label["base_z"] = float(base_z)
            label["cube_z_max"] = float(cz_max)
            label["cube_z_end"] = float(cz_end)
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
            label["sustain_frames"] = int(sustain_max)
            label["success_heuristic"] = bool(heuristic_success)
        episode_labels.append(label)

        dataset.save_episode()
        status = "OK" if success else "FAIL"
        if is_stack_task:
            zs = label["block_final_zs"]
            print(f"[gen] ep {ep+1}/{args.n_episodes} [{status}] "
                  f"stack=({sx:.3f},{sy:.3f}) "
                  f"block_zs=[{', '.join(f'{z:.3f}' for z in zs)}]")
        elif is_place_task:
            print(f"[gen] ep {ep+1}/{args.n_episodes} [{status}] "
                  f"pick=({cx:.3f},{cy:.3f}) place=({px:.3f},{py:.3f}) "
                  f"xy_err={label['place_xy_error']:.4f}m")
        else:
            pred_match = "✓" if predicate_success == label.get("success_heuristic", predicate_success) else "≠"
            print(f"[gen] ep {ep+1}/{args.n_episodes} [{status}] "
                  f"cube=({cx:.3f},{cy:.3f}) lift={cz_max-base_z:.4f}m "
                  f"sustain={label.get('sustain_frames', 0)} pred={pred_match}")

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
        "action_space": "ee_delta_7d [delta_pos(3)+delta_axangle(3)+gripper(1)]",
        "state_space": "ee_state_8d [pos(3)+axangle(3)+gripper(2)]",
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
