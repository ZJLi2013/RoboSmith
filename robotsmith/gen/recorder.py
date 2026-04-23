"""LeRobot dataset recorder — create, record frames, evaluate, and summarize."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from robotsmith.gen.franka import (
    ACTION_NAMES, STATE_NAMES,
    to_numpy, get_ee_state, compute_ee_delta,
)
from robotsmith.gen.sim_env import SimEnv, render_cam
from robotsmith.tasks import evaluate_predicate
from robotsmith.tasks.task_spec import TaskSpec


def create_dataset(
    repo_id: str,
    fps: int = 30,
    use_videos: bool = True,
):
    """Create a LeRobotDataset with standard EE-delta features."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": list(STATE_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": list(ACTION_NAMES),
        },
        "observation.images.up": {
            "dtype": "video" if use_videos else "image",
            "shape": (3, 480, 640),
            "names": ["channel", "height", "width"],
        },
        "observation.images.wrist": {
            "dtype": "video" if use_videos else "image",
            "shape": (3, 480, 640),
            "names": ["channel", "height", "width"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        robot_type="franka",
        use_videos=use_videos,
    )
    print(f"[gen] LeRobot dataset created: {repo_id}")
    return dataset


def record_episode(
    env: SimEnv,
    dataset,
    traj: list[np.ndarray],
    task_spec: TaskSpec,
    primary_entity=None,
) -> list[float]:
    """Step through a trajectory, record frames, return per-step Z of primary_entity."""
    finger_vals = to_numpy(
        env.franka.get_dofs_position(env.finger_dof)
    ).astype(np.float32)
    prev_ee_state = get_ee_state(env.end_effector, finger_vals)
    obj_z_hist: list[float] = []

    for target_qpos in traj:
        finger_vals = to_numpy(
            env.franka.get_dofs_position(env.finger_dof)
        ).astype(np.float32)
        ee_state = get_ee_state(env.end_effector, finger_vals)

        img_up = render_cam(env.cam_up)
        img_wrist = render_cam(env.cam_wrist)

        gripper_cmd = float(target_qpos[7])
        env.franka.control_dofs_position(target_qpos, env.motors_dof)
        env.scene.step()

        finger_next = to_numpy(
            env.franka.get_dofs_position(env.finger_dof)
        ).astype(np.float32)
        next_ee = get_ee_state(env.end_effector, finger_next)
        action = compute_ee_delta(prev_ee_state, next_ee, gripper_cmd)
        prev_ee_state = next_ee

        if primary_entity is not None:
            cz = float(to_numpy(primary_entity.get_pos())[2])
            obj_z_hist.append(cz)

        dataset.add_frame({
            "observation.state": ee_state,
            "action": action,
            "observation.images.up": img_up,
            "observation.images.wrist": img_wrist,
            "task": task_spec.instruction,
        })

    return obj_z_hist


def evaluate_episode(
    env: SimEnv,
    task_spec: TaskSpec,
    pick_obj_names: list[str],
    initial_positions: dict[str, np.ndarray],
) -> bool:
    """Run the task's success predicate against current entity positions."""
    env_state = {"object_positions": {}, "initial_positions": initial_positions}
    for name in pick_obj_names:
        ent = env.entity_map.get(name)
        if ent is not None:
            env_state["object_positions"][name] = to_numpy(
                ent.get_pos()
            ).copy()
    return evaluate_predicate(
        task_spec.success_fn, env_state, task_spec.success_params
    )


def save_summary(
    out_dir: Path,
    task_spec: TaskSpec,
    repo_id: str,
    n_episodes: int,
    frames_per_episode: int | None,
    fps: int,
    episode_labels: list[dict],
    workspace_xy: tuple,
):
    """Write gen_summary.json, episode_labels.json, success_episode_ids.json."""
    success_ids = [e["episode_index"] for e in episode_labels if e["success"]]
    failure_ids = [e["episode_index"] for e in episode_labels if not e["success"]]
    sr = len(success_ids) / max(len(episode_labels), 1)

    x_range = (workspace_xy[0][0], workspace_xy[1][0])
    y_range = (workspace_xy[0][1], workspace_xy[1][1])

    summary = {
        "task": task_spec.to_dict(),
        "repo_id": repo_id,
        "n_episodes": n_episodes,
        "frames_per_episode": frames_per_episode,
        "total_frames": n_episodes * int(frames_per_episode or 0),
        "fps": fps,
        "robot": "franka_panda",
        "action_space": "ee_delta_7d [delta_pos(3)+delta_axangle(3)+gripper(1)]",
        "state_space": "ee_state_8d [pos(3)+axangle(3)+gripper(2)]",
        "workspace_xy_range": {"x": list(x_range), "y": list(y_range)},
        "success_episode_ids": success_ids,
        "failure_episode_ids": failure_ids,
        "success_rate": sr,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gen_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "episode_labels.json").write_text(
        json.dumps(episode_labels, indent=2), encoding="utf-8")
    (out_dir / "success_episode_ids.json").write_text(
        json.dumps({"success_episode_ids": success_ids}, indent=2),
        encoding="utf-8")

    print(f"\n[gen] success_rate: {len(success_ids)}/{n_episodes} = {sr:.0%}")
    print(json.dumps(summary, indent=2))
