"""Asset-level pick rollout validation workflow."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil

import numpy as np

from robotsmith.cap.adapters import (
    to_legacy_scene_config,
    to_legacy_skills,
    to_legacy_task_spec,
)
from robotsmith.cap.intents import intent_sequence, pick
from robotsmith.cap.predicates import object_lifted
from robotsmith.cap.scene_api import layout, obj, region, scene
from robotsmith.cap.task_api import task
from robotsmith.sim.franka import HOME_QPOS, to_numpy_f32
from robotsmith.rollout.episode_layout import (
    build_episode_positions,
    derive_skill_targets,
)
from robotsmith.skills import run_skills
from robotsmith.rollout.video_export import copy_episode_videos
from robotsmith.scenes.config import SceneConfig, TASK_WORKSPACE_XY
from robotsmith.tasks import evaluate_success
from robotsmith.tasks.task_spec import TaskSpec

FPS = 30
EPISODE_DIAGNOSTIC_FIELDS = (
    "episode_index",
    "success",
    "frame_count",
    "object_scale",
    "selected_candidate_index",
    "selected_policy_bucket",
    "selected_policy_bucket_rank",
    "requested_policy_bucket",
    "requested_policy_bucket_missing",
    "selected_approach_bin",
    "selected_orientation_bin",
    "hard_ok",
    "no_feasible_grasp",
    "p0_bucket_hard_ok_counts",
    "score_exec",
)


@dataclass
class AssetPickValidationConfig:
    """Configuration for asset-level pick rollout validation."""

    assets: list[str]
    output_dir: Path
    n_episodes: int = 3
    seed: int = 42
    no_videos: bool = False
    clean: bool = False
    settle_steps: int = 30
    grasp_planner: str = "auto"
    grasp_policy_bucket_override: str | None = None


@dataclass
class AssetPickEpisodePlan:
    """Prepared trajectory and state for one asset-pick validation episode."""

    initial_positions: dict[str, np.ndarray]
    scene_state: dict
    trajectory: list[np.ndarray]


def _asset_category(asset_name: str) -> str:
    return asset_name.split("_", 1)[0]


def _object_pose_label(env, name: str) -> dict:
    quat = env.object_quats.get(name)
    if quat is None:
        return {}
    return {"object_quat_wxyz": [float(v) for v in quat]}


def _make_task(asset_name: str) -> TaskSpec:
    category = _asset_category(asset_name)
    cap_scene = _make_cap_scene(asset_name)
    cap_task = task(
        name=f"pick_{asset_name}",
        scene=cap_scene,
        instruction=f"Pick up {asset_name}",
        success=object_lifted("object", z_margin=0.02),
    )
    intents = intent_sequence([pick("object", category=category)])
    legacy_task = to_legacy_task_spec(cap_task)
    legacy_task.skills = to_legacy_skills(intents, cap_task)
    return legacy_task


def _make_scene(asset_name: str) -> SceneConfig:
    return to_legacy_scene_config(_make_cap_scene(asset_name))


def _make_cap_scene(asset_name: str):
    lower, upper = TASK_WORKSPACE_XY
    return scene(
        f"pick_{asset_name}",
        layout=layout(
            regions=[
                region(
                    "task_workspace",
                    ((lower[0], lower[1]), (upper[0], upper[1])),
                ),
            ],
        ),
        objects=[
            obj("object", asset=asset_name, region="task_workspace"),
        ],
    )


def _pick_trace(scene_state: dict) -> dict:
    for trace in scene_state.get("skill_traces", []):
        if trace.get("skill") == "pick" and trace.get("target") == "object":
            return trace
    return {}


def _prepare_asset_pick_episode(
    env,
    task_spec: TaskSpec,
    *,
    rng: random.Random,
    executor,
    motion_params,
    settle_steps: int,
) -> AssetPickEpisodePlan:
    pick_names, place_names = derive_skill_targets(task_spec)
    positions = build_episode_positions(
        env,
        rng,
        pick_names,
        place_names,
        task_spec=task_spec,
    )
    obj_xy_map = {
        name: (pos[0], pos[1])
        for name, pos in positions.items()
        if name in env.entity_map
    }
    settled_positions = env.reset(obj_xy_map, settle_steps=settle_steps)
    for name, pos in settled_positions.items():
        positions[name] = pos.copy()

    initial_positions = {
        name: pos.copy()
        for name, pos in positions.items()
        if name in env.entity_map
    }
    scene_state = {
        "home_qpos": HOME_QPOS.copy(),
        "positions": positions,
        "object_heights": env.object_heights,
        "assets": env.asset_map,
        "object_quats": env.object_quats,
        "object_scales": env.object_scales,
        "table_surface_z": env.table_surface_z,
        "skill_traces": [],
    }
    trajectory = run_skills(
        task_spec.skills,
        env.planner,
        executor,
        env.solve_ik,
        scene_state,
        motion_params,
    )
    return AssetPickEpisodePlan(
        initial_positions=initial_positions,
        scene_state=scene_state,
        trajectory=trajectory,
    )


def _evaluate_asset_pick_task(
    env,
    task_spec: TaskSpec,
    *,
    initial_positions: dict[str, np.ndarray],
) -> bool:
    object_positions = {
        name: to_numpy_f32(ent.get_pos()).copy()
        for name, ent in env.entity_map.items()
    }
    env_state = {
        "object_positions": object_positions,
        "initial_positions": initial_positions,
    }
    return evaluate_success(task_spec.success, env_state)


def run_asset_pick(config: AssetPickValidationConfig, asset_name: str) -> dict:
    """Run rollout validation for one asset and return its per-asset summary.

    This builds a single-object pick scene, records episodes, evaluates success,
    writes dataset artifacts/summaries under ``config.output_dir``, and returns
    the diagnostics consumed by metadata-guided rollout orchestration.
    """

    from robotsmith.sim.franka import to_numpy
    from robotsmith.sim.recorder import (
        create_dataset,
        record_episode,
        save_summary,
    )
    from robotsmith.sim.sim_env import SimEnv
    from robotsmith.execution import MotionExecutor
    from robotsmith.motion import MotionParams

    task_spec = _make_task(asset_name)
    scene_config = _make_scene(asset_name)
    repo_id = f"local/robotsmith-pick-{asset_name}"

    env = SimEnv.build(
        scene_config,
        seed=config.seed,
        fps=FPS,
        grasp_planner=config.grasp_planner,
    )
    if config.grasp_policy_bucket_override:
        env.asset_map["object"].metadata.grasp_policy_bucket = config.grasp_policy_bucket_override
    object_scale = env.object_scales.get("object", 1.0)
    print(f"======== START {asset_name} ========")
    print(
        "[scene] "
        "pose=upright "
        f"metric_scale={object_scale} "
        f"grasp_planner={config.grasp_planner} "
        f"settle_steps={config.settle_steps}"
    )
    executor = MotionExecutor()
    motion_params = MotionParams()
    cache_root = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
    if cache_root.exists():
        shutil.rmtree(cache_root)
    camera_names = ("up",)
    dataset = create_dataset(
        repo_id=repo_id,
        fps=FPS,
        use_videos=not config.no_videos,
        camera_names=camera_names,
    )

    rng = random.Random(config.seed)
    out_dir = config.output_dir / asset_name / f"franka_gen_{task_spec.name}"
    episode_labels: list[dict] = []
    episode_diagnostics: list[dict] = []
    episode_frame_counts: dict[int, int] = {}
    frames_per_episode = None
    primary_entity = env.entity_map.get("object")

    for ep in range(config.n_episodes):
        episode_plan = _prepare_asset_pick_episode(
            env,
            task_spec,
            rng=rng,
            executor=executor,
            motion_params=motion_params,
            settle_steps=config.settle_steps,
        )
        pose_label = _object_pose_label(env, "object")

        traj = episode_plan.trajectory
        scene_state = episode_plan.scene_state
        pick_trace = _pick_trace(scene_state)
        if frames_per_episode is None:
            frames_per_episode = len(traj)
            print(f"[rollout] trajectory: {frames_per_episode} frames/episode")
        episode_frame_counts[ep] = len(traj)

        record_episode(
            env,
            dataset,
            traj,
            task_spec,
            primary_entity=primary_entity,
            camera_names=camera_names,
        )
        success = _evaluate_asset_pick_task(
            env,
            task_spec,
            initial_positions=episode_plan.initial_positions,
        )
        label = {
            "episode_index": ep,
            "task": task_spec.name,
            "asset": asset_name,
            "success": bool(success),
            "frame_count": len(traj),
            "object_scale": pick_trace.get("object_scale", object_scale),
            "selected_candidate_index": pick_trace.get("selected_candidate_index"),
            "selected_policy_bucket": pick_trace.get("selected_policy_bucket"),
            "selected_policy_bucket_rank": pick_trace.get(
                "selected_policy_bucket_rank",
            ),
            "requested_policy_bucket": pick_trace.get("requested_policy_bucket"),
            "requested_policy_bucket_missing": pick_trace.get(
                "requested_policy_bucket_missing",
            ),
            "selected_approach_bin": pick_trace.get("selected_approach_bin"),
            "selected_orientation_bin": pick_trace.get("selected_orientation_bin"),
            "hard_ok": pick_trace.get("hard_ok"),
            "no_feasible_grasp": pick_trace.get("no_feasible_grasp"),
            "p0_bucket_hard_ok_counts": pick_trace.get("p0_bucket_hard_ok_counts"),
            "score_exec": pick_trace.get("score_exec"),
        }
        label.update(pose_label)
        ent = env.entity_map.get("object")
        if ent:
            fpos = to_numpy(ent.get_pos())
            label["object_final_pos"] = [float(v) for v in fpos]
        episode_labels.append(label)
        episode_diagnostics.append({
            key: label[key] for key in EPISODE_DIAGNOSTIC_FIELDS
        })
        dataset.save_episode()

        status = "OK" if success else "FAIL"
        print(f"[rollout] {asset_name} ep {ep + 1}/{config.n_episodes} [{status}]")

    if hasattr(dataset, "consolidate"):
        dataset.consolidate(run_compute_stats=True)
    dataset_root = Path(dataset.root)
    save_summary(
        out_dir,
        task_spec,
        repo_id,
        config.n_episodes,
        frames_per_episode,
        FPS,
        episode_labels,
        workspace_xy=((0.40, -0.20), (0.70, 0.20)),
    )

    success_ids = [e["episode_index"] for e in episode_labels if e["success"]]
    copied = []
    if not config.no_videos:
        copied.extend(copy_episode_videos(
            dataset_root,
            [e["episode_index"] for e in episode_labels],
            config.output_dir / asset_name / "videos",
            asset_name,
            frames_per_episode,
            episode_frame_counts,
            FPS,
        ))
        print(f"[video] copied {len(copied)} episode videos for {asset_name}")

    return {
        "asset": asset_name,
        "success_episode_ids": success_ids,
        "n_episodes": config.n_episodes,
        "success_rate": len(success_ids) / max(config.n_episodes, 1),
        "videos": copied,
        "dataset_root": str(dataset_root),
        "episode_diagnostics": episode_diagnostics,
    }


def run_asset_pick_validation(config: AssetPickValidationConfig) -> dict:
    """Run pick validation for all configured assets and write a summary JSON."""

    from robotsmith.sim.sim_env import ensure_display

    if config.clean and config.output_dir.exists():
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    ensure_display()

    results = []
    for asset_name in config.assets:
        try:
            results.append(run_asset_pick(config, asset_name))
        except Exception as exc:
            print(f"[error] {asset_name}: {exc}")
            results.append({
                "asset": asset_name,
                "success_episode_ids": [],
                "n_episodes": config.n_episodes,
                "success_rate": 0.0,
                "videos": [],
                "error": str(exc),
            })

    summary = {
        "assets": config.assets,
        "n_episodes": config.n_episodes,
        "results": results,
        "successful_assets": [r["asset"] for r in results if r["success_episode_ids"]],
    }
    summary_path = config.output_dir / config.assets[0] / "asset_pick_validation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    root_summary = config.output_dir / "asset_pick_validation_summary.json"
    root_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
