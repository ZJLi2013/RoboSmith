"""Asset-level pick rollout validation workflow."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil
from typing import Callable

import numpy as np

from robotsmith.scenario.adapters import to_legacy_scene_config, to_legacy_skills
from robotsmith.scenario.intents import intent_sequence, pick
from robotsmith.scenario.predicates import object_lifted
from robotsmith.scenario.scene_api import layout, obj, region, scene
from robotsmith.scenario_runtime.runner import EpisodeRun
from robotsmith.sim.world_state import collect_world_state
from robotsmith.rollout.episode_layout import (
    build_episode_positions,
    derive_skill_targets,
)
from robotsmith.rollout.video_export import copy_episode_videos
from robotsmith.scenes.config import SceneConfig, TASK_WORKSPACE_XY
from robotsmith.tasks import evaluate_success
from robotsmith.tasks.task_spec import TaskSpec

FPS = 30
EPISODE_DIAGNOSTIC_FIELDS = (
    "episode_index",
    "success",
    "frame_count",
    "selected_candidate_index",
    "no_candidates",
    "no_trial_success",
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
    object_xy: tuple[float, float] | None = None
    grasp_selection: str = "trial"
    # When every episode fails, write a diagnosis request for the repair agent under <diagnose_dir>/<asset>/.
    diagnose_dir: Path | None = None


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
    intents = intent_sequence([pick("object", category=category)])
    return TaskSpec(
        name=f"pick_{asset_name}",
        instruction=f"Pick up {asset_name}",
        scene=cap_scene.name,
        success=object_lifted("object", z_margin=0.02),
        skills=to_legacy_skills(intents, cap_scene),
    )


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


def _pick_trace(skill_traces: list[dict]) -> dict:
    for trace in skill_traces:
        if trace.get("skill") == "pick" and trace.get("target") == "object":
            return trace
    return {}


def _reset_asset_pick_episode(
    env,
    task_spec: TaskSpec,
    *,
    rng: random.Random,
    settle_steps: int,
    object_xy: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Reset the object to a sampled (or the pinned) workspace point and settle; return the settled positions."""
    pick_names, place_names = derive_skill_targets(task_spec)
    positions = build_episode_positions(
        env,
        rng,
        pick_names,
        place_names,
        task_spec=task_spec,
    )
    if object_xy is not None:
        positions["object"][:2] = object_xy
    obj_xy_map = {
        name: (pos[0], pos[1])
        for name, pos in positions.items()
        if name in env.entity_map
    }
    settled_positions = env.reset(obj_xy_map, settle_steps=settle_steps)
    for name, pos in settled_positions.items():
        positions[name] = pos.copy()

    return {
        name: pos.copy()
        for name, pos in positions.items()
        if name in env.entity_map
    }


def _run_pick(env, task_spec: TaskSpec, *, executor, motion_params, dataset, record_fn: Callable, camera_names,
              grasp_selection: str) -> EpisodeRun:
    """Run the task's pick from the settled scene and record it into ``dataset``; the run holds the traces and,
    when the pick stopped with no motion, ``stopped``."""
    run = EpisodeRun(env, task_spec, executor=executor, motion_params=motion_params, dataset=dataset,
                     record_fn=record_fn, target_frames={}, camera_names=camera_names, env_state_names=(),
                     grasp_selection=grasp_selection)
    run.run_steps(list(task_spec.skills), [])
    return run


def _evaluate_asset_pick_task(
    env,
    task_spec: TaskSpec,
    *,
    initial_positions: dict[str, np.ndarray],
) -> bool:
    env_state = {**collect_world_state(env), "initial_positions": initial_positions}
    return evaluate_success(task_spec.success, env_state)


def _start_state(env, initial_positions: dict[str, np.ndarray]) -> dict:
    """The settled start the planner sees: object pose and simulator AABB."""
    ent = env.entity_map["object"]
    aabb = ent.get_AABB()
    aabb = (aabb.cpu().numpy() if hasattr(aabb, "cpu") else np.asarray(aabb)).reshape(-1, 2, 3)[0]
    return {"pos": [float(v) for v in initial_positions["object"]],
            "quat": [float(v) for v in env.object_quats["object"]], "aabb": aabb.tolist()}


def _write_diagnosis_request(config: AssetPickValidationConfig, env, asset_name: str, start: dict,
                             label: dict) -> None:
    from robotsmith.grasp.diagnostics import request

    asset = env.asset_map["object"]
    object_xy = list(config.object_xy) if config.object_xy else start["pos"][:2]
    packet = request.evidence(
        urdf_path=asset.urdf_path, category=_asset_category(asset_name), pos=start["pos"], quat=start["quat"],
        aabb=start["aabb"], table_z=env.table_surface_z, gripper=env.planner.gripper, plans=start["plans"],
        label=label, object_xy=object_xy)
    path = request.write_request(
        config.diagnose_dir / asset_name, packet=packet, table_z=env.table_surface_z, gripper=env.planner.gripper,
        context={"asset": asset_name, "object_xy": object_xy, "seed": config.seed,
                 "settle_steps": config.settle_steps})
    print(f"[diagnose] {asset_name}: request written to {path}")


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

    env = SimEnv.build(scene_config, seed=config.seed, fps=FPS)
    print(f"======== START {asset_name} ========")
    print(f"[scene] pose=upright grasp_selection={config.grasp_selection} settle_steps={config.settle_steps}")
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

    for ep in range(config.n_episodes):
        initial_positions = _reset_asset_pick_episode(
            env,
            task_spec,
            rng=rng,
            settle_steps=config.settle_steps,
            object_xy=config.object_xy,
        )
        pose_label = _object_pose_label(env, "object")
        if ep == 0:
            start = _start_state(env, initial_positions)
        run = _run_pick(env, task_spec, executor=executor, motion_params=motion_params, dataset=dataset,
                        record_fn=record_episode, camera_names=camera_names, grasp_selection=config.grasp_selection)
        if ep == 0:
            plans = getattr(env.planner, "last_plans", None)
            start["plans"] = list(plans) if plans is not None else None

        n_frames = run.frame_count
        stopped = {k: run.stopped[k] for k in ("cause", "evidence", "steps")} if run.stopped else None
        pick_trace = _pick_trace(run.skill_traces)
        if frames_per_episode is None:
            frames_per_episode = n_frames
            print(f"[rollout] trajectory: {frames_per_episode} frames/episode")
        episode_frame_counts[ep] = n_frames

        success = _evaluate_asset_pick_task(
            env,
            task_spec,
            initial_positions=initial_positions,
        )
        label = {
            "episode_index": ep,
            "task": task_spec.name,
            "asset": asset_name,
            "success": bool(success),
            "frame_count": n_frames,
            "stopped": stopped,
            "selected_candidate_index": pick_trace.get("selected_candidate_index"),
            "no_candidates": bool(pick_trace.get("no_candidates", False)),
            "no_trial_success": pick_trace.get("no_trial_success"),
            "trials": pick_trace.get("trials"),
            "trial_seconds": pick_trace.get("trial_seconds"),
            "object_initial_pos": [float(v) for v in initial_positions["object"]],
            "finger_final_qpos": [float(v) for v in to_numpy(env.franka.get_dofs_position(env.finger_dof))],
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
        if n_frames:
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
    if config.diagnose_dir and not success_ids and start["plans"] is not None:
        _write_diagnosis_request(config, env, asset_name, start, episode_labels[0])
    copied = []
    if not config.no_videos:
        copied.extend(copy_episode_videos(
            dataset_root,
            [e["episode_index"] for e in episode_labels if e["frame_count"]],
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
