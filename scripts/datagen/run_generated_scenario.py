"""Debug/smoke consume an authoring-ready generated scenario."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import shutil
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from robotsmith.cap.adapters import resolved_target_positions, target_frames
from robotsmith.rollout.video_export import copy_episode_videos
from robotsmith.scenario_runtime import materialize_candidate
from robotsmith.scenario_runtime.materialize import ScenarioRuntimePackage
from robotsmith.scenario_runtime.runner import (
    collect_object_positions,
    evaluate_scenario_task,
    resolve_static_target_positions,
    run_scenario_episode,
)
from robotsmith.tasks import find_leaf
from robotsmith.tasks.task_spec import TaskSpec


@dataclass
class ScenarioSmokeConfig:
    output_dir: Path
    n_episodes: int = 1
    seed: int = 42
    settle_steps: int = 30
    grasp_planner: str = "auto"
    fps: int = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug-run a generated RoboSmith scenario")
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/generated_scenario_run"))
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--grasp-planner", choices=["auto", "learned"], default="auto")
    return parser.parse_args()


def run_scenario_smoke(
    package: ScenarioRuntimePackage,
    config: ScenarioSmokeConfig,
) -> dict:
    """Run a lightweight scenario smoke with videos and diagnostics."""

    from robotsmith.sim.sim_env import SimEnv, ensure_display
    from robotsmith.sim.recorder import (
        articulated_joint_names,
        create_dataset,
        record_episode,
    )
    from robotsmith.execution import MotionExecutor
    from robotsmith.motion import MotionParams

    ensure_display()
    env = SimEnv.build(
        package.legacy_scene,
        seed=config.seed,
        fps=config.fps,
        grasp_planner=config.grasp_planner,
    )
    executor = MotionExecutor()
    motion_params = MotionParams()
    repo_id = f"local/robotsmith-scenario-{package.legacy_task.name}"
    cache_root = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
    if cache_root.exists():
        shutil.rmtree(cache_root)
    camera_names = ("up",)
    env_state_names = articulated_joint_names(env)
    dataset = create_dataset(
        repo_id=repo_id,
        fps=config.fps,
        use_videos=True,
        camera_names=camera_names,
        env_state_names=env_state_names,
    )
    # Frames (live-resolved each segment by the runtime) for motion anchoring;
    # static xyz markers for success/diagnostics. Anchorless targets resolve from
    # the scene spec; asset-anchored *static* targets (placement affordances on a
    # fixture) are resolved against the live world so they too become success
    # markers (articulated/opening anchors stay excluded — they move with a joint).
    scene_frames = target_frames(package.scenario_scene)
    target_positions = {
        name: np.asarray(xyz, dtype=np.float32)
        for name, xyz in resolved_target_positions(package.scenario_scene).items()
    }
    target_positions.update(resolve_static_target_positions(env, scene_frames))

    episodes = []
    episode_frame_counts: dict[int, int] = {}
    for ep in range(config.n_episodes):
        result = run_scenario_episode(
            env,
            package.legacy_task,
            executor=executor,
            motion_params=motion_params,
            dataset=dataset,
            record_fn=record_episode,
            settle_steps=config.settle_steps,
            target_frames=scene_frames,
            camera_names=camera_names,
            env_state_names=env_state_names,
        )
        episode_frame_counts[ep] = result.frame_count
        success = evaluate_scenario_task(
            env,
            package.legacy_task,
            initial_positions=result.initial_positions,
            reference_positions=target_positions,
        )
        final_positions = collect_object_positions(env)
        episodes.append({
            "episode_index": ep,
            "success": bool(success),
            "frame_count": result.frame_count,
            "diagnostics": episode_diagnostics(
                package.legacy_task,
                initial_positions=result.initial_positions,
                final_positions=final_positions,
                target_positions=target_positions,
                skill_traces=result.skill_traces,
            ),
        })
        dataset.save_episode()

    if hasattr(dataset, "consolidate"):
        dataset.consolidate(run_compute_stats=True)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(dataset.root)
    videos = copy_episode_videos(
        dataset_root,
        [ep["episode_index"] for ep in episodes],
        config.output_dir / "videos",
        package.legacy_task.name,
        None,
        episode_frame_counts,
        config.fps,
    )

    final_joint_positions = (
        env.get_joint_positions() if hasattr(env, "get_joint_positions") else {}
    )
    summary = {
        "scenario": package.legacy_task.name,
        "source_path": str(package.source_path),
        "status": "datagen_success"
        if all(ep["success"] for ep in episodes)
        else "datagen_failed",
        "n_episodes": config.n_episodes,
        "success_count": sum(1 for ep in episodes if ep["success"]),
        "output_dir": str(config.output_dir),
        "dataset_root": str(dataset_root),
        "env_state_names": list(env_state_names),
        "final_joint_positions": _json_safe(final_joint_positions),
        "videos": videos,
        "episodes": episodes,
    }
    (config.output_dir / "scenario_run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def episode_diagnostics(
    task_spec: TaskSpec,
    *,
    initial_positions: dict[str, np.ndarray],
    final_positions: dict[str, np.ndarray],
    target_positions: dict[str, np.ndarray],
    skill_traces: list[dict],
) -> dict:
    diagnostics = {
        "initial_positions": _positions_to_json(initial_positions),
        "final_positions": _positions_to_json(final_positions),
        "target_positions": _positions_to_json(target_positions),
        "skill_traces": _json_safe(skill_traces),
    }
    in_container = find_leaf(task_spec.success, "object_in_container")
    if in_container is not None:
        obj_name = in_container.params.get("object")
        container_name = in_container.params.get("container")
        obj_pos = final_positions.get(obj_name)
        target_pos = final_positions.get(container_name)
        if target_pos is None:
            target_pos = target_positions.get(container_name)
        if obj_pos is not None and target_pos is not None:
            diagnostics["success_xy_error"] = float(
                np.linalg.norm(obj_pos[:2] - target_pos[:2])
            )
    return diagnostics


def _materialize_summary(package: ScenarioRuntimePackage) -> dict:
    return {
        "scenario": package.legacy_task.name,
        "source_path": str(package.source_path),
        "scene": package.legacy_scene.name,
        "objects": [obj.name_override for obj in package.legacy_scene.objects],
        "task": package.legacy_task.name,
        "success": package.legacy_task.success.to_dict(),
        "skills": [skill.to_dict() for skill in package.legacy_task.skills],
    }


def _positions_to_json(positions: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {
        name: [float(v) for v in np.asarray(pos, dtype=np.float32).tolist()]
        for name, pos in positions.items()
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = parse_args()
    package = materialize_candidate(args.scenario)
    if args.materialize_only:
        print(json.dumps(_materialize_summary(package), indent=2))
        return

    if not args.smoke:
        raise SystemExit("only --materialize-only or --smoke is implemented")

    summary = run_scenario_smoke(
        package,
        ScenarioSmokeConfig(
            output_dir=args.output_dir,
            n_episodes=args.n_episodes,
            seed=args.seed,
            settle_steps=args.settle_steps,
            grasp_planner=args.grasp_planner,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
