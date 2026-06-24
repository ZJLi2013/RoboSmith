"""Stage-1 policy onboarding and Stage-2 metadata-guided rollout runner."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from robotsmith.assets.schema import NO_POSITIVE_GRASP_POLICY_BUCKET
from robotsmith.grasp.policy_onboarding import (
    POLICY_BUCKETS,
    eligible_probe_buckets,
    load_metadata,
    save_policy_bucket,
    select_positive_winners,
)
from robotsmith.rollout.asset_pick_validation import (
    AssetPickValidationConfig,
    run_asset_pick_validation,
)


@dataclass
class RolloutOrchestratorConfig:
    assets: list[str]
    assets_root: Path
    output_dir: Path
    probe_episodes: int = 2
    rollout_episodes: int = 5
    seed: int = 42
    grasp_planner: str = "learned"
    no_videos: bool = False
    probe_buckets: list[str] | tuple[str, ...] = POLICY_BUCKETS
    stage1_only: bool = False


def _run_validation(
    *,
    asset_name: str,
    output_root: Path,
    n_episodes: int,
    seed: int,
    grasp_planner: str,
    no_videos: bool,
    clean: bool,
    probe_bucket: str | None = None,
) -> dict[str, Any]:
    summary = run_asset_pick_validation(
        AssetPickValidationConfig(
            assets=[asset_name],
            output_dir=output_root,
            n_episodes=n_episodes,
            seed=seed,
            no_videos=no_videos,
            clean=clean,
            grasp_planner=grasp_planner,
            grasp_policy_bucket_override=probe_bucket,
        )
    )
    results = summary.get("results", [])
    if not results:
        return {
            "asset": asset_name,
            "success_episode_ids": [],
            "n_episodes": 0,
            "success_rate": 0.0,
            "error": "empty validation summary",
        }
    return results[0]


def _probe_asset(
    config: RolloutOrchestratorConfig,
    asset_name: str,
    buckets: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    probe_results = []
    for bucket in buckets:
        output_root = config.output_dir / asset_name / "stage1" / bucket
        result = _run_validation(
            asset_name=asset_name,
            output_root=output_root,
            n_episodes=config.probe_episodes,
            seed=config.seed,
            grasp_planner=config.grasp_planner,
            no_videos=True,
            clean=True,
            probe_bucket=bucket,
        )
        success_count = len(result.get("success_episode_ids", []))
        probe_results.append({
            "bucket": bucket,
            "success_count": success_count,
            "n_episodes": result.get("n_episodes", config.probe_episodes),
            "success_rate": result.get("success_rate", 0.0),
            "result": result,
        })
    return select_positive_winners(probe_results), probe_results


def _scan_asset(config: RolloutOrchestratorConfig, asset_name: str) -> dict[str, Any]:
    return _run_validation(
        asset_name=asset_name,
        output_root=config.output_dir / asset_name / "stage1" / "p0_scan",
        n_episodes=1,
        seed=config.seed,
        grasp_planner=config.grasp_planner,
        no_videos=True,
        clean=True,
    )


def run_asset(config: RolloutOrchestratorConfig, asset_name: str) -> dict[str, Any]:
    meta = load_metadata(asset_name, config.assets_root)
    existing_bucket = meta.grasp_policy_bucket
    record: dict[str, Any] = {
        "asset": asset_name,
        "initial_grasp_policy_bucket": existing_bucket,
    }

    if existing_bucket == NO_POSITIVE_GRASP_POLICY_BUCKET:
        record["stage1"] = "skipped_existing_no_positive"
        record["final_grasp_policy_bucket"] = existing_bucket
        record["status"] = "no_positive"
        return record
    if existing_bucket:
        record["stage1"] = "skipped_existing_metadata"
        bucket = existing_bucket
    else:
        scan_result = _scan_asset(config, asset_name)
        eligible_buckets = eligible_probe_buckets(scan_result, config.probe_buckets)
        buckets, probe_results = _probe_asset(config, asset_name, eligible_buckets)
        record["stage1"] = {
            "status": "completed",
            "p0_scan": scan_result,
            "eligible_probe_buckets": eligible_buckets,
            "probe_results": probe_results,
        }
        if not eligible_buckets:
            record["status"] = "no_eligible_probe_buckets"
            return record
        if not buckets:
            record["status"] = "no_positive"
            return record
        bucket = buckets[0] if len(buckets) == 1 else buckets
        if len(buckets) > 1:
            record["tied_buckets"] = buckets
        save_policy_bucket(asset_name, config.assets_root, bucket)
        record["written_grasp_policy_bucket"] = bucket

    if config.stage1_only:
        record["final_grasp_policy_bucket"] = bucket
        record["status"] = "stage1_done"
        return record

    stage2_root = config.output_dir / asset_name / "stage2"
    result = _run_validation(
        asset_name=asset_name,
        output_root=stage2_root,
        n_episodes=config.rollout_episodes,
        seed=config.seed,
        grasp_planner=config.grasp_planner,
        no_videos=config.no_videos,
        clean=True,
    )
    record["stage2"] = result
    record["final_grasp_policy_bucket"] = bucket
    record["status"] = "done"
    return record


def run_rollout_orchestrator(config: RolloutOrchestratorConfig) -> dict[str, Any]:
    """Top-level entry: run the Stage-1/Stage-2 flow over ``config.assets``.

    For each asset, ``run_asset`` does Stage-1 probe (per-bucket validation +
    positive-winner bucket selection, persisted to metadata) and, unless
    ``stage1_only``, the Stage-2 metadata-guided rollout. Per-asset records are
    aggregated into a summary dict, also written to
    ``output_dir/metadata_guided_rollout_summary.json``.
    """
    config.assets_root = config.assets_root.resolve()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    records = [run_asset(config, asset_name) for asset_name in config.assets]
    summary = {
        "assets": config.assets,
        "probe_episodes": config.probe_episodes,
        "rollout_episodes": config.rollout_episodes,
        "results": records,
    }
    summary_path = config.output_dir / "metadata_guided_rollout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
