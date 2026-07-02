"""Experiment CLI for metadata-guided rollout validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from robotsmith.grasp.policy_onboarding import POLICY_BUCKETS
from robotsmith.rollout.rollout_orchestrator import (
    RolloutOrchestratorConfig,
    run_rollout_orchestrator,
)

DEFAULT_ASSETS = (
    "block_blue",
    "bowl_02",
    "figurine_01",
    "figurine_02",
    "figurine_03",
    "fruit_01",
    "fruit_03",
    "mug_01",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run metadata-guided rollout flow")
    parser.add_argument("--assets", nargs="+", default=list(DEFAULT_ASSETS))
    parser.add_argument("--assets-root", type=Path, default=REPO_ROOT / "assets")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/metadata_guided_rollout"),
    )
    parser.add_argument("--probe-episodes", type=int, default=2)
    parser.add_argument("--rollout-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grasp-planner", choices=["auto", "learned"], default="learned")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--probe-buckets", nargs="+", default=list(POLICY_BUCKETS))
    parser.add_argument("--stage1-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_rollout_orchestrator(
        RolloutOrchestratorConfig(
            assets=args.assets,
            assets_root=args.assets_root,
            output_dir=args.output_dir,
            probe_episodes=args.probe_episodes,
            rollout_episodes=args.rollout_episodes,
            seed=args.seed,
            grasp_planner=args.grasp_planner,
            no_videos=args.no_videos,
            probe_buckets=args.probe_buckets,
            stage1_only=args.stage1_only,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
