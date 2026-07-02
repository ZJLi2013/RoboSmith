"""CLI wrapper for asset pick rollout validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from robotsmith.rollout.asset_pick_validation import (
    AssetPickValidationConfig,
    run_asset_pick_validation,
)

# Regression set: one asset per category from the c10/c11 "one success per
# asset" demo (output/exp_c10_c11_one_success_per_asset_demo). Validates that
# GraspGen + policy-bucket selection has no regression after refactors.
# Objects-only here because git tracks asset metadata/URDF but ignores meshes
# (assets/objects/**/*.glb etc) and all of assets/generated/*; append the c11
# generated assets (battery_1_trellis2_20260527, glass_2_trellis2_20260527,
# moto_1_trellis2_20260527, plastic_1_trellis2_20260527) via --assets once they
# are synced onto the run host.
DEFAULT_ASSETS = [
    "apple_01",
    "carton_01",
    "die_01",
    "mug_02",
    "fruit_03",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate learned picks across assets")
    ap.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS)
    ap.add_argument("--n-episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=Path, default=Path("output/exp34e_asset_validation"))
    ap.add_argument("--no-videos", action="store_true")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument(
        "--settle-steps",
        type=int,
        default=30,
        help="Physics settle steps after each reset before planning.",
    )
    ap.add_argument("--grasp-planner", choices=["auto", "learned"], default="auto")
    ap.add_argument(
        "--grasp-policy-bucket-override",
        default=None,
        help="Probe-only bucket override for Stage 1; does not edit metadata.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_asset_pick_validation(
        AssetPickValidationConfig(
            assets=args.assets,
            output_dir=args.output_dir,
            n_episodes=args.n_episodes,
            seed=args.seed,
            no_videos=args.no_videos,
            clean=args.clean,
            settle_steps=args.settle_steps,
            grasp_planner=args.grasp_planner,
            grasp_policy_bucket_override=args.grasp_policy_bucket_override,
        )
    )
    print("======== VALIDATION SUMMARY ========")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
