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

DEFAULT_ASSETS = [
    "apple_02",
    "die_01",
    "mug_02",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate picks across assets")
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
    ap.add_argument("--grasp-selection", choices=["trial", "trial_all"], default="trial",
                    help="trial: try candidates in simulation and keep the first that lifts; trial_all: try every one.")
    ap.add_argument("--object-xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="Place the object here every episode instead of sampling the workspace.")
    ap.add_argument("--diagnose-dir", type=Path, default=None,
                    help="When every episode fails, write the repair agent's request "
                         "(robotsmith.grasp.diagnostics) under DIR/<asset>/.")
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
            object_xy=tuple(args.object_xy) if args.object_xy else None,
            grasp_selection=args.grasp_selection,
            diagnose_dir=args.diagnose_dir,
        )
    )
    print("======== VALIDATION SUMMARY ========")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
