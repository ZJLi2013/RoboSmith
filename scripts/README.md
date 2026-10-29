# RoboSmith Scripts

Runnable entry points around the `robotsmith/` package, grouped by workflow
stage: **assets → datagen → setup**.

## `assets/` — asset onboarding

- `list_assets.py`: print the asset catalog a scenario author picks from (id, kind, size, place targets, movable joints).
- `render_asset_table.py`: render Genesis table QA views of the assets as built.
- `review_calibration.py`: run the asset reviewer on cases with a known verdict; run it after changing the review instructions, package or readings.

## `datagen/` — data generation (SDG)

- `run_generated_scenario.py`: run a YAML scenario passed via `--scenario` (`--materialize-only` validates and lowers it).
- `validate_asset_picks.py`: asset-driven pick rollout + dataset.
- `snapshot_scenario.py`: render static layout snapshots (overview/topdown PNG).
## `setup/` — environment & service helpers

Not product logic — ROCm environment plumbing:

- `_install_scatter_shim.py`: `torch_scatter` shim installed by the Docker build.
- `run_scenario.sh`: canonical env-driven e2e launcher (RoboSmith + rocRobo sidecar); defaults to the MI300 / gfx942 image; override `NAME`/`SCENARIO`/`IMAGE`, and the host mounts `REPO`/`HF_CACHE`.

The dev-only `rocrobo_serve_smoke.py` under `scripts/setup/` is gitignored.

## Top-Level

- `train_smolvla.py`: policy training entry.

## Local-only diagnostics (gitignored, not shipped)

- `scripts/ablation_study/` (grasp ablation harness) 
- `scripts/debug_tools/` (grasp visualization / endpoint probes) 
