# RoboSmith Scripts

Runnable entry points around the `robotsmith/` package, grouped by workflow
stage: **assets → datagen → setup**.

## `assets/` — asset onboarding

- `import_objaverse.py`: curate Objaverse assets into `assets/objects/`.
- `compute_stable_poses.py`: write stable-pose metadata for assets.
- `render_asset_table.py`: render Genesis table QA views (frame/scale audit).

## `datagen/` — data generation (SDG)

- `run_generated_scenario.py`: run a CAP scenario module passed via `--scenario`.
- `validate_asset_picks.py`: asset-driven pick rollout + dataset.
- `run_metadata_guided_rollout.py`: Stage-1/Stage-2 metadata-guided rollout.
- `snapshot_scenario.py`: render static layout snapshots (overview/topdown PNG).

The grasp data-gen entry points and their relationship are explained in
[docs/grasping.md §运行入口](../docs/grasping.md#运行入口cli); this list stays
descriptive only.

## `setup/` — environment & service helpers

Not product logic — ROCm/GraspGen environment plumbing:

- `_convert_spconv_ckpt.py`, `_install_scatter_shim.py`: ROCm GraspGen setup helpers (invoked by the Docker build / quick-start).
- `run_scenario.sh`: canonical env-driven e2e launcher (RoboSmith + rocRobo sidecar); defaults to wbb3/MI350, override `NAME`/`SCENARIO`/`PLANNER`/`IMAGE` for other nodes. See [docs/runtime.md §E2E](../docs/runtime.md#e2e-robosmith-plus-rocrobo-sidecar).

The dev-only `rocrobo_serve_smoke.py` under `scripts/setup/` is gitignored.

## Top-Level

- `train_smolvla.py`: policy training entry.

## Local-only diagnostics (gitignored, not shipped)

- `scripts/ablation_study/` (grasp ablation harness) 
- `scripts/debug_tools/` (grasp visualization / endpoint probes) 
