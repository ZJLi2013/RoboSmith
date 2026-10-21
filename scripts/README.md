# RoboSmith Scripts

Runnable entry points around the `robotsmith/` package, grouped by workflow
stage: **assets → datagen → setup**.

## `assets/` — asset onboarding

- `render_asset_table.py`: render Genesis table QA views of the assets as built; `--record-upright pass|fail` writes a
  human `upright` check record.

## `datagen/` — data generation (SDG)

- `run_generated_scenario.py`: run a CAP scenario module passed via `--scenario`.
- `validate_asset_picks.py`: asset-driven pick rollout + dataset.
- `run_metadata_guided_rollout.py`: Stage-1/Stage-2 metadata-guided rollout.
- `snapshot_scenario.py`: render static layout snapshots (overview/topdown PNG).
## `setup/` — environment & service helpers

Not product logic — ROCm/GraspGen environment plumbing:

- `_convert_spconv_ckpt.py`, `_install_scatter_shim.py`: ROCm GraspGen setup helpers (invoked by the Docker build / quick-start).
- `run_scenario.sh`: canonical env-driven e2e launcher (RoboSmith + rocRobo sidecar); defaults to the MI300 / gfx942 image; override `NAME`/`SCENARIO`/`PLANNER`/`IMAGE`, and the host mounts `REPO`/`GRASPGEN_DIR`/`GRASPGEN_MODELS`/`HF_CACHE`.

The dev-only `rocrobo_serve_smoke.py` under `scripts/setup/` is gitignored.

## Top-Level

- `train_smolvla.py`: policy training entry.

## Local-only diagnostics (gitignored, not shipped)

- `scripts/ablation_study/` (grasp ablation harness) 
- `scripts/debug_tools/` (grasp visualization / endpoint probes) 
