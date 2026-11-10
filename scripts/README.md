# RoboSmith Scripts

Runnable entry points around the `robotsmith/` package, grouped by workflow
stage: **assets → datagen → acceptance**, plus **setup**.

## `assets/` — asset onboarding

- `list_assets.py`: print the asset catalog a scenario author picks from (id, kind, size, place targets, movable joints).
- `render_asset_table.py`: render Genesis table QA views of the assets as built.
- `review_calibration.py`: run the asset reviewer on cases with a known verdict; run it after changing the review instructions, package or readings.

## `datagen/` — run scenarios (each run also records a LeRobot dataset)

- `run_generated_scenario.py`: run a YAML scenario passed via `--scenario` (`--materialize-only` validates and lowers it).
- `validate_asset_picks.py`: asset-driven pick rollout + dataset.
- `snapshot_scenario.py`: render static layout snapshots (overview/topdown PNG).

## `acceptance/` — acceptance families ([docs/acceptance.md](../docs/acceptance.md))

- `certify.py`: run the manipulation chain on a family's instances and judge every physics step; writes `instances.jsonl` and `summary.json`.
- `witness.py`: the solvability witness, independent of the chain under test.
- `batch.py`: the batch runner both use; a worker that hangs past an attempt's timeout is killed from outside and resumed.
- `make_fixtures.py`: write the URDF-primitive fixtures of the families into `assets/`.

## `setup/` — environment & service helpers

Not product logic — ROCm environment plumbing:

- `_install_scatter_shim.py`: `torch_scatter` shim installed by the Docker build.
- `run_scenario.sh`: canonical env-driven e2e launcher; defaults to the MI300 / gfx942 image; override `NAME`/`SCENARIO`/`IMAGE`, and the host mounts `REPO`/`HF_CACHE`.

## Local-only diagnostics (gitignored, not shipped)

- `scripts/ablation_study/` (grasp ablation harness)
- `scripts/debug_tools/` (grasp visualization / endpoint probes)
