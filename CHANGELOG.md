# Changelog

All notable user-facing changes are documented here. This project follows Semantic Versioning.

## [Unreleased]

## [0.1.0] - 2026-10-20

First public release: scripted-teacher synthetic data generation. Assets plus a CAP scenario go in; Genesis
rollouts filtered by the scenario's success tree come out as a LeRobot dataset with videos and a run summary. It
covers rigid pick / place with learned grasp proposals (GraspGen), prismatic open / close, long-horizon
`open → pick → place → close` drawer tasks, and collision-aware free segments through the rocRobo sidecar, on
MI300 (gfx942) and MI350 (gfx950). This pipeline backs the AAI Day 2026 workshop
[interaction_sdg_workshop](https://github.com/PhysicalAI-AIM/interaction_sdg_workshop), which generates the
drawer `open → pick → place → close` dataset on gfx942.

Not validated in this release: the effect of the generated data on policy training. `scripts/train_smolvla.py`
trains on a recorded dataset, but there is no closed-loop evaluation entry and no behavior-cloning result.

### Added

- CAP scenario API (scene, task, skill intents, success tree) and four example scenarios under `scenarios/`.
- `pick` routes its free segments (home → pre-grasp, lift / retreat) through `plan_motion` when the motion
  backend is collision-aware; the final approach stays Cartesian.
- Unit test suite under `tests/`, run in CI on every pull request.
- `snapshot_scenario.py` accepts `--overview-pos`, `--overview-lookat`, and `--fov`.
- Contribution guide, code of conduct, security policy, AI contribution policy, and issue / PR templates.
- `scripts/assets/import_objaverse.py`, which produced the Objaverse assets in `assets/objects/`.
- `scripts/train_smolvla.py` for SmolVLA post-training on a recorded dataset, and a `lerobot` extra.
- `docs/design.md` (design and key concepts) and `docs/quick-start.md` (fresh GPU node to a first recorded run).

### Changed

- `scripts/setup/run_scenario.sh` takes host mounts from `REPO`, `GRASPGEN_DIR`, `GRASPGEN_MODELS`, and
  `HF_CACHE`.
- The `pick_place_into_drawer` camera looks at the drawer opening.
- `docker/README.md` builds from `docker/Dockerfile.gfx942` / `docker/Dockerfile.gfx950`.

### Known issues

- rocRobo `plan_motion` solves endpoint IK from random seeds only, so a near-singular endpoint can fall back to a
  collision-blind straight line.
- Contact segments are straight lines without collision checks, and any failed `plan_motion` falls back to one.
- Articulation supports prismatic joints only.
- One scene per process: a second `SimEnv.build` in the same process fails with `Genesis already initialized`.
- GraspGen is under an NVIDIA research / evaluation license; it is fetched into local images and not distributed
  with RoboSmith.
