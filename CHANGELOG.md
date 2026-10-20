# Changelog

All notable user-facing changes are documented here. This project follows Semantic Versioning.

## [Unreleased]

### Added

- `pick` routes its free segments (home → pre-grasp, lift / retreat) through `plan_motion` when the motion
  backend is collision-aware; the final approach stays Cartesian.
- Unit test suite under `tests/`, run in CI on every pull request.
- `snapshot_scenario.py` accepts `--overview-pos`, `--overview-lookat`, and `--fov`.
- Contribution guide, code of conduct, security policy, AI contribution policy, and issue / PR templates.

### Changed

- `scripts/setup/run_scenario.sh` takes host mounts from `REPO`, `GRASPGEN_DIR`, `GRASPGEN_MODELS`, and
  `HF_CACHE`.
- The `pick_place_into_drawer` camera looks at the drawer opening.

### Known issues

- rocRobo `plan_motion` solves endpoint IK from random seeds only, so a near-singular endpoint can fall back to a
  collision-blind straight line.
