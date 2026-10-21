# Changelog

All notable user-facing changes are documented here. This project follows Semantic Versioning.

## [Unreleased]

### Added

- `robotsmith.assets.producers`: asset generation (TRELLIS.2 and Hunyuan3D backends, `generate_asset`) and
  `mesh_to_urdf`, moved from RocRecon.
- `robotsmith.scenario.relations`: a relation solver that places scenario objects per seed from asset geometry
  (`on`, `near: robot`, `clear_of`, `facing: robot`, fixture `anchor_xy`), and the `object_in_box` predicate behind
  `inside` / `on`, which tests an object against a place target's collision box.

### Changed

- Assets store `asset.json` (kind, source and build files with a build hash, annotations with their writer) and an
  append-only `checks.jsonl` instead of `metadata.json`. Check records bind to the build hash and stop counting when
  the files change. The audit writes `mesh_consistency` and `gripper_fit` records; `render_asset_table.py` writes a
  human `upright` record. Grasp policy buckets moved to `assets/grasp_policy_buckets.json`. Loaded values are
  unchanged for all assets.
- Asset geometry is derived from the URDF and meshes at load (`robotsmith/assets/derive.py`) instead of being stored:
  `size_cm` is the visual-geometry extent, the task joint is the movable joint, and `handles` / `place_targets` name
  a link or collision box whose position is read from the URDF. `drawer_cabinet` grows from 45 to 57.5 cm in x (the
  stored value left out the front panel and handle) and its opening lip moves 10 mm to the tray's front edge.
- Asset build files are metric and `+Z` up: the upright rotation and metric scale are baked into the meshes and URDFs
  of `apple_01`, `die_01`, `lego_02`, `mug_02` and `drawer_cabinet` (0.7), with URDF mass and inertia scaled to match.
  World-frame geometry is unchanged (within 0.001 mm). Fixture yaw moved from the asset into the scenes (`yaw_deg`).
  `gripper_fit` is a check only and no longer rewrites the scale. `render_asset_table.py --apply-upright-index`
  becomes `--record-upright pass|fail`.
- Scenarios are YAML data validated by a JSON Schema instead of Python modules with `build()`; the scenario loader no
  longer executes code. Teacher steps moved to `scenarios/teacher/<name>.yaml`. `robotsmith.cap` is renamed
  `robotsmith.scenario`. The four scenarios were migrated; three success judgements change on purpose: a die on the
  drawer cabinet's top is no longer inside the drawer, anywhere on the lower shelf now counts, and `near: robot`
  samples a wider initial layout than the old hand-written regions.
- `assets/objects/` keeps only the assets the scenarios load: `apple_01`, `die_01`, `lego_02`, `mug_02`, `plane`,
  `table_simple`.
- `README.md` is in English, with the Chinese version in `README.zh-CN.md`; the demo videos render as GIFs. The code
  layout and the upstream projects moved to `docs/design.md`.

### Removed

- `mass_kg`, `density_kg_m3`, `restitution`, `stable_poses` and `grasp_strategy` asset fields (mass comes from the
  URDF; the rest had no readers), `assets/catalog.json` and `AssetLibrary.save_catalog`,
  `scripts/assets/compute_stable_poses.py`, and `scripts/assets/import_objaverse.py`.
- `task_joints`, `joint_init` (default 0; scenes still override), `canonical_frame`, `mounting_holes`,
  `upright_candidates`, and the `tray_floor_z` / `travel_fraction` / `extent_xy` place-target fields, with the
  `render_asset_table.py` modes that read `upright_candidates`.
- The `object_in_container` predicate (xy threshold around a point) and per-scenario camera poses (the camera now
  frames the solved scene).
- The `metric_scale` and `upright` annotations, `AssetMetadata.task_poses`, `scene_state["object_scales"]`, and
  `robotsmith.scenes.pose_utils`.

### Fixed

- The gfx942 image pins `tensordict==0.7.2`. Unpinned, the GraspGen layer pulled a release that requires
  `torch>=2.11`, which replaced the ROCm torch 2.6 with a CUDA wheel and broke startup
  (`operator torchvision::nms does not exist`).
- `docs/quick-start.md` clones rocRobo with its `pyroki` submodule and builds the `rocrobo/pyroki` sidecar image,
  which is not published.

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
