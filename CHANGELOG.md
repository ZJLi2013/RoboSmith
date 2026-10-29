# Changelog

All notable user-facing changes are documented here. This project follows Semantic Versioning.

## [Unreleased]

### Added

- `robotsmith.scenario_runtime.handoff`: every skill of a scenario episode reports whether it left the world the way
  the next skill needs it (drawer opened, object lifted, object carried until release and resting on the target,
  drawer closed). `scenario_run_summary.json` lists the handoffs and, for a failed episode, the first unfinished step
  with a candidate cause (`motion?`, `grip?`, `release?`), or `unknown` when every handoff passed.
- `robotsmith.assets.producers`: asset generation (TRELLIS.2 and Hunyuan3D backends, `generate_asset`) and
  `mesh_to_urdf`, moved from RocRecon.
- `robotsmith.scenario.relations`: a relation solver that places scenario objects per seed from asset geometry
  (`on`, `near: robot`, `clear_of`, `facing: robot`, fixture `anchor_xy`), and the `object_in_box` predicate behind
  `inside` / `on`, which tests an object against a place target's collision box.
- `scripts/assets/list_assets.py`: the asset catalog for scenario authors. Given only `scenarios/AGENT_PROMPT.md` and
  this catalog, GPT-6 Astra wrote 8 of 9 one-sentence requests as valid YAML scenarios with the intended success
  trees and declined a request for an asset the library does not have.
- `AssetLibrary.search_assets(query, category=, size_cm=)`: admitted assets of a category, nearest size first, or
  sharing a query word when no category is given. `list_assets.py` shows each asset's category and takes
  `--category`, `--size` and `--query`.

### Added

- Grasp repair loop for geometric candidates (`robotsmith.grasp.diagnostics`). `validate_asset_picks --diagnose-dir`
  writes, when every episode fails, a request for a diagnosis agent: the evidence of that run (settled pose, simulator
  AABB, horizontal sections, candidates and trials, table height and gripper limits) plus a playbook of known failure
  signatures. `python -m robotsmith.grasp.diagnostics.tool` is the agent's `try_grasp`; `...diagnostics.writeback`
  re-trials the agent's repair once and, if it lifts, writes it to the asset.
- `GeometricGraspPlanner` puts grasps written back to the asset (`grasp_candidates` annotation) ahead of its own
  candidates, while a passing `trial_grasp` check is current for the asset's revision, build files and embodiment
  (`robotsmith.grasp.annotations`).

### Changed

- Geometric candidates chosen by trial are the only grasp path for single-asset validation and scenario runs.
  Scenario runs try each candidate from the live scene and restore it afterwards (`SimEnv.trial_from_here`).
  `--grasp-planner` is gone from both scripts and `validate_asset_picks --grasp-selection` takes `trial` (default)
  or `trial_all`.
- `SimEnv.reset` raises `SceneInvalid` when an object moves more than 5 cm, drops more than 5 cm or tips more than
  30 deg from where it was placed during the settle steps. Single-asset validation then reports the asset as an error
  without trying grasps, and scenario runs stop with the object and readings, instead of starting from a rolled or
  fallen object.
- Admission refuses duplicates (`robotsmith.assets.dedup`). Each build records once which admitted assets it may
  duplicate: the same build hash refuses it, and an asset of the same category, kind, joints and place targets with
  every edge within 20% needs a `dedup_review` verdict from a reviewer who saw both. GPT-6 Astra judged
  `drawer_cabinet_d1` a duplicate of `drawer_cabinet_a` and `chair_02` of `chair_01`, so both are drafts, and
  `office48` places `chair_01` twice; 12 assets are admitted.

- An asset is admitted when its current build has an `llm_review` pass and a redistributable `source.license`, and is
  a draft otherwise (`store.draft_reasons`). Scenarios raise `AssetNotAdmitted` for a draft, `fetch` and `publish`
  skip drafts and report how many, and `list_assets.py` shows each status. Review no longer moves directories:
  producers write `assets/<id>/` directly and `assets/pending/` is gone. `drawer_cabinet` and `two_layer_supporter`
  passed a GPT-6 Astra review with simulator probes; `plane`, `table_simple` and `drawer_cabinet_blender` stay drafts,
  since nothing loads their geometry. The scenario table is always the solid block sized by `table_size`, built
  without looking up `table_simple`. `publish` no longer raises `PublishRefused`.
- Assets live in `assets/<id>/` instead of `objects/`, `generated/` and `articraft/`, and `asset.json` names a
  `category`. Ids no longer carry a scene name: the `office48_*`
  assets are `bottle_01`, `phone_01`, `chair_01`, `chair_02` and `cabinet_01`, and `behavior_living_radio` is
  `radio_01`; build hashes and check records are unchanged. A scenario's `scene` may name a layout
  `scenes/<name>.yaml` shared by several scenarios: `office48_desk` and the new `office48_phone` use
  `scenes/office48.yaml`. The AHa-3D importer takes `--scene` instead of `--prefix` and writes the layout,
  `review.package admit` takes `--library` instead of `--generated`, and `blender_item` requires `--category`.
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
- `assets/objects/` keeps only the assets the scenarios load: `apple_02`, `block_01`, `die_01`, `mug_02`, `plane`,
  `table_simple`. `apple_02` and `block_01` are built in Blender by GPT-5.6 Sol and admitted on an agent review; they
  replace the Objaverse `apple_01` (lying on its side) and `lego_02` (a minifigure, not a block).
- `README.md` is in English, with the Chinese version in `README.zh-CN.md`; the demo videos render as GIFs. The code
  layout and the upstream projects moved to `docs/design.md`.

### Removed

- GraspGen and everything built around it: the learned planner and model wrapper, the support-clearance screen,
  feasibility gate and 9-bucket policy, policy onboarding and the metadata-guided rollout orchestrator, the spconv
  checkpoint converter, and the `grasp_policy_bucket` asset field with its `assets/grasp_policy_buckets.json` sidecar.
  It no longer ran in any scenario: it found meshes only by the file names `visual.obj` / `collision.obj`.
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
- The grasp support-band check spans each finger pad `finger_open` from the grasp centre. `finger_open` is per
  finger, and the check used half of it, so grasps whose lower pad enters the support surface passed P0.
- Geometric grasp candidates close along the Panda hand's local Y axis, the axis its finger joints move on. They
  put the closing direction in X, so the fingers closed across the other side of the object.
- Top-down geometric candidates keep the TCP at least `GripperSpec.pad_below_tcp + support_clearance` (8 + 10 mm)
  above the object's bottom. The pads reach 8 mm below the TCP, and on low objects they dragged on the table while
  closing.

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
