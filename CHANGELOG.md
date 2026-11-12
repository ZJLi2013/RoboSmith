# Changelog

All notable user-facing changes are documented here. This project follows Semantic Versioning.

## [Unreleased]

### Added

- The README shows one recorded episode of acceptance families F0, F1 and F5 as a GIF linking to its mp4 in
  `videos/`, each from `scripts/datagen/run_generated_scenario.py --smoke`; F2 has none because its wall hides the
  grasp from the recording camera.
- `robotsmith.motion.joint_planner.JointPlanner`: a lazy RRT-Connect in joint space whose every state and edge point
  is an `ExecutionGate.check_point` and whose path is issued by `verify`. Edges are subdivided by
  `SimEnv.lever_arms` so no point of the robot or held object moves more than 5 mm per step. One budget of 1500
  state checks or 30 s covers the search and the final `verify` (`verify(..., budget=)`); past it the connect
  issues nothing and raises `Refused("search_exhausted")`. `SimEnv.hold(qpos, name, T)` poses a held object without
  stepping. Pick and place plan their free segments with it.
- Six meta actions (`sample`, `move_to`, `move_linear`, `gripper`, `attach`, `detach`): each pick and place runs one
  fixed composition of them through `EpisodeRun`, the only entry. A pick step may give its lift
  (`lift: {direction, dist}`, world +Z 0.10 m by default). `steps_to_skills` checks a step list against the hand it
  starts with: an unknown object or place target, a place with nothing held and a pick while holding are each a
  `PlanInvalid` naming the step, raised before any physics step; YAML and schema errors are raised as they are.
  The relation solver's `SolveError` becomes `PlanInvalid` (with the `SolveError` as its cause) only at this entry,
  and the agent path wraps it as `PlanError` with the same message. A valid step list runs as before: only the
  exception type of an invalid one changed.
- Acceptance families under `scenarios/acceptance/`, each varying one constraint: F0 open table, F1 wall gap,
  F2 over barrier, F4 caged (unsolvable), and the held-out F5. `scripts/acceptance/witness.py` shows each positive
  instance solvable independently of the chain under test; `scripts/acceptance/certify.py --scenario <yaml>` runs the
  chain on any scenario with a `split` and judges every physics step, writing `instances.jsonl` and `summary.json`.
  The objects each instance moves are the plan's pick targets, not a fixed name.
- `scenarios/pick_place_block_on_mat.yaml` and `scenarios/pick_place_mug_on_mat.yaml`: the F0 layout with `block_01`
  and `mug_02` in place of the die.
- `scripts/acceptance/batch.supervise`: the witness and certify batches run in a worker process; an attempt past
  600 s has its process group killed from outside, gets an `error` row, and the batch resumes. This replaces the
  in-process timer, which a simulator step holding the GIL could block.

- Step bindings (`robotsmith.scenario.adapters.to_step_bindings`): each plan step gets the completion condition it
  can be judged by. A `place` binds to the success tree's single positive `on` / `inside` relation for the object the
  preceding `pick` took. A relation whose parent or target disagrees with the place fails at load. Trees with
  `any_of` / `not`, an object placed twice, or several candidate relations stay `unchecked`, as do steps not yet
  migrated. Lowered `on` / `inside` params keep the place-target name, so two shelves of one fixture stay distinct.
  The runtime `TaskSpec` carries `step_bindings`, and `scenario_run_summary.json` records them; a `place` step is
  judged by its binding.

- `robotsmith.scenario_runtime.handoff`: every skill of a scenario episode reports whether it left the world the way
  the next skill needs it (drawer opened, object lifted, object carried until release and resting on the target,
  drawer closed). `scenario_run_summary.json` lists the handoffs and, for a failed episode, the first unfinished step
  with a candidate cause (`motion?`, `grip?`, `release?`), or `unknown` when every handoff passed.
- `robotsmith.assets.producers`: assets written as Blender programs by an agent (`blender_item`) and imported from
  an AHa-3D room snapshot (`aha3d_snapshot`).
- `robotsmith.scenario.relations`: a relation solver that places scenario objects per seed from asset geometry
  (`on`, `near: robot`, `clear_of`, `facing: robot`, fixture `anchor_xy`), and the `object_in_box` predicate behind
  `inside` / `on`, which tests an object against a place target's collision box.
- `scripts/assets/list_assets.py`: the asset catalog for scenario authors. Given only `scenarios/AGENT_PROMPT.md` and
  this catalog, GPT-6 Astra wrote 8 of 9 one-sentence requests as valid YAML scenarios with the intended success
  trees and declined a request for an asset the library does not have.
- `AssetLibrary.search_assets(query, category=, size_cm=)`: admitted assets of a category, nearest size first, or
  sharing a query word when no category is given. `list_assets.py` shows each asset's category and takes
  `--category`, `--size` and `--query`.
- Grasp repair loop for geometric candidates (`robotsmith.grasp.diagnostics`). `validate_asset_picks --diagnose-dir`
  writes, when every episode fails, a request for a diagnosis agent: the evidence of that run (settled pose, simulator
  AABB, horizontal sections, candidates and trials, table height and gripper limits) plus a playbook of known failure
  signatures. `python -m robotsmith.grasp.diagnostics.tool` is the agent's `try_grasp`; `...diagnostics.writeback`
  re-trials the agent's repair once and, if it lifts, writes it to the asset.
- `GeometricGraspPlanner` puts grasps written back to the asset (`grasp_candidates` annotation) ahead of its own
  candidates, while a passing `trial_grasp` check is current for the asset's revision, build files and embodiment
  (`robotsmith.grasp.annotations`).

### Changed

- Pick and place are planned by `skills/rule_policy.RulePolicy` over one `skills/manipulation_tools.ManipulationTools`
  held by `EpisodeRun`; `runner.plan_skill` passes the policy whether a place follows, the following single-phase
  place and the trial explicitly, in place of `ctx.skills` / `ctx.index` and `scene_state["grasp_trial"]`.
  `skills/pick.py` and `skills/place.py` are gone, and `SkillContext` no longer has `skills`, `index` or `planner`.
  F0 playground seeds 0–4 give the same outcome, certification and per-stage physics steps before and after.
- Pick and place pass the execution gate as a whole before any physics step: free segments are planned by
  `JointPlanner`, the rest is `move_linear` and `gripper`, and what cannot pass is refused with a reason and zero
  steps. Pick opens the fingers before approaching and lifts straight up 0.10 m; place detaches before releasing.
- The Franka loads fully gravity-compensated (`gravity_compensation=1.0`), so it holds the configuration the gate
  checked instead of sagging under gravity.
- `ExecutionGate.check_point` is the one point check: joint limits, FK residual where a pose is asked for,
  self-collision between links that are not joined, and contacts the gate does not permit; `verify` runs it on every
  point. A motion that holds an object takes an explicit `Holding(target, t_hand_object)`, checked where it attaches.
  A `VerifiedMotion` is read-only and only the gate issues one: it and `Holding` keep their arrays as bytes, and
  each read of `trajectory` / `t_hand_object` is a fresh read-only view. A step stopped before any physics raises
  `Refused(label, evidence)` (`search_exhausted`, `unsupported`, `infeasible`), and a pick whose trials stepped physics
  and lifted nothing raises `ExecutionFailed`; both replace `Infeasible`. The episode result field `infeasible` is now
  `stopped`, with `cause`, `evidence` and `steps`.
- Plan files live in `plans/<same name>.yaml` beside the scenario, with the keys `name` and `steps` (a `pick`
  step may carry a `lift`); `--planner file` runs their steps (was `teacher`).
- Scenario `pick` steps are judged by the new `grasped` predicate: two different finger links touch the target and
  it rose at least 2 cm from the step's own start, read before the step is planned (`step_start_positions` in the
  judged state, separate from the episode start). `collect_world_state` adds `finger_contacts`, the finger links
  touching each entity. Every built-in skill now has a completion condition; `handoff.check` only diagnoses.
- Scenario `open` / `close` steps are judged by `joint_opened` / `joint_closed` on the joint the executor drags,
  at the end of the drag with no hold. `skills.frames.resolve_drag_joint` is the one answer for which joint and
  which endpoint (the upper limit or `distance` for open, the lower limit for close), used by both the drag and the
  load-time binding; `close` now drags to a nonzero lower limit instead of to 0. The handoff check keeps the joint
  travel as a diagnostic. A missing joint observation raises `JudgeInputError`.
- A scenario `place` is judged by the same `on` / `inside` predicate as the task: after its last phase the arm holds
  for 0.5 s, recorded into the episode, and the step's load-time binding is evaluated on that state. The step keeps
  its verdict, binding and judged state; an unchecked binding claims nothing, and the old 8 cm landing threshold is
  gone (the per-frame samples remain as diagnostics). When the place is the last step, the task judge reuses its
  hold. A failed episode also lists the success-tree leaves that failed (`failed_leaves`). The skill registry is
  `SKILL_DEFS` (`SkillDef(primitive, post, settle_s)`), replacing `SKILL_PRIMITIVES`.
- Judging reads the simulator through one module, `robotsmith/sim/world_state.py`: `collect_world_state` reads one
  step and never advances it; `settle_window` holds the arm at its last command and is the only judging code that
  steps. A zero settle steps nothing. Scenario runs and single-asset validation both judge from it. Missing or
  non-finite observations raise `JudgeInputError`. `scenario_run_summary.json` records `success_tree` and each
  episode's `judge_state`, so a verdict re-judges offline to the same result.
- `on` / `inside` success predicates judge the simulator's state after a 0.5 s settle (the object's world collision
  AABB, contact with the target's asset and none with the robot, no motion in the window) instead of a load-time
  upright bottom offset. A block placed standing on its end now counts as on the shelf, and an object still held,
  bridging a drawer rim or still moving does not. `robotsmith.sim.recorder.evaluate_episode`, which nothing called,
  is removed.
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
  longer executes code. Steps moved to the plan file `scenarios/plans/<name>.yaml`. `robotsmith.cap` is renamed
  `robotsmith.scenario`. The four scenarios were migrated; three success judgements change on purpose: a die on the
  drawer cabinet's top is no longer inside the drawer, anywhere on the lower shelf now counts, and `near: robot`
  samples a wider initial layout than the old hand-written regions.
- `assets/objects/` keeps only the assets the scenarios load: `apple_02`, `block_01`, `die_01`, `mug_02`, `plane`,
  `table_simple`. `apple_02` and `block_01` are built in Blender by GPT-5.6 Sol and admitted on an agent review; they
  replace the Objaverse `apple_01` (lying on its side) and `lego_02` (a minifigure, not a block).
- `README.md` is in English, with the Chinese version in `README.zh-CN.md`. The code layout and the upstream projects
  moved to `docs/design.md`. Its old-chain demo videos are removed: they showed hard-coded waypoints that no longer run.

### Removed

- Scenario and runtime fields nothing read: `TaskSpec.contact_objects` / `episode_length` / `dart_sigma` (so
  `TaskSpec.to_dict()`, and the task metadata a recorded dataset carries, no longer has these keys; `from_dict`
  ignores them in older files), `derive_contact_objects`, `SceneConfig.description` / `robot`,
  `RegionSpec.min_distance` / `keep_out_radius` and `ObjectSpec.pose` (with the `scene_api.region` / `obj` parameters
  and the checks on them), and the uncalled builders `intents.close_`, `predicates.negate`,
  `scene_api.on_articulated` / `on_articulated_opening`. `adapters.resolved_target_positions` is gone too: every YAML
  scenario's place target is anchored, so it was always empty, and `certify.py` and `run_generated_scenario.py` take
  their static target points from `resolve_static_target_positions` alone. Every scenario loads the same apart from
  the removed keys, and F0 playground seeds 0–4 give the same outcome, certification and per-stage physics steps.

- The second task and success-tree types: `scenario.specs.TaskSpec`, `TaskSuccessCfg` / `AllOf` / `AnyOf` / `Not`,
  `to_legacy_task_spec`, `validators.validate_task` / `success_refs` and `scenario/task_api.py`. The scenario loader and
  the `predicates` builders now build `tasks.TaskSpec` and `SuccessNode` directly, and check the tree's names at load
  (`validators.check_task`, same error text). `to_legacy_skills` and `validate_skill_intent_sequence` take the scene,
  and `ScenarioRuntimePackage` has no `scenario_task`; each materialization gets its own copy of the loaded task.
  All 14 scenarios at seeds 0–4 in both load modes load the same apart from the removed keys above; the non-YAML
  single-asset tasks, and the serialization, bindings and errors of any / not / nested all success trees, are
  unchanged; materializing one candidate in both orders neither rewrites nor shares the first task.

- The second scene and target types: `SceneSpec`, `LayoutSpec`, `RegionSpec`, `ObjectSpec`, `TargetPositionSpec`,
  `to_legacy_scene_config`, `adapters.target_frames`, `validators.validate_scene` and `scenario/scene_api.py`
  (`FrameRef` stays). `load_scenario` returns `(SceneConfig, target_frames, TaskSpec, intents)` and refuses a
  duplicate object id with `ValidationError`; `ScenarioRuntimePackage` carries `target_frames` instead of
  `scenario_scene`, and each materialization gets its own copy of the loaded scene. `check_task`, `to_legacy_skills`
  and `validate_skill_intent_sequence` take the scene and the target names. All 14 scenarios at seeds 0–4 in both
  load modes load the same, the single-asset entry's `_make_scene` builds the same config for `die_01`, `mug_02` and
  `block_01`, and F0 playground seeds 0–4 give the same outcome, certification and per-stage physics steps.

- The second step type: `SkillIntentSpec` / `SkillIntentSequenceSpec`, the `scenario/intents.py` builders,
  `to_legacy_skills` and `validators.validate_skill_intent_sequence`, whose checks `steps_to_intents` already made on
  every path. `steps_to_intents` is now `steps_to_skills` and returns `list[Skill]`; `load_scenario` returns the plan's
  skills, `ScenarioCandidate.intents` is `skills`, and `ScenarioRuntimePackage` has no `scenario_intents`.
  `to_step_bindings` and `bindings_from_steps` take `list[Skill]`. Fixed step inputs give the same skills, bindings and
  errors, `_make_task` builds the same single-asset tasks, the 14 scenarios load the same, and F0 playground seeds 0–4
  give the same outcome, certification and steps.

- The candidate / package wrappers: `ScenarioCandidate`, `ScenarioRuntimePackage`, `load_candidate`,
  `materialize_candidate` and `scenario_runtime/loader.py` / `materialize.py`. `load_scenario(path, *, seed,
  agent_plan=False)` returns one `Scenario` (`source_path`, `scene`, `task`, `target_frames`); plan mode fills the
  task's skills and step bindings at load, agent mode leaves them empty and carries every place target. The plan file
  is still read and checked in both modes. `relation_params` and `CONTACT_TOL_M` moved to `scenario.adapters`, so that
  `load` (which now builds the step bindings) depends on `adapters` one way instead of a circular import, and
  `scenario/validators.py` is merged into `scenario.load` (`ValidationError`, `check_object_ids`, `check_task`).
  The 14 scenarios load the same, six kinds of bad input raise the same errors in both modes, loads share no mutable
  objects, and F0 playground seeds 0–4 give the same outcome, certification and steps.

- A plan file's `actions` and their item-by-item translation: `actions_to_intents`, `check_plan`, `load_plan`,
  `pick_part` / `place_part`, `meta_actions.check_part` and `Skill.params["actions"]`. They could only spell the one
  composition the steps already run; `pick_actions` / `place_actions` now build the lists from the candidate and the
  step's `lift`, and the five acceptance plan files keep only their `steps`. A plan file with `actions` is rejected.
  F0 playground seeds 0–4 give the same outcome, certification and per-stage physics steps before and after.

- Grasp and motion fields nothing read: `GraspPlan.waypoints` and `quality`, `motion.types.Waypoint`,
  `GripperSpec.lift`, the `rng` / `object_height` / `scale` parameters of `GraspPlanner.plan`,
  `GeometricGraspPlanner.last_filter_counts` and the pick trace's `filter_counts`, `RuntimeMotionWaypoint.segment_type`,
  `MotionParams.fine_joint_vel` / `descend_steps` / `lift_steps` / `place_descend_steps` / `retreat_steps`, and the
  test-only chain evaluation (`evaluate_motion_chain`, `classify_segments`, `MotionChainMetrics`). `GraspPlan.retreat_pos`
  / `retreat_quat` are optional and only place plans set them. F0 playground seeds 0–4 give the same outcome,
  certification and per-stage physics steps before and after.

- Assets no scenario, scene, test or document loads: the drafts `chair_02` and `drawer_cabinet_d1` (duplicates of
  `chair_01` and `drawer_cabinet_a`), `drawer_cabinet_blender` (the build of `drawer_cabinet_a`), `drawer_cabinet_b`
  and `drawer_cabinet_c` (no license, review failed), and the admitted `drawer_cabinet_a`, a near-duplicate of the
  `drawer_cabinet` the drawer scenarios use.

- The rocRobo motion sidecar and its collision-blind straight-line fallback: it planned in its own simplified world,
  which disagreed with the simulator's geometry. `run_skills`, `plan_segment`, `LIFT` and
  `MotionExecutor.pick` / `place` went with the hard-coded motion templates.
- Image-to-3D asset generation (the TRELLIS.2 and Hunyuan3D backends, `generate_asset`), `mesh_to_urdf`, and the
  `robotsmith.assets.catalog` helpers only they called; `scripts/train_smolvla.py` and the `viz` extra. Opaque meshes
  have no parts or joints to build a simulation asset from, and policy training is out of scope.
- The plan-file keys `teacher` / `plan`; plan files take only `name` and `steps`.
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
- `GripperSpec.width_margin`: geometric candidates close across any horizontal axis up to the full opening
  (`2 × finger_open`, 80 mm). The execution gate already rejects a finger touching the object anywhere on the
  pick's commanded descent; a touch that only happens in replay is a certified violation.

### Fixed

- The gfx942 image pins `tensordict==0.7.2`. Unpinned, the GraspGen layer pulled a release that requires
  `torch>=2.11`, which replaced the ROCm torch 2.6 with a CUDA wheel and broke startup
  (`operator torchvision::nms does not exist`).
- The grasp support-band check spans each finger pad `finger_open` from the grasp centre. `finger_open` is per
  finger, and the check used half of it, so grasps whose lower pad enters the support surface passed P0.
- Geometric grasp candidates close along the Panda hand's local Y axis, the axis its finger joints move on. They
  put the closing direction in X, so the fingers closed across the other side of the object.
- Top-down geometric candidates hold the finger pads inside the object's height, at the centre-of-mass height and
  then as shallow as they fit (`GripperSpec.pad_below_tcp` / `pad_above_tcp`, from the Panda pad geometry), over the
  centre of mass and then the box centre. They used fixed depths below the top over the box centre; on low objects the
  pads dragged on the table while closing, and a box centre pulled off by a handle put both contacts at a pad edge.
  A finger that still meets the support is rejected by the execution gate, so no clearance margin is added.

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
