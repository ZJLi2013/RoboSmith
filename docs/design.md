# RoboSmith design

RoboSmith turns **assets plus a declarative task** into **successful expert episodes recorded as a LeRobot
dataset**. You write *what* should happen; RoboSmith decides *how*: it builds the scene, plans a grasp for every
episode, generates a joint trajectory, rolls it out in Genesis, and keeps the success verdict next to the data.

It is data infrastructure, not a policy-training framework. The one training entry, `scripts/train_smolvla.py`,
consumes the dataset RoboSmith writes.

## Data flow

```text
asset library ──► scenario (CAP: scene + task + skill intents + success)
                      │  cap/adapters.py lowers it to the runtime
                      ▼
                  SimEnv (Genesis + Franka) ── reset, settle
                      │
     per skill:   grasp planner ──► GraspPlan ──► MotionExecutor ──► MotionPlanner (Genesis IK or rocRobo)
                      │
                      ▼
                  rollout + success tree ──► LeRobot dataset, videos, run summary
```

Each arrow is one typed contract, and each layer has one owner in the code:

| Contract | Type / module |
|---|---|
| Asset → scene | `AssetMetadata` (`robotsmith/assets/`) |
| Scenario → runtime | `robotsmith/cap/adapters.py` |
| Grasp → execution | `GraspPlan`, including its waypoints |
| Execution → motion backend | the `MotionPlanner` protocol (`robotsmith/motion/`) |
| Simulator / recorder | `robotsmith/sim/sim_env.py`, `robotsmith/sim/recorder.py` (the only adapters) |

## Code layout

```text
robotsmith/
├── assets/            # Asset / AssetMetadata / AssetLibrary; producers/ generates assets
├── cap/               # declarative task layer: scene, task, success, skill intents
├── scenes/            # SceneConfig / ObjectPlacement, upright pose, Genesis loader
├── tasks/             # TaskSpec, predicates, task presets
├── scenario_runtime/  # scenario load, materialize, segment-level runner
├── grasp/             # learned grasp planner + feasibility gate
├── motion/            # motion planning, rocRobo backend, collision world
├── execution/         # executor: skills to joint trajectories
├── skills/            # pick / place / open / close + registry
├── sim/               # Genesis SimEnv, Franka, LeRobot recorder
└── rollout/           # rollout orchestration, episode layout, video export
```

## Key concepts

### Asset and affordance

An asset is a directory `assets/objects/<name>/` (`model.urdf`, `visual.*`, `collision.obj`, `provenance.json`,
`asset.json`, `checks.jsonl`); articulated assets live in `assets/articraft/<name>/`. New assets come from the
producers in `robotsmith/assets/producers/`: an image-to-3D backend (TRELLIS.2 or Hunyuan3D) followed by
`mesh_to_urdf`. The backends import their model repositories lazily and run in their own image, not in the Genesis
one.

`asset.json` holds the `kind` (`rigid`, `articulated`, `fixture`), the source and build file lists with a hash of the
build files, and annotations; each annotation records who wrote it (`human`, `harness`, `agent:<model>`,
`check:<name>@<version>`, or `legacy` for values migrated without a known writer). `checks.jsonl` is append-only: one
record per check run, bound to the asset revision and build hash. A record whose hash no longer matches the files does
not count. Mass, inertia and joint limits live only in the URDF. Grasp policy buckets are a grasp-module result and
live in the sidecar `assets/grasp_policy_buckets.json`. The loader assembles all of this into `AssetMetadata`.

What an asset can do follows from its annotations, not from its category:

| Asset kind | Unlocked by | Actions | Success predicates |
|---|---|---|---|
| rigid | a graspable mesh with a verified upright pose and a metric scale | `pick`, `place` | `object_in_container`, `object_above`, `stacked`, `objects_aligned` |
| articulated | `task_joints` + `handles` | `open`, `close` | `joint_opened`, `joint_closed` |
| supporter / fixture | `kind: fixture` or any movable joint | none (it is a place target or an obstacle) | — |

Three annotations carry most of the physics:

- **`metric_scale`** is the single owner of scale. Point clouds, grasp gates, and support clearances are all computed
  in metres from it; a wrong scale makes all of them wrong without an error.
- **Frames.** Mesh frame → object frame (canonical, `+Z` up) → scene frame. `canonical_frame` is the import
  correction; `upright` is the pose used at runtime and counts as verified only with a passing `upright` check record,
  written after human visual QA (`scripts/assets/render_asset_table.py`).
- **Gate.** An asset enters default experiments only with a passing `upright` record and a passing
  `mesh_consistency` record. The audit also writes a `gripper_fit` record (8 cm Franka gripper) and sets
  `metric_scale` to its recommendation. Fixtures are exempt: they load with `fixed=True`, are never grasped, and use
  their bounding box for placement collision.

Articulated assets keep their kinematics in the URDF. Annotations only add task semantics: `task_joints`, `handles`,
`joint_init`, and `place_targets`.

### Scenario and CAP

A scenario is `scene + task + skill intents + success`, written in Python with the CAP API
(`robotsmith/cap/`: `SceneSpec`, `RegionSpec`, `ObjectSpec`, `TargetPositionSpec`, `TaskSpec`; helpers in
`cap/scene_api.py` and `cap/task_api.py`). Examples live in [`scenarios/`](../scenarios/); the authoring contract for
agents is [`scenarios/AGENT_PROMPT.md`](../scenarios/AGENT_PROMPT.md).

- **Skill intents** are limited to `pick`, `place`, `open`, `close`. A scenario never writes waypoints, controls the
  gripper, or calls Genesis.
- **Success** is a tree of `all_of` / `any_of` / `negate` over predicates. A new predicate is one function in
  `cap/predicates.py` plus its registration.
- **Target positions** are named points, not physical objects; place targets on an articulated fixture
  (`on_articulated_opening(...)`) follow the joint as it opens.
- `ObjectSpec.joint_init` sets a scene-level initial joint value that overrides the asset's.
- Scenario loading executes Python, so only load trusted scenario files.

### Grasp: generate, then only subtract

Each episode generates candidates on the object as it actually settled, then filters them down to one. Later steps
never add candidates back:

1. **Generate**: GraspGen diffusion on a 2000-point cloud in metres (`grasp/graspgen_wrapper.py`).
2. **Support clearance**: drop grasps whose gripper mesh comes within 1 cm of the object's own support surface
   (`grasp/collision_screen.py::support_clearance_mask`). This runs before top-k.
3. **Geometric rejection**: the grasp point must sit above the support surface, the approach must not come from
   below; keep the top `GRASP_TOP_K` (100).
4. **Expand to waypoints**: pre-grasp (backed off along the approach axis) → grasp → close → retreat.
5. **Feasibility gate** (`grasp/feasibility.py`): the fingers must land in the support band and IK must succeed on
   every segment without a joint jump above 0.8 rad. This gate does not check collisions.
6. **Select**: the best-scoring survivor, or — when the asset's metadata names a `grasp_policy_bucket` — the best
   survivor in that bucket. If nothing survives, the episode records `no_feasible_grasp` instead of executing an
   ungated grasp.

The support surface is the target object's own base, not the table, so objects on shelves or inside drawers reject
grasps that reach in from below. Grasp pose axes: local X opens the fingers, Y is palm height, Z is the approach.

### Motion: segment-level routing

Every skill compiles into a sequence of waypoints executed by `motion/chain.py::execute_motion_chain`. Each segment
is routed by what it is, not by which skill it belongs to:

| Segment | Route | Collision-aware |
|---|---|---|
| finger-only (close / open gripper) | arm locked, fingers move | — |
| contact (final approach, drag handle, descend, side insertion) | Cartesian straight line | no, by design |
| free (approach, lift, transport, retreat) | `plan_motion` when a collision world exists, otherwise IK + interpolation | yes, with rocRobo |

Contact segments are never motion-planned: near contact, a planner's detours are worse than a short controlled line.
A pick whose Cartesian segment jumps IK branches is rejected rather than executed.

### Motion backends and the collision world

`ROCROBO_BACKEND` selects the backend:

- **off** → `GenesisBackend`: IK only; `plan_motion` always fails, so free segments use interpolation.
- **on** → `RocRoboBackend`: collision-free IK and trajectory optimisation from
  [rocRobo](https://github.com/ZJLi2013/rocRobo) (JAX). JAX and PyTorch do not share a ROCm process well, so rocRobo
  runs in a sidecar container and RoboSmith talks to it over JSONL through `docker exec`.

With a collision-aware backend, `motion/rocrobo_world.py::build_obstacle_world` builds the world from live poses:
fixture collision boxes plus a ground half-space, with articulated parts placed at their current joint value
(prismatic joints only). During `place`, the held object is attached to the hand as a set of spheres. rocRobo plans
the 7 arm joints; the client adds the two finger joints back.

If `plan_motion` fails, the segment falls back to a straight line **without collision checks**; the log line
`straight-line fallback` marks it.

### Articulation

Articulated assets load as URDF joint bodies, reset to `joint_init`, and expose joint values in
`env_state.joint_positions`. `open` / `close` drag the handle along the joint axis ("open" means the joint value
increases), with travel scaled by `metric_scale`. Long-horizon tasks such as `open → pick → place → close` re-anchor
every segment on the live simulation through `skills/frames.py::resolve_frame`, so a place target inside a drawer
moves with the drawer.

### Rollout and data

Per episode: reset objects to their resolved positions, settle for `settle_steps` (30), record the settled poses as
the initial state, run the skills, then evaluate the success tree. The run writes:

- a LeRobot dataset with the `up` camera; articulated joint values are appended to `observation.environment_state`;
- per-episode videos and `scenario_run_summary.json` (success per episode, diagnostics, skill traces).

## Current limits

- **Collision awareness covers free segments only.** Contact segments are straight lines, and a failed plan falls back
  to an unchecked straight line.
- Articulation supports prismatic joints only; revolute open / close, push / press, and bimanual tasks are not
  implemented.
- One asset per process: `SimEnv.build` initialises Genesis every time, so a second build in the same process fails.
- The first `plan_motion` of a new scene shape compiles for about 10 minutes; later runs reuse the JAX disk cache.
- Data comes from scripted IK replay only; there is no DAgger-style or closed-loop evaluation backend yet.

## Built on

- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis): GPU physics simulation.
- [LeRobot](https://github.com/huggingface/lerobot): dataset format and the SmolVLA training entry.
- [rocRobo](https://github.com/ZJLi2013/rocRobo) on [PyRoki](https://github.com/chungmin99/pyroki): collision-free IK
  and trajectory optimisation.
- [GraspGen](https://github.com/NVlabs/GraspGen): learned grasp proposals (NVIDIA research / evaluation license).
- [Objaverse](https://objaverse.allenai.org/) and [Articraft](https://github.com/mattzh72/articraft): asset sources.
- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) and [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1):
  image-to-3D backends for generated assets.
