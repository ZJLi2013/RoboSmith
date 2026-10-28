# RoboSmith design

RoboSmith turns **assets plus a declarative task** into **successful expert episodes recorded as a LeRobot
dataset**. You write *what* should happen; RoboSmith decides *how*: it builds the scene, plans a grasp for every
episode, generates a joint trajectory, rolls it out in Genesis, and keeps the success verdict next to the data.

It is data infrastructure, not a policy-training framework. The one training entry, `scripts/train_smolvla.py`,
consumes the dataset RoboSmith writes.

## Data flow

```text
asset library ──► scenario YAML (scene relations + task + success) + teacher plan
                      │  scenario/load.py solves relations; scenario/adapters.py lowers to the runtime
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
| Scenario → runtime | `robotsmith/scenario/load.py`, `robotsmith/scenario/adapters.py` |
| Grasp → execution | `GraspPlan`, including its waypoints |
| Execution → motion backend | the `MotionPlanner` protocol (`robotsmith/motion/`) |
| Simulator / recorder | `robotsmith/sim/sim_env.py`, `robotsmith/sim/recorder.py` (the only adapters) |

## Code layout

```text
robotsmith/
├── assets/            # Asset / AssetMetadata / AssetLibrary; producers/ generates assets
├── scenario/          # YAML scenarios: schema, relation solver, success, lowering
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

An asset is a directory `assets/<id>/` (`model.urdf`, `visual.*`, `collision.obj`, `provenance.json`,
`asset.json`, `checks.jsonl`); `asset.json` names its `kind` and `category`, and the id carries no scene name, so one
asset can appear in any scene. A recorded room is a layout `scenes/<name>.yaml` that lists asset ids and their
poses; scenarios name it instead of copying it. New assets come from the
producers in `robotsmith/assets/producers/`: an LLM agent writing a Blender script for one object (`blender_item`,
the default), a snapshot of a reconstructed room (`aha3d_snapshot`), or an image-to-3D backend (TRELLIS.2 or
Hunyuan3D) followed by `mesh_to_urdf`. Producers write `assets/<id>/` as a draft; the backends import their model
repositories lazily and run in their own image, not in the Genesis one.

`asset.json` holds the `kind` (`rigid`, `articulated`, `fixture`), the source and build file lists with a hash of the
build files, the source's SPDX `license` and `attribution`, and annotations; each annotation records who wrote it (`human`, `harness`, `agent:<model>`,
`check:<name>@<version>`, or `legacy` for values migrated without a known writer). `checks.jsonl` is append-only: one
record per check run, bound to the asset revision and build hash. A record whose hash no longer matches the files does
not count. Mass, inertia and joint limits live only in the URDF. Grasp policy buckets are a grasp-module result and
live in the sidecar `assets/grasp_policy_buckets.json`. The loader assembles all of this into `AssetMetadata`.

What an asset can do follows from its annotations, not from its category:

| Asset kind | Unlocked by | Actions | Success predicates |
|---|---|---|---|
| rigid | an admitted graspable mesh | `pick`, `place` | `inside` / `on` (`object_in_box`), `lifted`, `stacked`, `aligned` |
| articulated | a movable URDF joint + `handles` | `open`, `close` | `joint_opened`, `joint_closed` |
| supporter / fixture | `kind: fixture` or any movable joint | none (it is a place target or an obstacle) | — |

- **Build files are the geometry.** The URDF and meshes are in metres with `+Z` up, so the asset frame is the
  upright object frame and the scene adds only a position and a yaw. Point clouds, grasp gates, and support
  clearances are all computed from the files as they are; there is no scale or upright annotation to apply.
- **Gate.** An asset is admitted when its current build has a passing `llm_review` record and its `source.license`
  may be redistributed (CC0-1.0, CC-BY-4.0, MIT, Apache-2.0, BSD-2/3-Clause), and it duplicates no admitted asset;
  otherwise it is a draft, and `store.draft_reasons` says which condition fails. Scenarios refuse drafts (`AssetNotAdmitted`), the asset package
  carries admitted assets only, and `scripts/assets/list_assets.py` shows each asset's status. The table under a
  scenario is a solid block sized by the scenario and the ground is Genesis' plane; neither is an asset.
- **Review.** `robotsmith/assets/review/` packs what the reviewer sees: the request stating the asset's use, renders
  in the simulator's frame, readings derived from the build files, and the URDF. The reviewer (an LLM) writes its own
  simulator probes, runs them, and returns a verdict; `admit` appends it to `checks.jsonl` as `llm_review`, bound to
  the revision and build hash it judged. Moving files does not admit anything.
- **Duplicates.** `admit` also runs `robotsmith.assets.dedup.check`, which records once per build, as a
  `dedup_check`, the admitted assets the new build may duplicate: the same build hash, or the same category, kind,
  movable joints and place targets with every edge within 20%. The same build hash refuses the asset outright; any
  other candidate needs a `dedup_review` verdict (`variant` or `duplicate`) from a reviewer who saw both, bound to
  both sides' revision and build hash. The gate reads only these records, so admission is the same on a fresh clone
  without build files; a candidate rebuilt or demoted after the check no longer counts. There is no embedding index.
- **Search.** `AssetLibrary.search_assets(query, category=, size_cm=)` returns admitted assets of a category,
  nearest size first, or those sharing a query word when no category is given; `list_assets.py --category --size
  --query` prints the same. An agent maps a request to a category from the catalog and searches before asking a
  producer for a new asset.
- **Package.** Build files are not in git; git tracks `asset.json`, `checks.jsonl` and source files. The build files
  of admitted assets live in a Hugging Face dataset (`lidavidsh/robosmith-assets`) at `<id>/<build_hash>/`, so a
  published path never changes and `asset.json` is the lock. `python -m robotsmith.assets.fetch` downloads files that
  are missing or do not match the hash and refuses a download that does not hash to the recorded value;
  `python -m robotsmith.assets.package publish` uploads new builds and regenerates the attribution table on the
  dataset card. Storage is addressed by build hash, not by scene or category, so reorganising the library never
  moves published files.

Geometry is derived from the URDF and meshes at load (`robotsmith/assets/derive.py`), not stored: `size_cm` is the
visual-geometry extent, the task joint is the movable joint, and a handle or place target only names a link or
collision box whose position the loader reads at zero joint configuration. Annotations keep only the semantic
choices: which link is the `handle`, which boxes are `place_targets` (with an optional insert `approach`).

### Scenario

A scenario is a YAML file: the scene as objects with relations (`on`, `near`, `clear_of`, `facing`, at most one
`anchor_xy` per fixture) and the task (instruction + success tree). It is data, validated against the JSON Schema in
`robotsmith/scenario/load.py`; nothing in it executes. Examples live in [`scenarios/`](../scenarios/); the authoring
contract for agents is [`scenarios/AGENT_PROMPT.md`](../scenarios/AGENT_PROMPT.md).

- **Relations are solved per seed** (`robotsmith/scenario/relations.py`) from asset geometry only: visual bounds,
  collision boxes, place targets and prismatic joints. `near: robot` samples the Franka workspace annulus, so one
  scenario is a distribution of layouts; `variations` swap fields per seed and `split` reserves seed ranges for
  playground and test.
- **Success** is a tree of `all_of` / `any_of` / `not` over predicates that read asset geometry, not hand-tuned
  thresholds: `inside` / `on` test the object against a place target's collision box (riding its joint). The tree is
  meant to be hidden from the executor.
- **Teacher plans** (`scenarios/teacher/<name>.yaml`) hold the scripted `pick` / `place` / `open` / `close` steps
  for expert rollouts; they never write waypoints, control the gripper, or call Genesis. Place targets on an
  articulated fixture follow the joint as it opens.

### Grasp: generate, then only subtract (GraspGen, `--grasp-planner learned`)

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

### Grasp: geometric candidates, chosen by trial (default)

Single-asset validation and scenario runs build candidates from the settled object's oriented box and `GripperSpec`
only (`grasp/geometric.py`), try them in simulation and keep the first one that lifts the object. Single-asset
validation tries each from the placed start state; a scenario tries each from the live scene (snapshot, replay,
restore: `SimEnv.trial_from_here`), so drawers already opened and objects already moved stay as they are. Two Panda
facts shape the candidates:

- **The fingers close along the hand's local Y axis** (`panda_finger_joint*` `axis="0 1 0"`), so the hand rotation
  is `[finger × approach, finger, approach]`.
- **The finger pads reach 8 mm below the TCP** (`GripperSpec.pad_below_tcp`), so top-down TCP height is floored at
  object bottom + `pad_below_tcp` + `support_clearance` (10 mm). Without the floor, the pads of a 2.4 cm tall object
  end 4 mm above the table and, with about 3 mm of tracking sag, drag on it while closing.

Holding is still open. The fingers are position-controlled, and a closing target of 0.01 m per finger does not hold
a generated 0.22 kg stapler although both pads touch it; it lifts with `finger_closed=0.0`. A generated 0.38 kg tape
dispenser does not lift even at 0.0, with both pads on it.

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

Articulated assets load as URDF joint bodies, reset to joint value 0 (or the scene's `joint_init`), and expose joint values in
`env_state.joint_positions`. `open` / `close` drag the handle along the joint axis ("open" means the joint value
increases) by the URDF joint travel. Long-horizon tasks such as `open → pick → place → close` re-anchor
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
- [Blender](https://www.blender.org/): LLM agents build single-object assets as Blender scripts.
- [Hugging Face Hub](https://huggingface.co/docs/hub/datasets): the dataset that stores build files by hash.
- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) and [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1):
  image-to-3D backends for generated assets.
