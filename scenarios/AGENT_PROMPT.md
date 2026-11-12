# Scenario Authoring Prompt

Use this prompt when asking an agent to write a RoboSmith scenario.

A scenario is one YAML file: the scene (objects and their relations) and the task
(instruction + success tree). It contains no positions beyond one optional planar
anchor per fixture, no robot actions and no code. The full schema is `SCHEMA` in
[`robotsmith/scenario/load.py`](../robotsmith/scenario/load.py).

## Shape

```yaml
scenario: apple_into_drawer          # id
robot: franka
scene:
  support: table_simple
  objects:
    - {id: drawer, asset: drawer_cabinet, on: support, anchor_xy: [0.80, 0.0], facing: robot}
    - {id: apple, asset: apple_02, on: support, near: robot, clear_of: [drawer.swept]}
task:
  instruction: Put the apple in the drawer.
  success:
    inside: {object: apple, container: drawer.drawer_opening}
variations:                          # optional: <object id>.<field>: [choices], one per seed
  apple.asset: [apple_02, die_01]
split: {playground: [0, 9], test: [100, 149]}   # optional seed ranges
```

`asset` is an asset id from `python scripts/assets/list_assets.py`, which also lists
each asset's place targets and movable joints. `<id>.<target>` names one of that
asset's place targets (e.g. `drawer_opening`, `shelf_lower`, `shelf_upper`). If the
request needs an asset that is not listed, say so instead of inventing an id.

To work in a recorded room, write `scene: <name>` with a layout from `scenes/` (e.g.
`scene: office48`) instead of the `scene:` block, and refer to that layout's object ids in the task.

## Relations

| Field | Meaning |
|---|---|
| `on: support` | rests on the tabletop |
| `on: <id>.<target>` | rests on a static place-target surface of another object, centred |
| `anchor_xy: [x, y]` | planar position, for fixtures only |
| `near: robot` | sampled in the robot's workspace (per seed) |
| `clear_of: [<id> \| <id>.swept]` | no xy overlap with the object, or with the volume its drawer sweeps |
| `facing: robot` | the asset's front (drawer pull direction, or the open side of a shelf) faces the robot |
| `yaw_deg` | explicit yaw, only for assets with no front direction |
| `joint_init` | initial joint values, default 0 |

## Success predicates

Combine with `all_of: [...]`, `any_of: [...]`, `not: {...}`.

| Predicate | True when |
|---|---|
| `inside: {object, container: <id>.<target>}` | object rests in that box (follows the drawer joint) |
| `on: {object, surface: <id>.<target>}` | object rests on that surface |
| `lifted: {object, height}` | object is `height` m above its start |
| `joint: {object, joint, at: open \| closed, tol}` | joint within `tol` of its limit |
| `stacked: {objects}` / `aligned: {objects, axis}` | ordered stack / line |

The success tree is hidden from the executor; write it so that states which do
not complete the task are false.

## Plan file (optional)

`--planner file` runs the plan file `plans/<same file name>` beside the scenario. This section is the one
reference for writing it.

`name` and `steps` are required, and nothing else is allowed.

- `steps` is a skill list: `pick: <id>`, `place: {target: <id>.<target>}`, `open: <id>`, `close: <id>`. Each
  pick / place step runs the composition below; `open` / `close` drag the asset's handle.
- A pick may add `lift: {direction: [x, y, z], dist}` on the next line: the straight move after the close, in the
  world frame, `dist` in m and above 0. Without it the lift is 0.10 m straight up.

```yaml
name: apple_into_drawer
steps:
  - open: drawer
  - pick: apple
  - place: {target: drawer.drawer_opening}
  - close: drawer
```

### What a pick and a place run

Each runs one fixed composition of six meta actions; the step chooses only the object, the place target and the
pick's lift:

```text
pick:  sample(grasp) · gripper(open) · move_to(pre) · move_linear(grasp) · gripper(close) · attach · move_linear(lift)
place: sample(place) · move_to(pre) · move_linear(place) · detach · gripper(open) · move_linear(retreat)
```

| Action | Effect |
|---|---|
| `sample` | candidates from the scene's geometry: a grasp has poses `pre` / `grasp`, a place `pre` / `place` / `retreat` |
| `move_to` | a planned joint-space move to that pose |
| `move_linear` | a straight line to that pose, or along the lift |
| `gripper` | opens or closes the fingers on the step's object |
| `attach` | the object moves with the hand from here on |
| `detach` | the object stays on the place target |

The candidates come from the scene's geometry, so a plan holds no positions or heights. A pick and its place are
one composition: both are checked before the first physics step, and the first grasp that passes is committed.

### Example: the die onto the mat

The acceptance families' plan file (`scenarios/acceptance/plans/f0_open.yaml`) with its lift written out:

<!-- example: f0_open -->
```yaml
name: acc_f0_open
steps:
  - pick: die
    lift: {direction: [0, 0, 1], dist: 0.10}
  - place: {target: mat.top}
```

Without the `lift` line it runs the same.

### `PlanInvalid` messages and fixes

The steps are checked in order, from the hand they start with, when the file loads; each message starts with
`step <i>:`:

| Message contains | Fix |
|---|---|
| `place <target> with nothing held` | pick the object before placing it |
| `pick <object> while holding <held>` | place what is held before the next pick |
| `unknown object '<id>'` | use an object id from the scene |
| the place target's name | use a place target the object's metadata lists |

A key the schema does not know (e.g. `actions` or `push`) is a jsonschema `ValidationError`, raised before these
checks: only `name` and `steps` are keys, and only `pick` (with `lift`), `place`, `open` and `close` are steps.

## Validate

```bash
python scripts/datagen/run_generated_scenario.py --scenario output/<exp>/<name>.yaml --materialize-only
```

This validates the schema, solves the relations for `--seed`, loads the plan file
(a `PlanInvalid` names the first bad step) and lowers the scenario; it needs no GPU. Promote it into `scenarios/` only after it validates and passes a smoke run.
