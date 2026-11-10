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

`name` and `steps` are required; `actions` is optional, and when present it is what runs.

- `steps` is a skill list: `pick: <id>`, `place: {target: <id>.<target>}`, `open: <id>`, `close: <id>`. Each
  pick / place step runs the public composition below; `open` / `close` drag the asset's handle.
- `actions` is a list of meta actions, one pick part and one place part per object moved. Only pick and place
  have meta actions, so a file with `actions` runs no `open` / `close`: write a task that opens a drawer as
  `steps` alone.

```yaml
name: apple_into_drawer
steps:
  - open: drawer
  - pick: apple
  - place: {target: drawer.drawer_opening}
  - close: drawer
```

### The six meta actions

| Action | Arguments | Effect |
|---|---|---|
| `sample` | `{kind: grasp, object, as}` or `{kind: place, object, surface, grasp, as}` | binds candidates under the name `as`: a grasp binding has poses `pre` / `grasp`, a place binding `pre` / `place` / `retreat` |
| `move_to` | `<binding>.<pose>` | a planned joint-space move to that pose |
| `move_linear` | `<binding>.<pose>`, or `{direction: [x, y, z], dist}` (world frame, m) | a straight line |
| `gripper` | `{width: open \| close, target}` | opens or closes the fingers; `target` is the object operated on |
| `attach` | `<grasp binding>` | the object moves with the hand from here on |
| `detach` | `<place binding>` | the object stays on the place binding's surface |

Each part runs in one order; another order is rejected:

```text
pick:  sample(grasp) · gripper(open) · move_to(<g>.pre) · move_linear(<g>.grasp) · gripper(close, target) · attach(<g>) · move_linear({direction, dist})
place: sample(place, grasp: <g>) · move_to(<p>.pre) · move_linear(<p>.place) · detach(<p>) · gripper(open, target) · move_linear(<p>.retreat)
```

What can vary: the object, the place surface, the binding names, and the direction and distance of the lift.
The candidates come from the scene's geometry, so a plan holds no positions or heights.

Binding rules, checked when the file loads:

- The actions start with a `sample`, and each part's references name the binding its own `sample` made.
- A place `sample` names the grasp binding it puts down (`grasp:`) and that grasp's `object`.
- A `gripper` with a `target` names the part's object; a close always names it.
- One object is held at a time: attach once, detach only what is held, and end holding nothing.

A pick and its place are one composition: both are checked before the first physics step, and the first grasp
that passes is committed.

### Example: the die onto the mat

The acceptance families' plan file (`scenarios/acceptance/plans/f0_open.yaml`):

<!-- example: f0_open -->
```yaml
name: acc_f0_open
steps:
  - pick: die
  - place: {target: mat.top}
actions:
  - sample: {kind: grasp, object: die, as: g}
  - gripper: {width: open}
  - move_to: g.pre
  - move_linear: g.grasp
  - gripper: {width: close, target: die}
  - attach: g
  - move_linear: {direction: [0, 0, 1], dist: 0.10}
  - sample: {kind: place, object: die, surface: mat.top, grasp: g, as: p}
  - move_to: p.pre
  - move_linear: p.place
  - detach: p
  - gripper: {width: open, target: die}
  - move_linear: p.retreat
```

Its `steps` alone run the same two parts; the `actions` decide the lift (here 0.10 m up).

### `PlanInvalid` messages and fixes

| Message contains | Fix |
|---|---|
| `the actions must start with a sample` | begin with the pick part's `sample` |
| `names no candidate sampled earlier` | sample the binding before referring to it, or fix the name's spelling |
| `names candidate 'g', not this part's 'p'` | refer to the current part's binding (`p.pre`, not `g.pre`) |
| `is not a pose of a place candidate` | use that binding's poses: grasp `pre` / `grasp`, place `pre` / `place` / `retreat` |
| `gripper target ... is not this part's object` / `close names no target` | name the object being picked or placed |
| `puts down 'mat', but its grasp 'g' holds 'die'` | set the place `sample`'s `object` to the grasped object |
| `a pick part runs ...; the first version runs ...` | put the part's actions in the order above |
| `must place what is held` / `ends holding` / `one grasp candidate` | one attach per pick, one detach per place |
| a schema message naming an unknown key (e.g. `push`) | only the six actions exist |

## Validate

```bash
python scripts/datagen/run_generated_scenario.py --scenario output/<exp>/<name>.yaml --materialize-only
```

This validates the schema, solves the relations for `--seed`, loads the plan file
(a `PlanInvalid` names the first bad action) and lowers the scenario; it needs no GPU. Promote it into `scenarios/` only after it validates and passes a smoke run.
