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
    - {id: apple, asset: apple_01, on: support, near: robot, clear_of: [drawer.swept]}
task:
  instruction: Put the apple in the drawer.
  success:
    inside: {object: apple, container: drawer.drawer_opening}
variations:                          # optional: <object id>.<field>: [choices], one per seed
  apple.asset: [apple_01, die_01]
split: {playground: [0, 9], test: [100, 149]}   # optional seed ranges
```

`asset` is an asset id from `assets/`. `<id>.<target>` names one of that asset's
`place_targets` (e.g. `drawer_opening`, `shelf_lower`, `shelf_upper`).

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

## Teacher plan (optional)

Scripted rollouts read `scenarios/teacher/<same name>.yaml`:

```yaml
teacher: apple_into_drawer
steps:
  - open: drawer
  - pick: apple
  - place: {target: drawer.drawer_opening}
  - close: drawer
```

## Validate

```bash
python scripts/datagen/run_generated_scenario.py --scenario output/<exp>/<name>.yaml --materialize-only
```

This validates the schema, solves the relations for `--seed` and lowers the
scenario. Promote it into `scenarios/` only after it validates and passes a smoke run.
