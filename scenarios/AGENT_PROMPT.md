# Scenario Authoring Prompt

Use this prompt when asking an agent to generate a RoboSmith scenario.

The agent must generate a Python scenario candidate, not runtime control code.
A scenario is:

```text
scenario = SceneSpec + TaskSpec + SkillIntentSequenceSpec
```

## Inputs

The user provides a natural language task request, for example:

```text
Move the die to the goal area while leaving the mug and apple as distractors.
```

The agent may use only the public RoboSmith CAP authoring API listed below.

## Allowed Imports

```python
from robotsmith.cap.intents import intent_sequence, pick, place
from robotsmith.cap.predicates import object_in_container, object_lifted, objects_aligned, stacked
from robotsmith.cap.predicates import all_of, any_of, negate
from robotsmith.cap.scene_api import layout, obj, region, scene, target_position
from robotsmith.cap.task_api import task
```

## Allowed Output Shape

The generated file must define these names:

```python
scenario_scene = ...
scenario_task = ...
scenario_intents = ...


def build():
    return scenario_scene, scenario_task, scenario_intents
```

## Authoring Rules

- Use logical names such as `"die"`, `"mug"`, `"apple"`, and `"goal"` consistently across scene objects, predicates, and intents.
- Use `obj(...)` only for physical objects that should appear in the simulator.
- Use `target_position(...)` for non-physical goals such as `"goal"` or `"drop_zone"`.
- Use `task(...)` to bind the scene, instruction, and success condition.
- Success may be a single predicate, or a composite via `all_of(...)`, `any_of(...)`, `negate(...)` (e.g. `all_of(object_in_container("apple", "bowl"), object_in_container("banana", "bowl"))`).
- Use `intent_sequence([...])` with only `pick(...)` and `place(...)`.
- Prefer named regions over hard-coded object positions unless the task requires an exact fixture.
- Keep the scenario robot-agnostic and runtime-agnostic.

## Reference Examples

Pick one object:

```python
from robotsmith.cap.intents import intent_sequence, pick
from robotsmith.cap.predicates import object_lifted
from robotsmith.cap.scene_api import layout, obj, region, scene
from robotsmith.cap.task_api import task

table_layout = layout(regions=[
    region("left_reachable", ((0.42, -0.14), (0.50, -0.06))),
])

scenario_scene = scene(
    "pick_die",
    layout=table_layout,
    objects=[obj("die", asset="die_01", region="left_reachable")],
)

scenario_task = task(
    "pick_die",
    scene=scenario_scene,
    instruction="Pick up the die.",
    success=object_lifted("die", z_margin=0.05),
)

scenario_intents = intent_sequence([
    pick("die", category="die"),
])


def build():
    return scenario_scene, scenario_task, scenario_intents
```

Move one object to a non-physical target:

```python
from robotsmith.cap.intents import intent_sequence, pick, place
from robotsmith.cap.predicates import object_in_container
from robotsmith.cap.scene_api import layout, obj, region, scene, target_position
from robotsmith.cap.task_api import task

table_layout = layout(regions=[
    region("left_reachable", ((0.42, -0.14), (0.50, -0.06))),
    region("goal_area", ((0.58, 0.08), (0.66, 0.16)), z=0.809),
])

scenario_scene = scene(
    "place_die_at_goal",
    layout=table_layout,
    objects=[obj("die", asset="die_01", region="left_reachable")],
    target_positions=[target_position("goal", region="goal_area")],
)

scenario_task = task(
    "place_die_at_goal",
    scene=scenario_scene,
    instruction="Place the die at the goal.",
    success=object_in_container("die", "goal", xy_threshold=0.04),
)

scenario_intents = intent_sequence([
    pick("die", category="die"),
    place("goal", category="die"),
])


def build():
    return scenario_scene, scenario_task, scenario_intents
```

## Forbidden Output

Do not generate:

- Genesis / simulator calls.
- Waypoints or joint trajectories.
- Gripper open / close commands.
- IK calls.
- Recorder or dataset code.
- Direct legacy `SceneConfig`, runtime `TaskSpec(success=...)`, or `Skill(...)` construction.
- Arbitrary success-check Python functions.

## Validation Contract

The generated candidate should be saved under an experiment output directory:

```text
output/<experiment>/generated_scenario.py
```

Then validate it by importing `build()` and running the normal adapters:

```python
from robotsmith.cap.adapters import to_legacy_scene_config, to_legacy_skills, to_legacy_task_spec

scenario_scene, scenario_task, scenario_intents = build()
legacy_scene = to_legacy_scene_config(scenario_scene)
legacy_task = to_legacy_task_spec(scenario_task)
legacy_task.skills = to_legacy_skills(scenario_intents, scenario_task)
```

Only promote a scenario into `scenarios/` after it validates, lowers, and passes
the intended smoke test.
