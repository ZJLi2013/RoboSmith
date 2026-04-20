"""Built-in task presets."""

from robotsmith.tasks.task_spec import TaskSpec

TASK_PRESETS: dict[str, TaskSpec] = {}


def _register(spec: TaskSpec) -> TaskSpec:
    TASK_PRESETS[spec.name] = spec
    return spec


pick_cube = _register(TaskSpec(
    name="pick_cube",
    instruction="Pick up the cube",
    scene="tabletop_simple",
    contact_objects=["cube", "table"],
    success_fn="object_above",
    success_params={"object": "cube", "reference": "table", "z_margin": 0.05},
    ik_strategy="pick",
))

mug_in_bowl = _register(TaskSpec(
    name="mug_in_bowl",
    instruction="Place the mug in the bowl",
    scene="tabletop_simple",
    contact_objects=["mug", "bowl", "table"],
    success_fn="object_in_container",
    success_params={"object": "mug", "container": "bowl"},
    ik_strategy="pick_and_place",
))

stack_blocks = _register(TaskSpec(
    name="stack_blocks",
    instruction="Stack the red, green, and blue blocks",
    scene="tabletop_simple",
    contact_objects=["block_red", "block_green", "block_blue", "table"],
    success_fn="stacked",
    success_params={"objects": ["block_red", "block_green", "block_blue"]},
    ik_strategy="stack",
))
