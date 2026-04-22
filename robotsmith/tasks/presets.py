"""Built-in task presets."""

from robotsmith.orchestration.skills import Skill
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
    skills=[Skill("pick", "cube", "block")],
    motion_type="pick",
))

place_cube = _register(TaskSpec(
    name="place_cube",
    instruction="Pick up the cube and place it at the target",
    scene="tabletop_simple",
    contact_objects=["cube", "target", "table"],
    success_fn="object_in_container",
    success_params={"object": "cube", "container": "target", "xy_threshold": 0.06},
    skills=[
        Skill("pick", "cube", "block"),
        Skill("place", "target", "block"),
    ],
    motion_type="pick_and_place",
))

mug_in_bowl = _register(TaskSpec(
    name="mug_in_bowl",
    instruction="Place the mug in the bowl",
    scene="tabletop_simple",
    contact_objects=["mug", "bowl", "table"],
    success_fn="object_in_container",
    success_params={"object": "mug", "container": "bowl"},
    skills=[
        Skill("pick", "mug", "mug"),
        Skill("place", "bowl", "mug"),
    ],
    motion_type="pick_and_place",
))

stack_blocks = _register(TaskSpec(
    name="stack_blocks",
    instruction="Stack the red, green, and blue blocks",
    scene="tabletop_simple",
    contact_objects=["block_red", "block_green", "block_blue", "table"],
    success_fn="stacked",
    success_params={"objects": ["block_red", "block_green", "block_blue"]},
    skills=[
        Skill("pick",  "block_red",    "block"),
        Skill("place", "stack_center", "block", {"place_z": 0.15}),
        Skill("pick",  "block_green",  "block"),
        Skill("place", "stack_center", "block", {"place_z": 0.19}),
        Skill("pick",  "block_blue",   "block"),
        Skill("place", "stack_center", "block", {"place_z": 0.23}),
    ],
    motion_type="pick_and_place",
    is_stack=True,
    n_stack=3,
))
