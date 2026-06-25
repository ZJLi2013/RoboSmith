"""CAP scenario: three random tabletop objects, pick the apple.

The simplest non-articulated demo: declare one reachable region, scatter three
builtin objaverse assets (apple / mug / lego block) inside it with random,
non-overlapping placement (seed-driven), then pick the apple. This contrasts the
long-horizon articulated drawer task with a single-step rigid pick over a small
cluttered scene — same ~15-line CAP shape, very different task.

Object resting heights come from each asset's geometry via the programmatic
scene backend, so only the region bounds are authored here (no per-object z).

NOTE: authored from asset dimensions; run once on the target gfx942 node to
confirm grasp feasibility before live use.
"""

from robotsmith.cap.intents import intent_sequence, pick
from robotsmith.cap.predicates import object_lifted
from robotsmith.cap.scene_api import layout, obj, region, scene
from robotsmith.cap.task_api import task

# One reachable tabletop patch in front of the arm; the three objects are
# scattered inside it with a minimum pairwise spacing so they neither overlap
# nor stack. The seed (passed at generation time) drives the exact placement.
table_zone = region(
    "table_zone",
    xy_bounds=((0.40, -0.20), (0.58, 0.20)),
    min_distance=0.14,
)

scenario_layout = layout(regions=[table_zone])

scenario_scene = scene(
    "pick_one_of_three",
    layout=scenario_layout,
    objects=[
        obj("apple", asset="apple_01", region="table_zone"),
        obj("mug", asset="mug_02", region="table_zone"),
        obj("block", asset="lego_02", region="table_zone"),
    ],
)

scenario_task = task(
    "pick_one_of_three",
    scene=scenario_scene,
    instruction="Pick up the apple from the table.",
    success=object_lifted("apple", z_margin=0.12),
)

scenario_intents = intent_sequence([
    pick("apple"),
])


def build():
    return scenario_scene, scenario_task, scenario_intents
