"""CAP scenario: move an object from the stand's UPPER shelf down to the LOWER shelf.

A non-articulated counterpart to the drawer task that isolates the
collision-free motion story. The ``two_layer_supporter`` is a rigid weldment:
two posts + a rear plate + two stacked shelves, open side facing +X. An object
sits on the upper shelf; the task is to pick it and place it on the lower shelf.
Because each shelf sits directly under the one above, the arm cannot drop
straight down onto the lower shelf — it must approach from the open (+X) side and
tuck under the upper shelf, which only a collision-aware planner (rocRobo) does
reliably. Clean "why collision-free planning" demo without any joints.

Asset geometry (stand local frame, origin = footprint_center_floor, BEFORE scale):
  - shelf_lower surface: (0, 0, 0.074)
  - shelf_upper surface: (0, 0, 0.174)   -> 0.10 m gap between shelves
The asset is loaded at metric_scale 3.0 (see the asset metadata), so the gap
becomes ~0.30 m — a more human-scale reach-in. SCALE below MUST match that
metadata value, since the hand-authored object / target points are scaled here.

NOTE: first-cut geometry, not yet tuned on hardware. At SCALE=3 the upper shelf
is ~0.52 m above the stand base (~1.30 m world on a 0.775 table), which MAY
exceed the Franka's reach — run the snapshot preview + a smoke rollout on the
target gfx942 node, and if the upper shelf is unreachable drop SCALE to ~1.5-2.0
(and the matching metadata metric_scale). Also verify ``place_z`` and that the
success predicate distinguishes the lower shelf from the upper one.
"""

from robotsmith.cap.intents import intent_sequence, pick, place
from robotsmith.cap.predicates import object_in_container
from robotsmith.cap.scene_api import layout, obj, scene, target_position
from robotsmith.cap.task_api import task

# Must match assets/articraft/two_layer_supporter metadata "metric_scale".
SCALE = 3.0

STAND_X, STAND_Y, TABLE_TOP = 0.50, 0.0, 0.775
SHELF_LOWER_Z = TABLE_TOP + 0.074 * SCALE   # lower-shelf top surface, world
SHELF_UPPER_Z = TABLE_TOP + 0.174 * SCALE   # upper-shelf top surface, world

scenario_layout = layout(regions=[])

scenario_scene = scene(
    "pick_place_onto_supporter",
    layout=scenario_layout,
    objects=[
        obj("supporter", asset="two_layer_supporter",
            fixed_position=(STAND_X, STAND_Y, TABLE_TOP)),
        # Apple starts resting on the UPPER shelf; physics settles the exact rest
        # height from the small initial clearance.
        obj("apple", asset="apple_01",
            fixed_position=(STAND_X, STAND_Y, SHELF_UPPER_Z + 0.05)),
    ],
    target_positions=[
        # Lower-shelf drop point (stand base + scaled shelf_lower offset).
        target_position(
            "supporter_lower_shelf",
            fixed_position=(STAND_X, STAND_Y, SHELF_LOWER_Z),
        ),
    ],
    camera_position=(2.6, -2.2, 1.9),
    camera_target=(0.45, 0.0, 0.7),
)

scenario_task = task(
    "pick_place_onto_supporter",
    scene=scenario_scene,
    instruction="Pick the apple off the upper shelf and place it on the lower shelf.",
    # Target-anchored so xy+z both pin the LOWER shelf (the upper shelf shares the
    # same xy, so z is what distinguishes them — verify the predicate honors it).
    success=object_in_container(
        "apple", "supporter_lower_shelf", xy_threshold=0.12, z_margin=0.10
    ),
)

scenario_intents = intent_sequence([
    pick("apple"),
    # Tuck under the upper shelf onto the lower shelf. place_z is the EE-flange
    # height above the table; first-cut for the scaled lower shelf (~0.22 m above
    # the table). Re-tune after the first smoke rollout.
    place("supporter_lower_shelf", place_z=0.45),
])


def build():
    return scenario_scene, scenario_task, scenario_intents
