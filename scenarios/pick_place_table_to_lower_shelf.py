"""CAP scenario: pick a die off the TABLE and side-insert it onto the LOWER shelf.

Isolation variant of ``pick_place_onto_supporter``. There the die starts on the
stand's UPPER shelf, so the *pick* itself is a constrained reach-in that adds
GraspGen stochasticity and near-singular wrist poses — noise that masks the one
problem we actually want to solve: the *place* side-insert into the occluded
lower shelf. Here the die instead starts on open tabletop in front of the
stand's open mouth, making the pick a trivial top-down tabletop grasp. The place
half is identical to ``pick_place_onto_supporter``: the lower shelf sits directly
under the upper one, so the arm cannot drop straight down — the gripper must
thread into the gap from the open side, which only a collision-aware planner
(rocRobo) does reliably.

Net effect: pick = noise-free / deterministic-ish, place = the exact same
occluded side-insert challenge. This lets us iterate milestone A (approach-axis
pre-insert standoff author) + ``planned`` side-insert (docs/features/
supporter_feature.md, place_insertion_strategy.md) without pick failures
confounding the signal.

Asset geometry (stand local frame, origin = footprint_center_floor):
  - shelf_lower surface: (0, 0, 0.054)
  - shelf_upper surface: (0, 0, 0.304)   -> 0.25 m gap between shelves
  - shelf depth (X, open side): 0.14 m
Authored at true metric scale (metadata metric_scale 1.0), so SCALE = 1.0.

The lower-shelf drop point and its side-insert approach come from the
supporter's ``shelf_lower`` placement affordance (asset metadata), resolved
against the stand's live pose — no hand-computed world coords / ``place_z``.
"""

from robotsmith.cap.intents import intent_sequence, pick, place
from robotsmith.cap.predicates import object_in_container
from robotsmith.cap.scene_api import layout, obj, on_placement, scene, target_position
from robotsmith.cap.task_api import task

# Must match assets/articraft/two_layer_supporter metadata "metric_scale".
SCALE = 1.0

STAND_X, STAND_Y, TABLE_TOP = 0.56, 0.0, 0.775

# Die spawn: open tabletop in front of the stand's open mouth (which faces the
# arm, world -X), well clear of the stand footprint (body occupies ~x[0.49,0.63])
# so it's a trivial, unobstructed top-down grasp. Physics settles the rest height
# from the small initial clearance.
DIE_X, DIE_Y = 0.40, 0.0
DIE_SPAWN_Z = TABLE_TOP + 0.03

scenario_layout = layout(regions=[])

scenario_scene = scene(
    "pick_place_table_to_lower_shelf",
    layout=scenario_layout,
    objects=[
        # Same stand, same orientation as pick_place_onto_supporter (yaw=0: open
        # mouth toward world -X = toward the arm; solid rear plate toward +X).
        obj("supporter", asset="two_layer_supporter",
            fixed_position=(STAND_X, STAND_Y, TABLE_TOP), yaw_deg=0.0),
        # Die starts on the TABLE in front of the open mouth (not on a shelf), so
        # the pick is a plain top-down tabletop grasp with no occlusion / no
        # near-singular reach-in. die_01 (~5 cm cube) grasps stably.
        obj("die", asset="die_01",
            fixed_position=(DIE_X, DIE_Y, DIE_SPAWN_Z)),
    ],
    target_positions=[
        # Lower-shelf drop point — anchored to the supporter's own ``shelf_lower``
        # placement affordance (surface point + side-insert approach in asset
        # metadata), resolved against the stand's live pose. No hand-computed
        # world coords here; tracks the stand's yaw automatically.
        target_position(
            "supporter_lower_shelf",
            anchor=on_placement("supporter", "shelf_lower"),
        ),
    ],
    # 3/4 high angle from the front-left (-X open side, -Y), looking down: reveals
    # the horizontal shelf SURFACES while peeking into the open mouth where the
    # die tucks in. Arm stays to the left of the stand, minimal occlusion.
    camera_position=(-0.2, -1.7, 2.1),
    camera_target=(0.56, 0.0, 0.85),
)

scenario_task = task(
    "pick_place_table_to_lower_shelf",
    scene=scenario_scene,
    instruction="Pick the die off the table and place it on the lower shelf.",
    # Target-anchored so xy+z both pin the LOWER shelf: z_tol enforces a two-sided
    # band around the lower-shelf world height (< half the 0.25 m inter-shelf gap)
    # and xy_threshold ~the shelf half-extent, so a die left on the table / dropped
    # off the shelf / never moved all fail.
    success=object_in_container(
        "die", "supporter_lower_shelf", xy_threshold=0.05, z_tol=0.06
    ),
)

scenario_intents = intent_sequence([
    pick("die"),
    # Place onto the lower shelf. No hand-authored place_z: the drop point is the
    # resolved shelf_lower world xyz, lifted by the die's grasp-relative offset
    # captured at pick; the final approach follows the affordance's side-insert
    # axis. The collision-free transit routes the gripper into the gap from the
    # open side.
    place("supporter_lower_shelf"),
])


def build():
    return scenario_scene, scenario_task, scenario_intents
