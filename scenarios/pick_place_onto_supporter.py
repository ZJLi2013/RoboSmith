"""CAP scenario: move an object from the stand's UPPER shelf down to the LOWER shelf.

A non-articulated counterpart to the drawer task that isolates the
collision-free motion story. The ``two_layer_supporter`` is a rigid weldment:
two posts + a rear plate + two stacked shelves, open side facing +X. An object
sits on the upper shelf; the task is to pick it and place it on the lower shelf.
Because the upper shelf sits directly over the lower one, the arm cannot drop
straight down from above onto the lower shelf — the collision-free planner must
route the gripper into the gap from the open (+X) side, past the upper shelf,
which only a collision-aware planner (rocRobo) does reliably. Clean "why
collision-free planning" demo without any joints.

The inter-shelf gap is 0.25 m (was 0.20 m): the bottom shelf was LOWERED (0.10 ->
0.05) so the upper-shelf pick height is unchanged from the verified-working pick.
The lower shelf sits directly under the upper one, so the die is inserted with a
TILTED ~45 deg side-approach (asset metadata): a pure top-down insert is blocked
by the upper shelf, and a pure horizontal tuck-under drives the wrist through a
near-singular pose that flips IK branches and flings the object (docs/refactor.md
R6). The 45 deg compromise threads in from the open side under the upper shelf,
and the widened gap gives the wrist room to stay off the singularity.

Asset geometry (stand local frame, origin = footprint_center_floor):
  - shelf_lower surface: (0, 0, 0.054)
  - shelf_upper surface: (0, 0, 0.304)   -> 0.25 m gap between shelves
  - shelf depth (X, open side): 0.14 m
The asset is authored at true metric scale (metadata metric_scale 1.0), sized for
a Franka reach-in over an ~80 mm object, so SCALE below is 1.0 (no extra scaling).
The upper shelf sits ~0.30 m above the stand base (~1.08 m world on a 0.775 table),
inside the Franka's dexterous reach.

The lower-shelf drop point and its (now top-down) approach come from the
supporter's ``shelf_lower`` placement affordance (asset metadata), resolved
against the stand's live pose — no hand-computed world coords / ``place_z`` in
this scenario (docs/refactor.md R4).
"""

from robotsmith.cap.intents import intent_sequence, pick, place
from robotsmith.cap.predicates import object_in_container
from robotsmith.cap.scene_api import layout, obj, on_placement, scene, target_position
from robotsmith.cap.task_api import task

# Must match assets/articraft/two_layer_supporter metadata "metric_scale".
SCALE = 1.0

STAND_X, STAND_Y, TABLE_TOP = 0.56, 0.0, 0.775
# Upper-shelf top surface (world) — only used to spawn the die resting on it. The
# LOWER-shelf drop point is NOT hand-computed here anymore: it comes from the
# supporter's own ``shelf_lower`` placement affordance via ``on_placement`` below
# (tracks the stand's pose/yaw; carries the side tuck-under approach).
SHELF_UPPER_Z = TABLE_TOP + 0.304 * SCALE

scenario_layout = layout(regions=[])

scenario_scene = scene(
    "pick_place_onto_supporter",
    layout=scenario_layout,
    objects=[
        # The open shelf side must face the arm so the collision-free transit can
        # route the gripper into the gap. The rendered mesh is effectively flipped vs
        # model.py's "open=+X local" label, so trust the snapshot, not the label:
        # snapshot output/supporter_snap (yaw=0) shows the SOLID rear plate pointing at
        # the +X/+Y overview camera and the OPEN mouth pointing at world -X = toward the
        # arm (arm sits at the stand's -X); yaw=180 (output/supporter_snap_yaw180) turns
        # the open mouth to +X, away from the arm. So yaw=0 is correct.
        obj("supporter", asset="two_layer_supporter",
            fixed_position=(STAND_X, STAND_Y, TABLE_TOP), yaw_deg=0.0),
        # Die starts resting on the UPPER shelf; physics settles the exact rest
        # height from the small initial clearance. die_01 (~5 cm cube, verified
        # upright + top-down grasp buckets) grasps far more stably than the apple.
        obj("die", asset="die_01",
            fixed_position=(STAND_X, STAND_Y, SHELF_UPPER_Z + 0.05)),
    ],
    target_positions=[
        # Lower-shelf drop point — anchored to the supporter's own ``shelf_lower``
        # placement affordance (surface point + top-down approach in asset
        # metadata), resolved against the stand's live pose. No hand-computed world
        # coords here; tracks the stand's yaw automatically.
        target_position(
            "supporter_lower_shelf",
            anchor=on_placement("supporter", "shelf_lower"),
        ),
    ],
    # 3/4 high angle from the front-left (-X open side, -Y), looking down: the
    # elevation reveals the horizontal shelf SURFACES (a pure -Y axis view was too
    # edge-on to see them), while the -X bias peeks into the open mouth where the
    # die tucks in. Arm stays to the left of the stand, minimal occlusion.
    camera_position=(-0.2, -1.7, 2.1),
    camera_target=(0.56, 0.0, 0.85),
)

scenario_task = task(
    "pick_place_onto_supporter",
    scene=scenario_scene,
    instruction="Pick the die off the upper shelf and place it on the lower shelf.",
    # Target-anchored so xy+z both pin the LOWER shelf. The upper shelf shares the
    # same xy, so z is what distinguishes them: z_tol enforces a two-sided band
    # around the lower-shelf world height (< half the 0.20 m inter-shelf gap), and
    # xy_threshold is tightened to ~the shelf half-extent so an apple left on the
    # upper shelf / dropped off the shelf / never moved all fail.
    success=object_in_container(
        "die", "supporter_lower_shelf", xy_threshold=0.05, z_tol=0.06
    ),
)

scenario_intents = intent_sequence([
    pick("die"),
    # Place onto the lower shelf. No hand-authored place_z: the drop point is the
    # resolved shelf_lower world xyz, lifted by the die's grasp-relative offset
    # captured at pick (feature12 §5 W2); the final approach follows the
    # affordance's axis (now top-down, refactor.md R4/R6). The collision-free
    # transit still routes the gripper into the gap from the open side.
    place("supporter_lower_shelf"),
])


def build():
    return scenario_scene, scenario_task, scenario_intents
