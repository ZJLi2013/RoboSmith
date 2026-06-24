"""CAP scenario: open drawer -> pick die -> place it inside -> close drawer.

First long-horizon task that composes rigid pick/place with articulated
open/close (Feature 12). The four-step intent sequence reuses primitives that
already exist; the only new capability exercised is ``place`` into the open
drawer tray via an explicit ``place_z`` plus a ``drawer_open_slot`` target
(a fixed world point for the opened tray interior).

The placed object is a ~5 cm cube (``die_01``): unlike a sphere it does not roll
out of the shallow open tray after release, so it stays put while the drawer is
closed. Geometry numbers below (open-slot XY, place_z, die spot) are derived from
the drawer_cabinet asset + open travel. The die is placed with wide spacing.
"""

from robotsmith.cap.intents import close_, intent_sequence, open_, pick, place
from robotsmith.cap.predicates import all_of, joint_closed, object_in_container
from robotsmith.cap.scene_api import layout, obj, on_articulated, scene, target_position
from robotsmith.cap.task_api import task

scenario_layout = layout(regions=[])

scenario_scene = scene(
    "pick_place_into_drawer",
    layout=scenario_layout,
    objects=[
        # Anchored low cabinet (same placement as open_drawer); drawer face
        # toward the arm so the pulled-out tray opens toward -X.
        obj("drawer", asset="drawer_cabinet", fixed_position=(0.80, 0.0, 0.775)),
        # Die well clear of the opened drawer. The pulled-out tray occupies
        # y in [-0.19, 0.19]; placing the die at y=0.32 lets the arm approach
        # top-down from the +Y side without the links clipping the drawer or
        # knocking it shut. z=0.80 rests the 5 cm cube (half-height 0.025) on
        # the 0.775 table top.
        obj("die", asset="die_01", fixed_position=(0.45, 0.32, 0.80)),
    ],
    target_positions=[
        # Tray-interior center, anchored to the drawer's live joint so it tracks
        # the actual opening instead of assuming "fully open".
        # local_offset = tray_floor in the drawer_cabinet frame (0.035,0,0.136);
        # the runtime applies the asset's metric_scale (0.7), 180 deg yaw and live
        # slide -> e.g. (0.53,0,0.87) at full slide, retracting as it closes.
        target_position(
            "drawer_open_slot",
            anchor=on_articulated(
                "drawer", "drawer_slide", local_offset=(0.035, 0.0, 0.136)
            ),
        ),
    ],
    camera_position=(2.5, -2.1, 1.8),
    camera_target=(0.40, 0.0, 0.55),
    # Extend the support table in +X/+Y so the anchored cabinet (x=0.80, off the
    # default 1.2x0.8 table edge) and the +Y die both rest on it instead of
    # overhanging. Keep the default 0.75 height so open/close reach is unchanged.
    table_size=(2.2, 0.9, 0.05),
)

scenario_task = task(
    "pick_place_into_drawer",
    scene=scenario_scene,
    instruction="Open the drawer, put the die inside, then close the drawer.",
    # Final state: die resting within the (now closed) drawer footprint AND the
    # drawer closed. xy_threshold is generous to cover the tray footprint.
    success=all_of(
        object_in_container("die", "drawer", xy_threshold=0.20, z_margin=0.0),
        joint_closed("drawer", "drawer_slide", closed_position=0.02),
    ),
)

scenario_intents = intent_sequence([
    open_("drawer"),
    # die_01 is a general pickable asset: it goes through the default learned
    # planner (GraspGen) under --grasp-planner auto. The drawer is articulated and
    # is handled by the drag_handle primitive, not grasp-planned.
    pick("die"),
    # Drop into the open tray. place_z is the EE-flange height above the table;
    # the held die hangs ~0.10 m below the flange, and the tray floor sits
    # ~0.10 m above the table, so the flange must clear floor_top + half_cube
    # + grasp_offset (~0.77+0.10+0.025+0.10). place_z=0.25 releases the die just
    # above the tray floor so it settles inside instead of being driven through it.
    place("drawer_open_slot", place_z=0.25),
    close_("drawer"),
])


def build():
    return scenario_scene, scenario_task, scenario_intents
