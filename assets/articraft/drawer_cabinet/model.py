"""Single-drawer nightstand with a spherical pull handle (one prismatic joint).

Authored from images/drawer.png (RoboSmith articulated-asset bring-up). Geometry in
meters; +X is the drawer pull-out direction (cabinet front), +Z is up.

Body 0.45 (X depth) x 0.60 (Y width) x 0.365 (Z height, legs 0.10). Single-drawer
height (half the original two-drawer carcass) so a top-down grasp clears the
cabinet top edge. Drawer travel 0..0.35 m. Spherical handle diameter 0.05 m for a
Franka parallel-jaw grasp.
"""

from sdk import (
    ArticulatedObject,
    ArticulationType,
    Box,
    Inertial,
    MotionLimits,
    MotionProperties,
    Origin,
    Sphere,
    TestContext,
    TestReport,
)

# Cabinet carcass (interior is the drawer cavity; front face at +X is open).
DEPTH = 0.45          # X
WIDTH = 0.60          # Y
LEG_H = 0.10          # leg height
CARCASS_H = 0.225     # carcass body height (single-drawer; was 0.45 two-drawer)
WALL = 0.018
TOP_H = 0.04

CARCASS_Z0 = LEG_H                 # 0.10
CARCASS_Z1 = LEG_H + CARCASS_H     # 0.325
DEPTH_HALF = DEPTH / 2.0           # 0.225
WIDTH_HALF = WIDTH / 2.0           # 0.30

# Drawer (closed pose == drawer part frame, articulation origin at identity).
DR_FRONT_T = 0.018                 # front-panel thickness (X)
DR_FRONT_W = 0.58                  # front-panel width (Y, overlay style)
DR_FRONT_H = 0.21                  # front-panel height (Z; single-drawer)
DR_FRONT_Z = 0.2125                # front-panel center height (carcass mid)
TRAY_W = 0.52                      # interior width (Y)
TRAY_D = 0.38                      # interior depth (X)
TRAY_H = 0.12                      # interior height (Z)
TRAY_WALL = 0.012
TRAY_BOTTOM_Z = 0.13               # tray floor center height (inside cavity)
HANDLE_R = 0.025                   # spherical handle ball radius (dia 0.05)
HANDLE_NECK_LEN = 0.05             # stem standing the ball off the panel (X)
HANDLE_NECK_W = 0.018              # stem cross-section (Y/Z) — thin so the
#                                    fingers wrap the ball, not the stem
TRAVEL = 0.35                      # max pull-out

# Front-panel inner face sits at the carcass front opening (x = DEPTH_HALF); the
# overlay panel extends forward from there.
FRONT_X = DEPTH_HALF + DR_FRONT_T / 2.0          # 0.234
FRONT_FACE_X = FRONT_X + DR_FRONT_T / 2.0          # 0.243 (outer front face)
HANDLE_NECK_X = FRONT_FACE_X + HANDLE_NECK_LEN / 2.0   # 0.268 (stem center)
# Ball center stands ~72 mm off the panel face → ~47 mm clearance behind the
# ball so a Franka jaw fully wraps the 50 mm ball (form closure on pull) without
# the fingers colliding with the drawer front.
HANDLE_BALL_X = FRONT_FACE_X + HANDLE_NECK_LEN + HANDLE_R - 0.003   # 0.315
TRAY_FRONT_X = DEPTH_HALF                          # tray front edge meets panel
TRAY_CENTER_X = TRAY_FRONT_X - TRAY_D / 2.0        # 0.035


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="drawer_cabinet")
    model.material("body_white", rgba=(0.90, 0.90, 0.90, 1.0))
    model.material("wood_top", rgba=(0.78, 0.62, 0.42, 1.0))
    model.material("handle_white", rgba=(0.85, 0.86, 0.88, 1.0))

    base = model.part("base")
    # Four legs (top of each leg meets the carcass bottom panel at z = LEG_H).
    leg = (0.04, 0.04, LEG_H)
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            base.visual(
                Box(leg),
                origin=Origin(xyz=(sx * (DEPTH_HALF - 0.04), sy * (WIDTH_HALF - 0.05), LEG_H / 2.0)),
                material="body_white",
                name=f"leg_{'p' if sx > 0 else 'n'}{'p' if sy > 0 else 'n'}",
            )
    # Carcass bottom panel.
    base.visual(
        Box((DEPTH, WIDTH, WALL)),
        origin=Origin(xyz=(0.0, 0.0, CARCASS_Z0 + WALL / 2.0)),
        material="body_white",
        name="bottom_panel",
    )
    # Back wall (-X).
    base.visual(
        Box((WALL, WIDTH, CARCASS_H)),
        origin=Origin(xyz=(-DEPTH_HALF + WALL / 2.0, 0.0, CARCASS_Z0 + CARCASS_H / 2.0)),
        material="body_white",
        name="back_wall",
    )
    # Side walls (+/-Y).
    for sy in (1.0, -1.0):
        base.visual(
            Box((DEPTH, WALL, CARCASS_H)),
            origin=Origin(xyz=(0.0, sy * (WIDTH_HALF - WALL / 2.0), CARCASS_Z0 + CARCASS_H / 2.0)),
            material="body_white",
            name=f"side_wall_{'p' if sy > 0 else 'n'}",
        )
    # Wooden top (slight overhang).
    base.visual(
        Box((DEPTH + 0.02, WIDTH + 0.02, TOP_H)),
        origin=Origin(xyz=(0.0, 0.0, CARCASS_Z1 + TOP_H / 2.0)),
        material="wood_top",
        name="top_panel",
    )
    base.inertial = Inertial.from_geometry(
        Box((DEPTH, WIDTH, CARCASS_Z1)),
        mass=18.0,
        origin=Origin(xyz=(0.0, 0.0, CARCASS_Z1 / 2.0)),
    )

    drawer = model.part("drawer")
    # Overlay front panel.
    drawer.visual(
        Box((DR_FRONT_T, DR_FRONT_W, DR_FRONT_H)),
        origin=Origin(xyz=(FRONT_X, 0.0, DR_FRONT_Z)),
        material="body_white",
        name="front_panel",
    )
    # Tray floor.
    drawer.visual(
        Box((TRAY_D, TRAY_W, TRAY_WALL)),
        origin=Origin(xyz=(TRAY_CENTER_X, 0.0, TRAY_BOTTOM_Z + TRAY_WALL / 2.0)),
        material="body_white",
        name="tray_floor",
    )
    # Tray back wall (-X).
    drawer.visual(
        Box((TRAY_WALL, TRAY_W, TRAY_H)),
        origin=Origin(xyz=(TRAY_CENTER_X - TRAY_D / 2.0 + TRAY_WALL / 2.0, 0.0, TRAY_BOTTOM_Z + TRAY_H / 2.0)),
        material="body_white",
        name="tray_back",
    )
    # Tray side walls (+/-Y).
    for sy in (1.0, -1.0):
        drawer.visual(
            Box((TRAY_D, TRAY_WALL, TRAY_H)),
            origin=Origin(xyz=(TRAY_CENTER_X, sy * (TRAY_W / 2.0 - TRAY_WALL / 2.0), TRAY_BOTTOM_Z + TRAY_H / 2.0)),
            material="body_white",
            name=f"tray_side_{'p' if sy > 0 else 'n'}",
        )
    drawer.inertial = Inertial.from_geometry(
        Box((TRAY_D, TRAY_W, DR_FRONT_H)),
        mass=1.5,
        origin=Origin(xyz=(TRAY_CENTER_X, 0.0, DR_FRONT_Z)),
    )

    # Pull handle on its OWN link `drawer_handle` (fixed joint to the drawer).
    # Semantically the drawer body (front panel + tray) and the touchable knob are
    # different things: the planner must AVOID the body but is ALLOWED to touch the
    # knob. Keeping them as one link forces a coarse all-or-nothing collision
    # exemption; splitting the knob (+ its neck stem) onto `drawer_handle` lets the
    # planner allow-list just this link during grasp while the drawer body stays a
    # hard obstacle. Geometry stays in drawer coordinates (fixed joint origin is
    # identity), so the knob's world pose is unchanged. The thin neck stands the
    # spherical knob off the front panel so a Franka jaw can fully wrap the ball
    # (form closure on pull) without the fingers hitting the drawer front.
    drawer_handle = model.part("drawer_handle", meta={"graspable": True})
    drawer_handle.visual(
        Box((HANDLE_NECK_LEN, HANDLE_NECK_W, HANDLE_NECK_W)),
        origin=Origin(xyz=(HANDLE_NECK_X, 0.0, DR_FRONT_Z)),
        material="handle_white",
        name="handle_neck",
    )
    drawer_handle.visual(
        Sphere(radius=HANDLE_R),
        origin=Origin(xyz=(HANDLE_BALL_X, 0.0, DR_FRONT_Z)),
        material="handle_white",
        name="handle",
    )
    drawer_handle.inertial = Inertial.from_geometry(
        Box((HANDLE_NECK_LEN + 2.0 * HANDLE_R, 2.0 * HANDLE_R, 2.0 * HANDLE_R)),
        mass=0.05,
        origin=Origin(xyz=(HANDLE_NECK_X, 0.0, DR_FRONT_Z)),
    )

    model.articulation(
        "drawer_slide",
        ArticulationType.PRISMATIC,
        parent=base,
        child=drawer,
        origin=Origin(xyz=(0.0, 0.0, 0.0)),
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(effort=120.0, velocity=1.0, lower=0.0, upper=TRAVEL),
        # Low damping/friction → a freely sliding drawer (axis is horizontal, so
        # no gravity self-slide); the grasp drives travel via form closure.
        motion_properties=MotionProperties(damping=0.3, friction=0.1),
    )
    # Handle rigidly bolted to the drawer front (no DOF).
    model.articulation(
        "handle_mount",
        ArticulationType.FIXED,
        parent=drawer,
        child=drawer_handle,
        origin=Origin(xyz=(0.0, 0.0, 0.0)),
    )

    return model


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    base = object_model.get_part("base")
    drawer = object_model.get_part("drawer")
    slide = object_model.get_articulation("drawer_slide")

    # Closed: drawer stays centered inside the carcass cavity on the non-motion axes.
    with ctx.pose({slide: 0.0}):
        ctx.expect_within(
            drawer, base, axes="yz", margin=0.02,
            name="drawer nests inside the carcass (closed)",
        )
        closed = ctx.part_world_position(drawer)
    # Open: drawer translates outward along +X by ~the full travel.
    with ctx.pose({slide: TRAVEL}):
        opened = ctx.part_world_position(drawer)
    ctx.check(
        "drawer slides out along +X",
        closed is not None and opened is not None and opened[0] > closed[0] + 0.30,
        details=f"closed={closed} open={opened}",
    )

    return ctx.report()


object_model = build_object_model()
