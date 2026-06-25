"""Two-layer support stand (rigid fixture, no movable joints).

Authored from images/2-layer-supporter.png (RoboSmith articraft bring-up).
Geometry in meters; +X is the open/front direction (hole faces and shelf access),
+Z is up, origin at the footprint center on the floor.

Two vertical posts (220 mm apart, 190 mm tall, 60 mm deep) bolt to the floor via
foot plates carrying 4x Oe6.5 mounting holes. The posts are bridged by a tall rear
plate and two horizontal shelves (a "two-layer" rack): the lower shelf ~70 mm up,
the upper shelf ~100 mm above it. Everything is one rigid weldment, so the URDF is
a single link with no articulation.
"""

from sdk import (
    ArticulatedObject,
    Box,
    Inertial,
    Origin,
    TestContext,
    TestReport,
)

# --- footprint / posts ------------------------------------------------------
WIDTH = 0.22          # Y, outer span between the two posts
DEPTH = 0.06          # X, post depth (drawing: 60 mm; inner cavity 40 mm)
POST_Y = 0.025        # Y, post thickness
POST_H = 0.178        # Z, post height above the foot plate
HALF_W = WIDTH / 2.0 - POST_Y / 2.0   # 0.0975, post center offset in Y

FOOT_H = 0.012        # foot plate thickness
FOOT_X = 0.10         # foot plate length in X (overhangs front/back for holes)
FOOT_Y = 0.03         # foot plate width in Y

POST_Z0 = FOOT_H                       # 0.012, post sits on top of the foot
POST_ZC = POST_Z0 + POST_H / 2.0       # 0.101
POST_Z1 = POST_Z0 + POST_H             # 0.190 (overall post height per drawing)

# --- shelves (the two layers) ----------------------------------------------
SHELF_X = 0.05        # shelf depth (X)
SHELF_Y = WIDTH - 2.0 * POST_Y         # 0.17, spans between post inner faces
SHELF_T = 0.008       # shelf thickness (Z)
SHELF_LOWER_Z = 0.07  # lower shelf center height (~60 mm clearance to floor)
SHELF_UPPER_Z = 0.17  # upper shelf center height (~100 mm above lower shelf)

# --- rear plate -------------------------------------------------------------
BACK_T = 0.008        # rear plate thickness (X)
BACK_Y = SHELF_Y      # 0.17
BACK_H = 0.13         # rear plate height
BACK_XC = -DEPTH / 2.0 + BACK_T / 2.0  # -0.026, flush with post rear face
BACK_ZC = 0.165       # spans 0.10..0.23, rises above the posts

# 4x Oe6.5 mm floor-mounting holes, two per foot (front/back), documented in
# metadata; not subtracted from the primitive geometry (cosmetic for sim).
HOLE_DIA = 0.0065


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="two_layer_supporter")
    model.material("alu", rgba=(0.80, 0.81, 0.83, 1.0))

    base = model.part("base")

    # Foot plates + posts (left/right).
    for sy in (1.0, -1.0):
        tag = "p" if sy > 0 else "n"
        base.visual(
            Box((FOOT_X, FOOT_Y, FOOT_H)),
            origin=Origin(xyz=(0.0, sy * HALF_W, FOOT_H / 2.0)),
            material="alu",
            name=f"foot_{tag}",
        )
        base.visual(
            Box((DEPTH, POST_Y, POST_H)),
            origin=Origin(xyz=(0.0, sy * HALF_W, POST_ZC)),
            material="alu",
            name=f"post_{tag}",
        )

    # Two shelves bridging the posts.
    base.visual(
        Box((SHELF_X, SHELF_Y, SHELF_T)),
        origin=Origin(xyz=(0.0, 0.0, SHELF_LOWER_Z)),
        material="alu",
        name="shelf_lower",
    )
    base.visual(
        Box((SHELF_X, SHELF_Y, SHELF_T)),
        origin=Origin(xyz=(0.0, 0.0, SHELF_UPPER_Z)),
        material="alu",
        name="shelf_upper",
    )

    # Tall rear plate.
    base.visual(
        Box((BACK_T, BACK_Y, BACK_H)),
        origin=Origin(xyz=(BACK_XC, 0.0, BACK_ZC)),
        material="alu",
        name="back_plate",
    )

    # One rigid weldment; approximate inertia with the overall bounding box.
    base.inertial = Inertial.from_geometry(
        Box((FOOT_X, WIDTH, BACK_ZC + BACK_H / 2.0)),
        mass=2.0,
        origin=Origin(xyz=(0.0, 0.0, (BACK_ZC + BACK_H / 2.0) / 2.0)),
    )

    return model


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    base = object_model.get_part("base")
    ctx.check(
        "single rigid link (no articulation)",
        len(object_model.articulations) == 0,
        details=f"articulations={list(object_model.articulations)}",
    )
    ctx.check(
        "two shelves present",
        {"shelf_lower", "shelf_upper"}.issubset({v.name for v in base.visuals}),
        details=f"visuals={[v.name for v in base.visuals]}",
    )
    return ctx.report()


object_model = build_object_model()
