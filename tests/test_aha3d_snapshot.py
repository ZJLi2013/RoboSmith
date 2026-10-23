"""AHa-3D snapshot import: which roots become the table, table objects, floor fixtures or are dropped."""

from robotsmith.assets.producers.aha3d_snapshot import plan


def _root(iid, cls, lo, hi):
    return {"instance_id": iid, "item": iid.split("/")[1], "semantic_class": cls, "lo": lo, "hi": hi,
            "origin": [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]}


SCENE = {"roots": [
    _root("furniture/desk", "furniture", [1.0, -1.0, 0.0], [1.6, 1.0, 0.72]),
    _root("decor/cup", "prop", [1.2, 0.1, 0.72], [1.28, 0.18, 0.82]),
    _root("decor/paper", "prop", [1.3, -0.3, 0.72], [1.5, -0.1, 0.7202]),
    _root("seating/chair", "furniture", [-0.25, -0.25, 0.0], [0.25, 0.25, 0.9]),
    _root("seating/stool", "furniture", [0.3, 0.8, 0.0], [0.7, 1.1, 0.5]),
    _root("lighting/floor_lamp", "furniture", [1.0, -0.5, 0.0], [1.3, -0.3, 1.6]),
    _root("storage/shelf", "furniture", [3.5, 3.0, 0.0], [4.0, 3.5, 1.8]),
    _root("lighting/wall_lamp", "furniture", [0.5, -2.0, 1.5], [0.7, -1.9, 1.7]),
    _root("structure/wall", "structure", [-3.0, -3.0, 0.0], [3.0, -2.9, 2.5]),
    {"instance_id": "decor/empty", "item": "empty", "semantic_class": "prop", "skipped": "no mesh geometry"},
]}


def test_plan_sorts_every_root_and_keeps_room_positions():
    roots = plan(SCENE, "furniture/desk", (0.0, 0.0))["roots"]
    fate = {i: (d["decision"], d.get("reason", "")) for i, d in roots.items()}
    assert fate == {
        "furniture/desk": ("table", ""),
        "decor/cup": ("support", ""),
        "decor/paper": ("dropped", "thinner than 5 mm"),
        "seating/chair": ("dropped", "stands where the robot stands"),
        "seating/stool": ("floor", ""),
        "lighting/floor_lamp": ("dropped", "overlaps the robot's table"),
        "storage/shelf": ("dropped", "more than 1.2 m from the robot"),
        "lighting/wall_lamp": ("dropped", "not on the table or the floor"),
        "structure/wall": ("dropped", "structure is not imported"),
        "decor/empty": ("dropped", "no mesh geometry"),
    }
    assert roots["decor/cup"]["anchor_xy"] == [1.24, 0.14]


def test_plan_expresses_positions_relative_to_the_robot_base():
    roots = plan(SCENE, "furniture/desk", (0.5, 0.0))["roots"]
    assert roots["decor/cup"]["anchor_xy"] == [0.74, 0.14]


def test_plan_follows_the_robot_heading():
    def turn(r):
        if "lo" not in r:
            return r
        (x0, y0), (x1, y1) = r["lo"][:2], r["hi"][:2]
        return _root(r["instance_id"], r["semantic_class"], [-y1, x0, r["lo"][2]], [-y0, x1, r["hi"][2]])

    turned = plan({"roots": [turn(r) for r in SCENE["roots"]]}, "furniture/desk", (0.0, 0.0), yaw_deg=90.0)
    assert turned["roots"] == plan(SCENE, "furniture/desk", (0.0, 0.0))["roots"]
    assert turned["yaw_deg"] == 90.0
