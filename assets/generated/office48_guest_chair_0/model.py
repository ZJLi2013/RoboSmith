"""Repair guest_chair_0 collision geometry without changing its appearance."""

from pathlib import Path

import bmesh
import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
DEST = ROOT / "out" / "guest_chair_0"
BACK_NAME = "guest_chair_0_curved_back"
Y_CUTS = (-0.20, -0.12, -0.04, 0.04, 0.12, 0.20)
Z_CUTS = (0.47, 0.55, 0.63, 0.71, 0.79)


def band(value, cuts):
    return sum(value >= cut for cut in cuts)


def split_backrest(obj):
    groups = {}
    for poly in obj.data.polygons:
        world_center = obj.matrix_world @ poly.center
        key = (band(world_center.y, Y_CUTS), band(world_center.z, Z_CUTS))
        groups.setdefault(key, set()).add(poly.index)

    created = []
    for (yi, zi), keep_faces in sorted(groups.items()):
        mesh = obj.data.copy()
        patch = bpy.data.objects.new(
            f"{BACK_NAME}_curve_{yi:02d}_incline_{zi:02d}", mesh
        )
        obj.users_collection[0].objects.link(patch)
        patch.matrix_world = obj.matrix_world.copy()

        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()
        remove_faces = [f for f in bm.faces if f.index not in keep_faces]
        bmesh.ops.delete(bm, geom=remove_faces, context="FACES")
        unused_verts = [v for v in bm.verts if not v.link_faces]
        if unused_verts:
            bmesh.ops.delete(bm, geom=unused_verts, context="VERTS")
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
        created.append(patch)

    bpy.data.objects.remove(obj, do_unlink=True)
    return created


def world_bounds(obj):
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    lo = Vector(tuple(min(p[i] for p in points) for i in range(3)))
    hi = Vector(tuple(max(p[i] for p in points) for i in range(3)))
    return lo, hi


bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(SOURCE / "body.glb"))

back = bpy.data.objects.get(BACK_NAME)
if back is None:
    raise RuntimeError(f"source mesh {BACK_NAME!r} was not found")
patches = split_backrest(back)

# The probe is a 4 cm square centred at (0.08, 0). Tight central patches must
# leave it clear above the seat while still containing every original face.
probe_x = (0.06, 0.10)
probe_y = (-0.02, 0.02)
for patch in patches:
    lo, hi = world_bounds(patch)
    overlaps_xy = not (
        hi.x <= probe_x[0]
        or lo.x >= probe_x[1]
        or hi.y <= probe_y[0]
        or lo.y >= probe_y[1]
    )
    if overlaps_xy and hi.z > 0.44679:
        raise RuntimeError(f"backrest patch still blocks probe: {patch.name}")

# Bake transforms while retaining the names of every untouched mesh object.
bpy.ops.object.select_all(action="DESELECT")
meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
for obj in meshes:
    world = obj.matrix_world.copy()
    obj.parent = None
    obj.matrix_world = world
for obj in meshes:
    obj.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

DEST.mkdir(parents=True, exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=str(DEST / "body.glb"),
    export_format="GLB",
    use_selection=True,
    export_apply=False,
)
(DEST / "item.json").write_bytes((SOURCE / "item.json").read_bytes())

lo_all = Vector(tuple(min(world_bounds(o)[0][i] for o in meshes) for i in range(3)))
hi_all = Vector(tuple(max(world_bounds(o)[1][i] for o in meshes) for i in range(3)))
print(f"SPLIT_BACKREST_PATCHES {len(patches)}")
print(
    "BBOX_M",
    f"{hi_all.x-lo_all.x:.6f}",
    f"{hi_all.y-lo_all.y:.6f}",
    f"{hi_all.z-lo_all.z:.6f}",
    "min_z",
    f"{lo_all.z:.6f}",
)
print("PROBE_CLEAR footprint x=[0.06,0.10] y=[-0.02,0.02]")
