"""Refine guest_chair_1's curved-back collision approximation."""

from pathlib import Path
import shutil

import bmesh
import bpy


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
DEST = ROOT / "out" / "guest_chair_1"
BACK_NAME = "guest_chair_1_curved_back"

# Partition the shell along both its recline and wraparound curvature. The
# boundaries sit between source mesh rows, keeping each collision AABB tight.
Y_EDGES = (-1.0, -0.205, -0.135, -0.070, -0.035, 0.035, 0.070, 0.135, 0.205, 1.0)
Z_EDGES = (
    -1.0, 0.445, 0.525, 0.585, 0.625, 0.645, 0.655, 0.675,
    0.715, 0.765, 0.825, 1.0,
)


def interval(value, edges):
    for index in range(len(edges) - 1):
        if edges[index] <= value < edges[index + 1]:
            return index
    return len(edges) - 2


def split_back(source_obj):
    """Clip exact surface patches without moving or restyling the shell."""
    occupied = {
        (z_index, y_index)
        for z_index in range(len(Z_EDGES) - 1)
        for y_index in range(len(Y_EDGES) - 1)
    }

    collection = source_obj.users_collection[0]
    pieces = []
    for z_index, y_index in sorted(occupied):
        mesh = source_obj.data.copy()
        piece = bpy.data.objects.new(
            f"{BACK_NAME}_z{z_index:02d}_y{y_index:02d}", mesh
        )
        collection.objects.link(piece)
        piece.matrix_world = source_obj.matrix_world.copy()

        bm = bmesh.new()
        bm.from_mesh(mesh)
        for coordinate, low, high, normal in (
            ("z", Z_EDGES[z_index], Z_EDGES[z_index + 1], (0.0, 0.0, 1.0)),
            ("y", Y_EDGES[y_index], Y_EDGES[y_index + 1], (0.0, 1.0, 0.0)),
        ):
            if low > -0.9:
                origin = (0.0, low, 0.0) if coordinate == "y" else (0.0, 0.0, low)
                bmesh.ops.bisect_plane(
                    bm, geom=list(bm.verts) + list(bm.edges) + list(bm.faces),
                    plane_co=origin, plane_no=normal, clear_inner=True,
                )
            if high < 0.9:
                origin = (0.0, high, 0.0) if coordinate == "y" else (0.0, 0.0, high)
                bmesh.ops.bisect_plane(
                    bm, geom=list(bm.verts) + list(bm.edges) + list(bm.faces),
                    plane_co=origin, plane_no=normal, clear_outer=True,
                )
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
        if mesh.vertices:
            pieces.append(piece)
        else:
            bpy.data.objects.remove(piece, do_unlink=True)
            bpy.data.meshes.remove(mesh)

    old_mesh = source_obj.data
    bpy.data.objects.remove(source_obj, do_unlink=True)
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)
    return pieces


def world_bounds(obj):
    points = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    return (
        min(point.x for point in points), max(point.x for point in points),
        min(point.y for point in points), max(point.y for point in points),
        min(point.z for point in points), max(point.z for point in points),
    )


bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(SOURCE / "body.glb"))

back = bpy.data.objects.get(BACK_NAME)
if back is None or back.type != "MESH":
    raise RuntimeError(f"Missing mesh object {BACK_NAME!r}")

# Bake imported transforms as required by the item contract.
bpy.ops.object.select_all(action="DESELECT")
for obj in bpy.data.objects:
    if obj.type == "MESH":
        obj.select_set(True)
bpy.context.view_layer.objects.active = back
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
pieces = split_back(back)

# Guard the opening explicitly called out by the reviewer.
crossing = []
for piece in pieces:
    bounds = world_bounds(piece)
    if bounds[2] <= 0.0 <= bounds[3] and bounds[4] <= 0.65 <= bounds[5]:
        crossing.append((piece.name, bounds))
if not crossing:
    raise RuntimeError("No refined back patch covers y=0,z=0.65")
for name, bounds in crossing:
    print(f"BACK_CHECK {name} x={bounds[0]:.4f}..{bounds[1]:.4f}")
    if bounds[0] < 0.18:
        raise RuntimeError(f"Back patch {name} still blocks the seat-side opening")

DEST.mkdir(parents=True, exist_ok=True)
shutil.copy2(SOURCE / "item.json", DEST / "item.json")

bpy.ops.object.select_all(action="DESELECT")
for obj in bpy.data.objects:
    if obj.type == "MESH":
        obj.select_set(True)
bpy.context.view_layer.objects.active = pieces[0]
bpy.ops.export_scene.gltf(
    filepath=str(DEST / "body.glb"),
    export_format="GLB",
    use_selection=True,
    export_apply=False,
)
print(f"WROTE {DEST / 'body.glb'} with {len(pieces)} curved-back patches")
