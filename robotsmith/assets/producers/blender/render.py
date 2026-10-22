"""Render the exported asset from four views: front, side, top, iso.

Run: ./run_blender.sh tools/render.py
Loads every glb under out/ (each item folder), frames them, and writes
out/renders/<view>.png with a 10 cm grid on the ground for scale.
"""

import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

out = Path("out")
renders = out / "renders"
renders.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
for glb in sorted(out.rglob("*.glb")):
    bpy.ops.import_scene.gltf(filepath=str(glb))
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
if not meshes:
    sys.exit("no glb found under out/")

lo = Vector((1e9, 1e9, 1e9))
hi = Vector((-1e9, -1e9, -1e9))
for o in meshes:
    for c in o.bound_box:
        w = o.matrix_world @ Vector(c)
        lo = Vector(map(min, lo, w))
        hi = Vector(map(max, hi, w))
center, size = (lo + hi) / 2, max((hi - lo).length, 0.05)
print(f"BBOX_M {hi.x - lo.x:.4f} {hi.y - lo.y:.4f} {hi.z - lo.z:.4f} min_z {lo.z:.4f}")

bpy.ops.mesh.primitive_grid_add(x_subdivisions=20, y_subdivisions=20, size=2.0, location=(center.x, center.y, 0))
grid = bpy.context.object
grid.display_type = "WIRE"
mod = grid.modifiers.new("w", "WIREFRAME")
mod.thickness = 0.002

world = bpy.data.worlds.new("w")
world.color = (0.9, 0.9, 0.9)
bpy.context.scene.world = world
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.rotation_euler = (math.radians(50), 0, math.radians(30))
bpy.context.scene.collection.objects.link(sun)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 16
scene.render.resolution_x = scene.render.resolution_y = 512
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
scene.collection.objects.link(cam)
scene.camera = cam

views = {"front": (1, 0, 0.15), "side": (0, -1, 0.15), "top": (0, 0, 1), "iso": (1, -1, 0.8)}
for name, d in views.items():
    direction = Vector(d).normalized()
    cam.location = center + direction * size * 2.2
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(renders / f"{name}.png")
    bpy.ops.render.render(write_still=True)
print("RENDERED", ", ".join(str(renders / f"{n}.png") for n in views))
