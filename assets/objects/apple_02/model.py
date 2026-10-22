import json
import math
from pathlib import Path

import bpy
from mathutils import Vector


OUT = Path("out/apple")
OUT.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.unit_settings.system = "METRIC"
bpy.context.scene.unit_settings.scale_length = 1.0


def material(name, rgba, roughness):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = rgba
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


skin = material("red_skin", (0.62, 0.012, 0.018, 1.0), 0.32)
wood = material("brown_stem", (0.16, 0.055, 0.016, 1.0), 0.72)

# The ordered profile starts at the flat underside, follows the outside upward,
# then curls inward and slightly downward to make the top dimple.
profile = [
    (0.0000, 0.0200),
    (0.0020, 0.0255),
    (0.0070, 0.0300),
    (0.0150, 0.0322),
    (0.0260, 0.0325),
    (0.0370, 0.0310),
    (0.0460, 0.0265),
    (0.0520, 0.0200),
    (0.0555, 0.0120),
    (0.0565, 0.0070),
    (0.0542, 0.0028),
]
segments = 64
verts = []

# A five-lobed modulation is subtle at the widest point and stronger near the
# shoulders, giving the toy a recognizable apple silhouette without sharpness.
for ring_index, (z, base_radius) in enumerate(profile):
    shoulder = max(0.0, min(1.0, (z - 0.025) / 0.030))
    lobe_strength = 0.010 + 0.025 * shoulder
    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        modulation = (1.0 - lobe_strength) + lobe_strength * math.cos(5.0 * angle)
        radius = base_radius * modulation
        verts.append((radius * math.cos(angle), radius * math.sin(angle), z))

bottom_center = len(verts)
verts.append((0.0, 0.0, 0.0))
top_center = len(verts)
verts.append((0.0, 0.0, 0.0538))

faces = []
for ring in range(len(profile) - 1):
    a0 = ring * segments
    b0 = (ring + 1) * segments
    for i in range(segments):
        j = (i + 1) % segments
        faces.append((a0 + i, a0 + j, b0 + j, b0 + i))
for i in range(segments):
    j = (i + 1) % segments
    faces.append((bottom_center, j, i))
    last = (len(profile) - 1) * segments
    faces.append((last + i, last + j, top_center))

mesh = bpy.data.meshes.new("apple_skin_mesh")
mesh.from_pydata(verts, [], faces)
mesh.update()
body = bpy.data.objects.new("apple_skin", mesh)
bpy.context.collection.objects.link(body)
body.data.materials.append(skin)
for polygon in body.data.polygons:
    polygon.use_smooth = polygon.center.z > 0.0002

# Keep the stem short and vertical. Its lower end disappears into the dimple.
bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=0.00235, depth=0.0160,
                                    location=(0.0, 0.0, 0.0620))
stem = bpy.context.object
stem.name = "apple_stem"
stem.data.name = "apple_stem_mesh"
stem.data.materials.append(wood)
bevel = stem.modifiers.new("soft_stem_edges", "BEVEL")
bevel.width = 0.00045
bevel.segments = 2
bpy.context.view_layer.objects.active = stem
bpy.ops.object.modifier_apply(modifier=bevel.name)
for polygon in stem.data.polygons:
    polygon.use_smooth = True

# Apply all transforms before exporting the one rigid part.
bpy.ops.object.select_all(action="SELECT")
for obj in bpy.context.selected_objects:
    if obj.type == "MESH":
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

bpy.ops.export_scene.gltf(
    filepath=str(OUT / "apple.glb"),
    export_format="GLB",
    use_selection=True,
    export_apply=True,
)

item = {
    "name": "apple",
    "kind": "rigid",
    "parts": [
        {
            "name": "apple",
            "mesh": "apple.glb",
            "collision": "mesh",
            "mass_kg": 0.15,
            "material": "plastic",
        }
    ],
}
(OUT / "item.json").write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")

print("BUILT", OUT / "apple.glb", OUT / "item.json")
