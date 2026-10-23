import json
from pathlib import Path

import bpy


OUT = Path("out/drawer_cabinet")


def material(name, color, metallic=0.0, roughness=0.45):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def box(name, size, location, mat, bevel=0.002):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    if bevel:
        mod = obj.modifiers.new("soft_edges", "BEVEL")
        mod.width = bevel
        mod.segments = 2
        mod.affect = "EDGES"
    return obj


def sphere(name, diameter, location, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=diameter / 2, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return obj


def export_part(part_name, objects):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.export_scene.gltf(
        filepath=str(OUT / f"{part_name}.glb"),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
    )


bpy.ops.wm.read_factory_settings(use_empty=True)
OUT.mkdir(parents=True, exist_ok=True)

wood = material("warm_oak", (0.52, 0.25, 0.09), roughness=0.38)
wood_edge = material("oak_edge", (0.34, 0.13, 0.035), roughness=0.42)
drawer_mat = material("drawer_oak", (0.68, 0.38, 0.14), roughness=0.4)
metal = material("brushed_steel", (0.16, 0.19, 0.22), metallic=0.78, roughness=0.26)

# Cabinet envelope: X 0.30, Y 0.42, Z 0.25.  The front is +X.
# Its opening is unobstructed at +X, and four short feet put the shell above z=0.
carcass = []
carcass.append(box("carcass_left_side", (0.30, 0.018, 0.202), (0.0, 0.201, 0.136), wood))
carcass.append(box("carcass_right_side", (0.30, 0.018, 0.202), (0.0, -0.201, 0.136), wood))
carcass.append(box("carcass_top", (0.30, 0.384, 0.018), (0.0, 0.0, 0.241), wood))
carcass.append(box("carcass_bottom", (0.30, 0.384, 0.018), (0.0, 0.0, 0.031), wood))
carcass.append(box("carcass_back", (0.018, 0.384, 0.192), (-0.141, 0.0, 0.136), wood_edge))

for sx in (-1, 1):
    for sy in (-1, 1):
        carcass.append(
            box(
                f"leg_{'front' if sx > 0 else 'rear'}_{'left' if sy > 0 else 'right'}",
                (0.038, 0.038, 0.022),
                (sx * 0.116, sy * 0.168, 0.011),
                wood_edge,
                bevel=0.003,
            )
        )

# Open tray at its closed joint position. Outer width/depth are 0.376/0.272 m.
# Clear interior: 0.356 m in Y, 0.252 m in X, and 0.160 m deep.
drawer = []
drawer.append(box("tray_floor", (0.272, 0.376, 0.012), (0.009, 0.0, 0.058), drawer_mat, bevel=0.0015))
drawer.append(box("tray_back", (0.010, 0.376, 0.172), (-0.122, 0.0, 0.138), drawer_mat, bevel=0.0015))
drawer.append(box("tray_left_side", (0.252, 0.010, 0.172), (0.009, 0.183, 0.138), drawer_mat, bevel=0.0015))
drawer.append(box("tray_right_side", (0.252, 0.010, 0.172), (0.009, -0.183, 0.138), drawer_mat, bevel=0.0015))
drawer.append(box("drawer_front_panel", (0.010, 0.376, 0.188), (0.140, 0.0, 0.136), drawer_mat, bevel=0.0025))

# Separate rigid handle, attached to the drawer in item.json. The 12 mm neck projects
# 40 mm along +X; its 35 mm knob gives the pulling gripper a positive stop.
handle = []
handle.append(box("handle_neck", (0.040, 0.012, 0.012), (0.165, 0.0, 0.136), metal, bevel=0.003))
handle.append(sphere("handle_knob", 0.035, (0.2025, 0.0, 0.136), metal))

export_part("carcass", carcass)
export_part("drawer", drawer)
export_part("handle", handle)

item = {
    "name": "drawer_cabinet",
    "kind": "articulated",
    "parts": [
        {"name": "carcass", "mesh": "carcass.glb", "mass_kg": 6.0, "material": "wood"},
        {"name": "drawer", "mesh": "drawer.glb", "mass_kg": 0.8, "material": "wood"},
        {
            "name": "handle",
            "mesh": "handle.glb",
            "mass_kg": 0.08,
            "material": "metal",
            "parent": "drawer",
        },
    ],
    "joints": [
        {
            "name": "drawer_slide",
            "type": "prismatic",
            "parent": "carcass",
            "child": "drawer",
            "axis": [1, 0, 0],
            "origin": [0, 0, 0],
            "lower": 0.0,
            "upper": 0.24,
        }
    ],
    "handles": [{"part": "handle", "object": "handle_neck"}],
    "place_targets": [{"name": "drawer_opening", "part": "drawer", "surface": "tray_floor"}],
}

(OUT / "item.json").write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")
print(f"Built {OUT}")
