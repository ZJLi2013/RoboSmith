"""Build a bright Duplo-style 2 x 4 toy brick for robot manipulation."""

import json
from pathlib import Path

import bpy


OUT_DIR = Path("out/block")


def apply_transforms(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    obj.select_set(False)


def add_box(name, dimensions, location, material):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    obj.data.materials.append(material)
    apply_transforms(obj)
    return obj


def add_stud(name, x, y, material):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=48,
        radius=0.005,
        depth=0.005,
        end_fill_type="NGON",
        location=(x, y, 0.0215),
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    apply_transforms(obj)
    return obj


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    material = bpy.data.materials.new("bright_orange_plastic")
    material.diffuse_color = (1.0, 0.12, 0.015, 1.0)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (1.0, 0.12, 0.015, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.28

    objects = [
        add_box(
            "block_body",
            dimensions=(0.064, 0.032, 0.019),
            location=(0.0, 0.0, 0.0095),
            material=material,
        )
    ]

    x_positions = (-0.024, -0.008, 0.008, 0.024)
    y_positions = (-0.008, 0.008)
    for row, y in enumerate(y_positions, start=1):
        for col, x in enumerate(x_positions, start=1):
            objects.append(add_stud(f"stud_{row}_{col}", x, y, material))

    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.export_scene.gltf(
        filepath=str(OUT_DIR / "block.glb"),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
    )

    item = {
        "name": "block",
        "kind": "rigid",
        "parts": [
            {
                "name": "block",
                "mesh": "block.glb",
                "mass_kg": 0.03,
                "material": "plastic",
            }
        ],
    }
    (OUT_DIR / "item.json").write_text(
        json.dumps(item, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
