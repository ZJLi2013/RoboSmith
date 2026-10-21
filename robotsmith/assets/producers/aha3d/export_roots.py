"""Export AHa-3D scene roots as Blender items (run inside Blender 5.2):

    blender -b scene.blend --python export_roots.py -- <out_dir> [<instance_id> ...]

A root is an object carrying ``instance_id`` (an EMPTY assembly or a single mesh) whose
parent carries none; without ids every root is exported. Each is
written as ``<out_dir>/<slug>/body.glb`` + ``item.json`` in the Blender item contract: one
part, bottom centre at the origin, room axes kept, every evaluated mesh (modifiers applied,
curves meshed) a uniquely named object, so each becomes one collision box.
``<out_dir>/scene.json`` records each root's class, world origin and bounds.
"""

import json
import re
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

# Whitebox stand-in: mass from the bounding box volume by class (kg / m^3).
DENSITY = {"prop": 400.0, "furniture": 250.0}


def slug(instance_id: str) -> str:
    return re.sub(r"\W", "_", instance_id.split("/")[-1])


def export_root(root, out: Path) -> dict:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    sources = [o for o in [root, *root.children_recursive] if o.type in ("MESH", "CURVE")]
    temp = []
    for o in sources:
        mesh = bpy.data.meshes.new_from_object(o.evaluated_get(depsgraph))
        if not mesh.vertices:
            continue
        t = bpy.data.objects.new(slug(o.name), mesh)
        t.matrix_world = o.matrix_world.copy()
        bpy.context.scene.collection.objects.link(t)
        temp.append(t)
    if not temp:
        return {"instance_id": root["instance_id"], "skipped": "no mesh geometry"}
    corners = [t.matrix_world @ Vector(v.co) for t in temp for v in t.data.vertices]
    lo = Vector([min(c[i] for c in corners) for i in range(3)])
    hi = Vector([max(c[i] for c in corners) for i in range(3)])
    origin = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
    for t in temp:
        t.matrix_world = Matrix.Translation(-origin) @ t.matrix_world
    item_dir = out / slug(root["instance_id"])
    item_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    for t in temp:
        t.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(item_dir / "body.glb"), export_format="GLB", use_selection=True)
    cls = root.get("semantic_class", "prop")
    kind = "fixture" if cls != "prop" else "rigid"
    size = hi - lo
    mass = DENSITY.get(cls, DENSITY["prop"]) * size.x * size.y * size.z
    item = {"name": item_dir.name, "kind": kind,
            "parts": [{"name": "body", "mesh": "body.glb", "mass_kg": round(max(mass, 0.01), 3)}]}
    (item_dir / "item.json").write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")
    for t in temp:
        bpy.data.objects.remove(t)
    return {"instance_id": root["instance_id"], "item": item_dir.name, "semantic_class": cls, "kind": kind,
            "origin": [round(v, 4) for v in origin], "lo": [round(v, 4) for v in lo], "hi": [round(v, 4) for v in hi]}


def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1:]
    out, wanted = Path(args[0]), set(args[1:])
    out.mkdir(parents=True, exist_ok=True)
    roots = [o for o in bpy.data.objects if o.get("instance_id") and (not wanted or o["instance_id"] in wanted)
             and not (o.parent and o.parent.get("instance_id"))]
    missing = wanted - {o["instance_id"] for o in roots}
    if missing:
        raise SystemExit(f"no root with instance_id {sorted(missing)}")
    scene = {"source": bpy.data.filepath, "roots": [export_root(r, out) for r in roots]}
    (out / "scene.json").write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")


main()
