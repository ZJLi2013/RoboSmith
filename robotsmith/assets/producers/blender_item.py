"""Compile a Blender-built item (``item.json`` + one glb per part, see
``blender/CONTRACT.md``) into an asset directory: ``model.urdf`` with one named
collision box per mesh object, one OBJ visual per part, and ``asset.json``. The asset is a
draft until its review passes.

    python -m robotsmith.assets.producers.blender_item <out/item> <assets/id> \
        --category mug [--build build.py] [--by agent:<model>] [--description "..."]
"""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

from robotsmith.assets import store

# glTF is +Y up; Blender's exporter converts from +Z up.
_Y_UP_TO_Z_UP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)


def _xyz(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def _objects(glb: Path) -> dict[str, trimesh.Trimesh]:
    """A part's named mesh objects in item coordinates (+Z up)."""
    scene = trimesh.load(str(glb), force="scene")
    out = {}
    for node in scene.graph.nodes_geometry:
        transform, geom = scene.graph[node]
        mesh = scene.geometry[geom].copy()
        mesh.apply_transform(_Y_UP_TO_Z_UP @ transform)
        out[node] = mesh
    return out


def _origin(parent: ET.Element, xyz) -> None:
    ET.SubElement(parent, "origin", xyz=_xyz(xyz), rpy="0 0 0")


def _link(robot: ET.Element, name: str, objects: dict[str, trimesh.Trimesh], frame, mass: float, asset_dir: Path,
          mesh_collision: bool = False) -> None:
    visual = trimesh.util.concatenate(list(objects.values()))
    visual.apply_translation(-frame)
    # trimesh names textures material_<i>.png; prefix them so parts do not overwrite each other.
    text, files = trimesh.exchange.obj.export_obj(visual, return_texture=True, mtl_name=f"{name}.mtl")
    mtl = files.pop(f"{name}.mtl", None)
    for fname, data in files.items():
        (asset_dir / f"{name}_{fname}").write_bytes(data)
        mtl = mtl.replace(fname.encode(), f"{name}_{fname}".encode())
    if mtl is not None:
        (asset_dir / f"{name}.mtl").write_bytes(mtl)
    (asset_dir / f"{name}.obj").write_text(text, encoding="utf-8", newline="\n")
    lo, hi = visual.bounds
    ex, ey, ez = hi - lo
    link = ET.SubElement(robot, "link", name=name)
    inertial = ET.SubElement(link, "inertial")
    _origin(inertial, (lo + hi) / 2)
    ET.SubElement(inertial, "mass", value=f"{mass:.6g}")
    ET.SubElement(
        inertial, "inertia",
        ixx=f"{mass / 12 * (ey**2 + ez**2):.6g}", iyy=f"{mass / 12 * (ex**2 + ez**2):.6g}",
        izz=f"{mass / 12 * (ex**2 + ey**2):.6g}", ixy="0", ixz="0", iyz="0",
    )
    vis = ET.SubElement(link, "visual", name=name)
    _origin(vis, (0, 0, 0))
    ET.SubElement(ET.SubElement(vis, "geometry"), "mesh", filename=f"{name}.obj")
    if mesh_collision:
        col = ET.SubElement(link, "collision", name=name)
        _origin(col, (0, 0, 0))
        ET.SubElement(ET.SubElement(col, "geometry"), "mesh", filename=f"{name}.obj")
        return
    for obj_name, mesh in objects.items():
        olo, ohi = mesh.bounds - frame
        col = ET.SubElement(link, "collision", name=obj_name)
        _origin(col, (olo + ohi) / 2)
        ET.SubElement(ET.SubElement(col, "geometry"), "box", size=_xyz(ohi - olo))


def compile_item(item_dir: Path, asset_dir: Path, *, build: Path | None = None,
                 by: str = "agent:blender", description: str | None = None, source: str = "blender",
                 provenance: dict | None = None, license: str = "Apache-2.0",
                 attribution: str | None = None, category: str | None = None) -> dict:
    spec = json.loads((item_dir / "item.json").read_text(encoding="utf-8"))
    parts = {p["name"]: p for p in spec["parts"]}
    root_name = spec["parts"][0]["name"]
    joint_of = {j["child"]: j for j in spec.get("joints", [])}
    parent_of = {n: joint_of[n]["parent"] if n in joint_of else p.get("parent", root_name)
                 for n, p in parts.items() if n != root_name}
    frames = {root_name: np.zeros(3)}

    def frame(name: str) -> np.ndarray:
        if name not in frames:
            frames[name] = (np.asarray(joint_of[name]["origin"], dtype=float) if name in joint_of
                            else frame(parent_of[name]))
        return frames[name]

    objects = {n: _objects(item_dir / p["mesh"]) for n, p in parts.items()}
    names = [o for objs in objects.values() for o in objs]
    if len(set(names)) != len(names):
        raise ValueError(f"{item_dir}: mesh object names must be unique across parts")

    if asset_dir.exists():
        shutil.rmtree(asset_dir)
    asset_dir.mkdir(parents=True)
    robot = ET.Element("robot", name=spec["name"])
    for n, p in parts.items():
        _link(robot, n, objects[n], frame(n), float(p["mass_kg"]), asset_dir, p.get("collision") == "mesh")
    for n, parent in parent_of.items():
        j = joint_of.get(n)
        joint = ET.SubElement(robot, "joint", name=j["name"] if j else f"{n}_mount", type=j["type"] if j else "fixed")
        _origin(joint, frame(n) - frame(parent))
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=n)
        if j:
            ET.SubElement(joint, "axis", xyz=_xyz(j["axis"]))
            ET.SubElement(joint, "limit", effort="120", velocity="1", lower=f"{j['lower']:.6g}", upper=f"{j['upper']:.6g}")
            ET.SubElement(joint, "dynamics", damping="0.3", friction="0.1")
    ET.indent(robot)
    (asset_dir / "model.urdf").write_text(ET.tostring(robot, encoding="unicode") + "\n", encoding="utf-8", newline="\n")
    if build is not None:
        shutil.copy(build, asset_dir / "model.py")
    if provenance is not None:
        (asset_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    annotations = {"description": store.annotation(description or spec["name"], by)}
    if spec.get("handles"):
        annotations["handles"] = store.annotation(
            [{"link": h["part"], "collision": h["object"]} for h in spec["handles"]], by)
    if spec.get("place_targets"):
        annotations["place_targets"] = store.annotation(
            [{"name": t["name"], "surface": t["surface"]} for t in spec["place_targets"]], by)
    return store.new_asset(asset_dir, spec["kind"], source, annotations, license=license, attribution=attribution,
                           category=category)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("item_dir", type=Path)
    ap.add_argument("asset_dir", type=Path)
    ap.add_argument("--build", type=Path)
    ap.add_argument("--by", default="agent:blender")
    ap.add_argument("--description")
    ap.add_argument("--category", required=True, help="what the item is, e.g. mug or drawer_cabinet")
    a = ap.parse_args()
    data = compile_item(a.item_dir, a.asset_dir, build=a.build, by=a.by, description=a.description,
                        category=a.category)
    print(f"wrote {a.asset_dir} ({data['kind']}, {len(data['build']['files'])} build files)")


if __name__ == "__main__":
    main()
