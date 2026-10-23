"""Review calibration: cases with a known verdict, rebuilt from source on every run.

Only sources live here (requests, Blender build scripts, mutations); assets and renders are rebuilt
locally. Run ``scripts/assets/review_calibration.py`` after changing REVIEW.md, the package or the readings.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

from robotsmith.assets import store
from robotsmith.assets.producers.blender_item import compile_item

HERE = Path(__file__).parent
KIT = HERE.parents[1] / "producers" / "blender"
LIBRARY = HERE.parents[3] / "assets"
DRAWER_DESC = "Single-drawer tabletop cabinet built in Blender by an LLM agent."
# Library glbs are stored +Z up, as the simulator reads them; Blender's glTF import assumes +Y up.
Z_UP_TO_Y_UP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)


def load_cases() -> list[dict]:
    return json.loads((HERE / "cases.json").read_text(encoding="utf-8"))


def _refresh(asset_dir: Path) -> None:
    data = store.read_asset(asset_dir)
    store.new_asset(asset_dir, data["kind"], data["source"]["type"], data["annotations"])


def solid_tray(asset_dir: Path) -> None:
    """Merge the drawer's ``tray_*`` collision boxes into one box that fills the tray."""
    urdf = asset_dir / "model.urdf"
    tree = ET.parse(urdf)
    drawer = next(link for link in tree.getroot().iter("link") if link.get("name") == "drawer")
    cols = [c for c in drawer.findall("collision") if c.get("name", "").startswith("tray_")]
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for c in cols:
        xyz = np.array([float(v) for v in c.find("origin").get("xyz").split()])
        half = np.array([float(v) for v in c.find("geometry/box").get("size").split()]) / 2
        lo, hi = np.minimum(lo, xyz - half), np.maximum(hi, xyz + half)
        drawer.remove(c)
    c = ET.SubElement(drawer, "collision", name="tray_floor")
    ET.SubElement(c, "origin", xyz=" ".join(f"{v:.6g}" for v in (lo + hi) / 2), rpy="0 0 0")
    ET.SubElement(ET.SubElement(c, "geometry"), "box", size=" ".join(f"{v:.6g}" for v in hi - lo))
    ET.indent(tree)
    urdf.write_bytes(ET.tostring(tree.getroot(), encoding="unicode").encode("utf-8") + b"\n")
    _refresh(asset_dir)


def tipped(asset_dir: Path) -> None:
    """Lay a rigid object on its side: visual and collision turn 90 deg about X, then rest on z = 0."""
    rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    col = trimesh.load(str(asset_dir / "collision.obj"), force="mesh", process=False)
    z_min = trimesh.transform_points(col.vertices, rot)[:, 2].min()
    t = trimesh.transformations.translation_matrix([0, 0, -z_min]) @ rot
    for name, geom in (("collision.obj", col), ("visual.glb", trimesh.load(str(asset_dir / "visual.glb")))):
        geom.apply_transform(t)
        geom.export(str(asset_dir / name))
    _refresh(asset_dir)


MUTATIONS = {"solid_tray": solid_tray, "tipped": tipped}


def _blender_ws(ws: Path) -> None:
    if ws.exists():
        shutil.rmtree(ws)
    (ws / "tools").mkdir(parents=True)
    for src, dst in ((KIT / "render.py", ws / "tools/render.py"), (KIT / "run_blender.sh", ws / "run_blender.sh")):
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")


def _blender(blender: list[str], script: str, ws: Path) -> None:
    subprocess.run([*blender, script], cwd=ws, check=True, capture_output=True)


def prepare(case: dict, work: Path, blender: list[str]) -> tuple[Path, Path]:
    """Build the case's asset under ``work/asset`` and its renders; return both directories."""
    ws, asset_dir = work / "ws", work / "asset"
    _blender_ws(ws)
    if asset_dir.exists():
        shutil.rmtree(asset_dir)
    if "build" in case:
        (ws / "build.py").write_text((HERE / case["build"]).read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        _blender(blender, "build.py", ws)
        compile_item(ws / "out/drawer_cabinet", asset_dir, build=HERE / case["build"], by="agent:gpt-5.6-sol",
                     description=DRAWER_DESC)
    else:
        # Earlier verdicts and the unreferenced visual.obj stay out of the reviewer's copy.
        shutil.copytree(LIBRARY / case["library"], asset_dir, ignore=shutil.ignore_patterns("checks.jsonl", "visual.obj"))
        _refresh(asset_dir)
    if "mutate" in case:
        MUTATIONS[case["mutate"]](asset_dir)
    if "library" in case:
        (ws / "out/item").mkdir(parents=True)
        visual = trimesh.load(str(asset_dir / "visual.glb"))
        visual.apply_transform(Z_UP_TO_Y_UP)
        visual.export(str(ws / "out/item/visual.glb"))
    _blender(blender, "tools/render.py", ws)
    return asset_dir, ws / "out/renders"
