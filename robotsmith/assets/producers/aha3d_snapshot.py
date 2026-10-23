"""AHa-3D scene export -> snapshot scenario.

The robot takes one spot in the room, facing ``--yaw`` degrees from room +X; the furniture it works
at becomes its table, and the roots it can use become assets at their recorded positions:

    python -m robotsmith.assets.producers.aha3d_snapshot <export_dir> --support furniture/desk \\
        --base 0.62 -1.30 --scene office48 --scenario scenarios/office48_desk.yaml --task task.yaml

``<export_dir>`` is the output of ``aha3d/export_roots.py``. Kept roots are compiled into
``assets/pending/<category>_<NN>/`` for review, the category being the item name without its
trailing index; their placements go to the layout ``scenes/<scene>.yaml`` beside the scenarios
directory, and the scenario names that layout. ``mapping.json`` beside the export records every
root's fate.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml

from robotsmith.assets.producers.blender_item import compile_item
from robotsmith.scenario.relations import ROBOT_BASE_HALF_M, table_size
from robotsmith.scenes.config import DEFAULT_WORKSPACE_XY, table_center_xy

FLOOR_REACH_M = 1.2
MIN_HEIGHT_M = 0.005
REST_TOL_M = 0.01
FLOOR_TOL_M = 0.05


def to_robot(xy, base_xy, yaw_deg: float = 0.0) -> np.ndarray:
    """Room-frame xy point(s) in the robot frame: origin at the base, +X along the robot's heading."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    return (np.asarray(xy, dtype=float) - np.asarray(base_xy, dtype=float)) @ np.array([[c, -s], [s, c]])


def box_to_robot(lo, hi, base_xy, yaw_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Robot-frame xy bounds of a room-frame box (bounds of its rotated corners)."""
    corners = to_robot([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]], base_xy, yaw_deg)
    return corners.min(axis=0), corners.max(axis=0)


def plan(scene: dict, support: str, base_xy, yaw_deg: float = 0.0) -> dict:
    """Each root's fate: ``table`` (replaced by the robot's table), ``support`` (rests on it),
    ``floor`` or ``dropped`` with a reason; positions in the robot frame."""
    base = np.asarray(base_xy, dtype=float)
    roots = {r["instance_id"]: r for r in scene["roots"]}
    desk = roots[support]
    rel = {i: box_to_robot(r["lo"], r["hi"], base, yaw_deg) for i, r in roots.items() if "lo" in r}
    out = {}
    for iid, r in roots.items():
        if iid == support:
            out[iid] = {"decision": "table"}
        elif "skipped" in r:
            out[iid] = {"decision": "dropped", "reason": r["skipped"]}
        elif r["hi"][2] - r["lo"][2] < MIN_HEIGHT_M:
            out[iid] = {"decision": "dropped", "reason": "thinner than 5 mm"}
        elif (abs(r["lo"][2] - desk["hi"][2]) < REST_TOL_M
              and all(desk["lo"][k] <= r["origin"][k] <= desk["hi"][k] for k in (0, 1))):
            out[iid] = {"decision": "support"}
        elif r["semantic_class"] not in ("furniture", "prop"):
            out[iid] = {"decision": "dropped", "reason": f"{r['semantic_class']} is not imported"}
        elif r["lo"][2] >= FLOOR_TOL_M:
            out[iid] = {"decision": "dropped", "reason": "not on the table or the floor"}
        elif np.all(rel[iid][0] < ROBOT_BASE_HALF_M) and np.all(rel[iid][1] > -ROBOT_BASE_HALF_M):
            out[iid] = {"decision": "dropped", "reason": "stands where the robot stands"}
        else:
            gap = np.maximum(np.maximum(rel[iid][0], -rel[iid][1]), 0.0)
            out[iid] = ({"decision": "floor"} if np.linalg.norm(gap) <= FLOOR_REACH_M
                        else {"decision": "dropped", "reason": f"more than {FLOOR_REACH_M} m from the robot"})
        if "origin" in r:
            out[iid]["anchor_xy"] = [round(float(v), 4) + 0.0 for v in to_robot(r["origin"][:2], base, yaw_deg)]
    size = table_size([rel[i] for i, d in out.items() if d["decision"] == "support"])
    cx, cy = table_center_xy(DEFAULT_WORKSPACE_XY)
    t_lo, t_hi = np.array([cx - size[0] / 2, cy - size[1] / 2]), np.array([cx + size[0] / 2, cy + size[1] / 2])
    for iid, d in out.items():
        if d["decision"] == "floor" and np.all(rel[iid][0] < t_hi) and np.all(rel[iid][1] > t_lo):
            out[iid] = {"decision": "dropped", "reason": "overlaps the robot's table", "anchor_xy": d["anchor_xy"]}
    return {"support": support, "base_xy_room": [float(v) for v in base], "yaw_deg": float(yaw_deg),
            "table_xy": [t_lo.round(3).tolist(), t_hi.round(3).tolist()], "roots": out}


def _new_id(category: str, taken: set[str]) -> str:
    n = 1
    while f"{category}_{n:02d}" in taken:
        n += 1
    taken.add(f"{category}_{n:02d}")
    return f"{category}_{n:02d}"


def _dump(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=None), encoding="utf-8", newline="\n")


def import_snapshot(export_dir: Path, *, support: str, base_xy, scene_name: str, assets: Path, scenario: Path,
                    task: dict, provenance: dict, yaw_deg: float = 0.0, license: str = "Apache-2.0",
                    attribution: str | None = None) -> dict:
    scene = json.loads((export_dir / "scene.json").read_text(encoding="utf-8"))
    mapping = plan(scene, support, base_xy, yaw_deg)
    roots = {r["instance_id"]: r for r in scene["roots"]}
    taken = {p.name for p in [*assets.glob("*"), *assets.glob("pending/*")] if p.is_dir()}
    objects = []
    for iid, d in mapping["roots"].items():
        if d["decision"] not in ("support", "floor"):
            continue
        r = roots[iid]
        category = re.sub(r"_\d+$", "", r["item"])
        d["asset"] = _new_id(category, taken)
        compile_item(export_dir / r["item"], assets / "pending" / d["asset"], by="aha3d", source="aha3d",
                     description=f"{iid} from an AHa-3D scene ({r['semantic_class']})",
                     provenance=provenance | {"instance_id": iid, "semantic_class": r["semantic_class"]},
                     license=license, attribution=attribution, category=category)
        objects.append({"id": r["item"], "asset": d["asset"], "on": d["decision"], "anchor_xy": d["anchor_xy"],
                        "yaw_deg": 0.0 - yaw_deg})
    _dump(scenario.parent.parent / "scenes" / f"{scene_name}.yaml",
          {"scene": scene_name, "support": "table_simple", "objects": objects})
    _dump(scenario, {"scenario": scenario.stem, "robot": "franka", "scene": scene_name, "task": task})
    (export_dir / "mapping.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_dir", type=Path)
    ap.add_argument("--support", required=True, help="instance_id of the furniture the robot works at")
    ap.add_argument("--base", type=float, nargs=2, required=True, help="robot base xy in the room frame")
    ap.add_argument("--yaw", type=float, default=0.0, help="robot heading in the room frame, degrees from +X")
    ap.add_argument("--scene", required=True, help="layout name, written to scenes/<scene>.yaml")
    ap.add_argument("--assets", type=Path, default=Path("assets"))
    ap.add_argument("--scenario", type=Path, required=True)
    ap.add_argument("--task", type=Path, required=True, help="YAML with the scenario's task section")
    ap.add_argument("--provenance", type=Path, help="JSON merged into every asset's provenance.json")
    ap.add_argument("--license", default="Apache-2.0", help="SPDX expression of the scene's assets")
    ap.add_argument("--attribution", help="credit line the license requires, e.g. the source video")
    a = ap.parse_args()
    prov = json.loads(a.provenance.read_text(encoding="utf-8")) if a.provenance else {}
    mapping = import_snapshot(a.export_dir, support=a.support, base_xy=a.base, scene_name=a.scene, assets=a.assets,
                              scenario=a.scenario, task=yaml.safe_load(a.task.read_text(encoding="utf-8")),
                              provenance=prov, yaw_deg=a.yaw, license=a.license, attribution=a.attribution)
    kept = {i: d for i, d in mapping["roots"].items() if d["decision"] in ("support", "floor")}
    print(json.dumps(kept, indent=1))


if __name__ == "__main__":
    main()
