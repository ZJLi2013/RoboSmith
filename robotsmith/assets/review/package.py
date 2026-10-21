"""Reviewer packages and admission for generated assets.

A generated asset is compiled into ``assets/pending/<id>/``. ``build_package`` lays out what
a reviewer reads (request, renders, compiled asset, readings, simulator helper); the reviewer
writes ``review.json``; ``admit`` records the verdict as an ``llm_review`` check and, on
accept, moves the asset into ``assets/generated/`` where ``AssetLibrary`` finds it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from robotsmith.assets import derive, store
from robotsmith.assets.schema import parse_urdf_collision_boxes, parse_urdf_joints

HERE = Path(__file__).parent


def readings(asset_dir: Path) -> dict:
    """What the compiled asset declares and what the scenario runtime derives from it."""
    urdf = asset_dir / "model.urdf"
    data = store.read_asset(asset_dir)
    meta = store.load_metadata(asset_dir)
    rest = derive.link_transforms(urdf)
    boxes = []
    for b in parse_urdf_collision_boxes(urdf):
        lo, hi = derive.box_bounds(b, rest[b.link])
        boxes.append({"name": b.name, "link": b.link, "size": list(b.size),
                      "bounds_at_rest": [lo.round(4).tolist(), hi.round(4).tolist()]})
    mass = {}
    for link in ET.parse(urdf).getroot().iter("link"):
        m = link.find("inertial/mass")
        mass[link.get("name")] = float(m.get("value")) if m is not None else None
    return {
        "id": data["id"],
        "kind": data["kind"],
        "size_cm": meta.size_cm,
        "joints": [{"name": j.name, "type": j.type, "parent": j.parent, "child": j.child,
                    "axis": list(j.axis), "lower": j.lower, "upper": j.upper}
                   for j in parse_urdf_joints(urdf)],
        "link_mass_kg": mass,
        "collision_boxes": boxes,
        "handles": meta.handles,
        "place_targets": meta.place_targets,
        "frame": "asset frame = world frame in the simulator: metres, +Z up, front faces +X, base on z=0",
    }


def build_package(asset_dir: Path, request: str, renders: Path, out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy(HERE / "REVIEW.md", out / "REVIEW.md")
    (out / "request.md").write_text(request, encoding="utf-8")
    shutil.copytree(asset_dir, out / "asset")
    shutil.copytree(renders, out / "renders")
    (out / "tools").mkdir()
    shutil.copy(HERE / "probe.py", out / "tools" / "probe.py")
    (out / "readings.json").write_text(json.dumps(readings(asset_dir), indent=2), encoding="utf-8")
    return out


def admit(asset_dir: Path, review: dict, generated_root: Path, by: str) -> Path:
    """Record ``review`` on the pending asset; on accept move it under ``generated_root``."""
    accepted = review["decision"] == "accept"
    store.append_check(asset_dir, "llm_review", "pass" if accepted else "fail",
                       {"claims": review["claims"], "feedback": review.get("feedback", "")}, by=by)
    if not accepted:
        return asset_dir
    dest = generated_root / asset_dir.name
    if dest.exists():
        shutil.rmtree(dest)
    generated_root.mkdir(parents=True, exist_ok=True)
    return Path(shutil.move(str(asset_dir), dest))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("package")
    p.add_argument("asset_dir", type=Path)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--renders", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("admit")
    a.add_argument("asset_dir", type=Path)
    a.add_argument("--review", type=Path, required=True)
    a.add_argument("--generated", type=Path, required=True)
    a.add_argument("--by", required=True)
    args = ap.parse_args()
    if args.cmd == "package":
        print(build_package(args.asset_dir, args.request.read_text(encoding="utf-8"), args.renders, args.out))
    else:
        print(admit(args.asset_dir, json.loads(args.review.read_text(encoding="utf-8")), args.generated, args.by))


if __name__ == "__main__":
    main()
