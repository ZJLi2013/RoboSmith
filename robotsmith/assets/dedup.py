"""Duplicate checks for the admission gate.

``check`` runs once per build, while the build files are present: it lists the admitted assets the
new build may duplicate and records them as a ``dedup_check``, so the gate later reads only the
lock and the records. Same build hash is a duplicate outright; any other candidate needs a
``dedup_review`` verdict (``variant`` or ``duplicate``) from a reviewer who saw both assets.

  python -m robotsmith.assets.dedup check assets/<id>
  python -m robotsmith.assets.dedup judge assets/<id> --verdicts verdicts.json --by agent:<model>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robotsmith.assets import derive, store
from robotsmith.assets.schema import parse_urdf_joints

RULE = "category+kind+joints+place_targets, each edge within 20%"
TOLERANCE = 0.2


def signature(asset_dir: Path) -> dict:
    data = store.read_asset(asset_dir)
    urdf = asset_dir / "model.urdf"
    size = derive.size_cm(urdf)
    if len(size) != 3 or min(size) <= 0:
        raise ValueError(f"{asset_dir.name}: size {size} cannot be compared")
    targets = data["annotations"].get("place_targets", {}).get("value", [])
    return {
        "category": data.get("category"),
        "kind": data["kind"],
        "joints": sorted(j.name for j in parse_urdf_joints(urdf) if j.type != "fixed"),
        "place_targets": sorted(t["name"] for t in targets),
        "size_cm": size,
    }


def near(a: list[float], b: list[float]) -> bool:
    edges = list(zip(sorted(a[:2]), sorted(b[:2]))) + [(a[2], b[2])]
    return all(abs(p - q) <= TOLERANCE * max(p, q) for p, q in edges)


def similar(a: dict, b: dict) -> bool:
    same = all(a[k] == b[k] for k in ("category", "kind", "joints", "place_targets"))
    return same and near(a["size_cm"], b["size_cm"])


def candidates(asset_dir: Path) -> list[dict]:
    """Admitted assets this build may duplicate, as they stand now."""
    data = store.read_asset(asset_dir)
    mine = None
    found = []
    for other in sorted(p.parent for p in asset_dir.parent.glob(f"*/{store.ASSET_FILE}")):
        if other == asset_dir or store.draft_reasons(other):
            continue
        o = store.read_asset(other)
        same_hash = o["build"]["hash"] == data["build"]["hash"]
        if not same_hash and mine is None:
            mine = signature(asset_dir)
        if same_hash or similar(mine, signature(other)):
            found.append({"id": o["id"], "revision": o["revision"], "build_hash": o["build"]["hash"],
                          "same_build_hash": same_hash})
    return found


def check(asset_dir: Path) -> dict:
    """Record the candidates for the current build once; later calls return the existing record."""
    data = store.read_asset(asset_dir)
    existing = store.current_checks(asset_dir, data, data["build"]["hash"]).get("dedup_check")
    if existing:
        return existing
    found = candidates(asset_dir)
    return store.append_check(asset_dir, "dedup_check", "candidates" if found else "none",
                              {"rule": RULE, "candidates": found}, by="check:dedup@1")


def judge(asset_dir: Path, verdicts: dict[str, dict], by: str) -> dict:
    """Record a reviewer's ``{candidate id: {verdict, reason}}`` against the candidates of the current check."""
    data = store.read_asset(asset_dir)
    rec = store.current_checks(asset_dir, data, data["build"]["hash"])["dedup_check"]
    pairs = []
    for c in rec["metrics"]["candidates"]:
        v = verdicts.get(c["id"])
        if v is None or c["same_build_hash"]:
            continue
        if v["verdict"] not in ("variant", "duplicate"):
            raise ValueError(f"{c['id']}: verdict must be variant or duplicate, got {v['verdict']}")
        pairs.append({"id": c["id"], "revision": c["revision"], "build_hash": c["build_hash"], "rule": RULE,
                      "verdict": v["verdict"], "reason": v.get("reason", "")})
    return store.append_check(asset_dir, "dedup_review", "recorded", {"pairs": pairs}, by=by)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("asset_dir", type=Path)
    j = sub.add_parser("judge")
    j.add_argument("asset_dir", type=Path)
    j.add_argument("--verdicts", type=Path, required=True)
    j.add_argument("--by", required=True)
    args = ap.parse_args()
    if args.cmd == "check":
        print(json.dumps(check(args.asset_dir)["metrics"]["candidates"]))
    else:
        judge(args.asset_dir, json.loads(args.verdicts.read_text(encoding="utf-8")), args.by)
    reasons = store.draft_reasons(args.asset_dir)
    print(f"{args.asset_dir.name}: {'draft (' + '; '.join(reasons) + ')' if reasons else 'admitted'}")


if __name__ == "__main__":
    main()
