"""On-disk asset model v1: ``asset.json`` annotations and append-only ``checks.jsonl``.

Every annotation is ``{value, by, note?}``. Check records bind to the asset
``revision`` and the hash of its build files; a record whose hash no longer
matches the files on disk does not count.

An asset is admitted when its current build has an ``llm_review`` pass, its
license may be redistributed, and it is not a duplicate of an admitted asset
(``robotsmith.assets.dedup``); otherwise it is a draft. Scenarios load, and the
package carries, admitted assets only.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robotsmith.assets import derive
from robotsmith.assets.schema import AssetMetadata, parse_urdf_joints

ASSET_FILE = "asset.json"
CHECKS_FILE = "checks.jsonl"
BUCKETS_FILE = "grasp_policy_buckets.json"
CHECK_VERSION = "1"
SOURCE_FILES = frozenset({"provenance.json", "model.py", "generation.json", "reference.png"})
REDISTRIBUTABLE = frozenset({"CC0-1.0", "CC-BY-4.0", "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"})


class AssetNotAdmitted(ValueError):
    pass


def build_hash(asset_dir: Path, files: list[str]) -> str:
    h = hashlib.sha256()
    for rel in sorted(files):
        digest = hashlib.sha256((asset_dir / rel).read_bytes()).hexdigest()
        h.update(rel.encode() + b"\0" + digest.encode() + b"\n")
    return h.hexdigest()


def read_asset(asset_dir: Path) -> dict:
    return json.loads((asset_dir / ASSET_FILE).read_text(encoding="utf-8"))


def write_asset(asset_dir: Path, data: dict) -> None:
    (asset_dir / ASSET_FILE).write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def new_asset(asset_dir: Path, kind: str, source_type: str, annotations: dict,
              license: str | None = None, attribution: str | None = None, category: str | None = None) -> dict:
    """Write a revision-1 ``asset.json`` covering every file already in ``asset_dir``.

    ``license`` is an SPDX expression; without one the asset cannot be published.
    """
    files = sorted(p.name for p in asset_dir.iterdir() if p.is_file() and p.name not in (ASSET_FILE, CHECKS_FILE))
    build = [f for f in files if f not in SOURCE_FILES]
    source = {"type": source_type}
    if license:
        source["license"] = license
    if attribution:
        source["attribution"] = attribution
    data = {
        "schema_version": 1,
        "id": asset_dir.name,
        "revision": 1,
        "kind": kind,
        **({"category": category} if category else {}),
        "source": source | {
            "rebuildable": False,
            "files": [f for f in files if f in SOURCE_FILES],
        },
        "build": {"files": build, "hash": build_hash(asset_dir, build)},
        "annotations": annotations,
    }
    write_asset(asset_dir, data)
    return data


def annotation(value: Any, by: str, note: str | None = None) -> dict:
    a = {"value": value, "by": by}
    if note:
        a["note"] = note
    return a


def set_annotation(asset_dir: Path, key: str, value: Any, by: str, note: str | None = None) -> None:
    data = read_asset(asset_dir)
    data["annotations"][key] = annotation(value, by, note)
    write_asset(asset_dir, data)


def append_check(
    asset_dir: Path,
    check: str,
    result: str,
    metrics: dict,
    by: str = "harness",
    embodiment: str | None = None,
) -> dict:
    data = read_asset(asset_dir)
    record = {
        "check": check,
        "version": CHECK_VERSION,
        "asset_revision": data["revision"],
        "build_hash": data["build"]["hash"],
        "embodiment": embodiment,
        "result": result,
        "metrics": metrics,
        "by": by,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(asset_dir / CHECKS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def current_checks(asset_dir: Path, data: dict, digest: str | None = None) -> dict[str, dict]:
    """Latest record per check bound to the asset's current revision and build files (``digest`` if given)."""
    path = asset_dir / CHECKS_FILE
    if not path.exists():
        return {}
    current_hash = digest or build_hash(asset_dir, data["build"]["files"])
    latest: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["asset_revision"] == data["revision"] and rec["build_hash"] == current_hash:
            latest[rec["check"]] = rec
    return latest


def redistributable(license: str | None) -> bool:
    return bool(license) and all(part.strip() in REDISTRIBUTABLE for part in license.split(" AND "))


def draft_reasons(asset_dir: Path, data: dict | None = None) -> list[str]:
    """Why the asset is a draft; empty when it is admitted. Reads the lock, so build files need not be present."""
    data = data or read_asset(asset_dir)
    reasons = []
    license = data["source"].get("license")
    if not license:
        reasons.append("source.license missing")
    elif not redistributable(license):
        reasons.append(f"license {license} not redistributable")
    checks = current_checks(asset_dir, data, data["build"]["hash"])
    review = checks.get("llm_review")
    if review is None:
        reasons.append("no llm_review on the current build")
    elif review["result"] != "pass":
        reasons.append("llm_review failed")
    return reasons + duplicate_reasons(asset_dir, checks)


def duplicate_reasons(asset_dir: Path, checks: dict[str, dict]) -> list[str]:
    """Candidates count while they are unchanged since the check and still admitted; a rebuilt
    candidate is the later asset, and its own check covers the pair."""
    dedup = checks.get("dedup_check")
    if dedup is None:
        return ["no dedup_check on the current build"]
    judged = {p["id"]: p for p in checks.get("dedup_review", {}).get("metrics", {}).get("pairs", [])}
    reasons, unjudged = [], []
    for c in dedup["metrics"]["candidates"]:
        other = asset_dir.parent / c["id"]
        if not (other / ASSET_FILE).exists():
            continue
        o = read_asset(other)
        if (o["revision"], o["build"]["hash"]) != (c["revision"], c["build_hash"]) or draft_reasons(other, o):
            continue
        if c["same_build_hash"]:
            reasons.append(f"same build hash as {c['id']}")
            continue
        p = judged.get(c["id"])
        if p is None or (p["revision"], p["build_hash"], p["rule"]) != (c["revision"], c["build_hash"], dedup["metrics"]["rule"]):
            unjudged.append(c["id"])
        elif p["verdict"] == "duplicate":
            reasons.append(f"duplicate of {c['id']} (dedup_review)")
    if unjudged:
        reasons.append(f"near-duplicate candidates without dedup_review: {', '.join(unjudged)}")
    return reasons


def load_buckets(assets_root: Path) -> dict:
    path = assets_root / BUCKETS_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_bucket(assets_root: Path, asset_name: str, bucket: str | list[str]) -> None:
    buckets = load_buckets(assets_root)
    buckets[asset_name] = bucket
    (assets_root / BUCKETS_FILE).write_text(
        json.dumps(dict(sorted(buckets.items())), indent=2) + "\n", encoding="utf-8"
    )


def load_metadata(asset_dir: Path, buckets: dict | None = None) -> AssetMetadata:
    """Assemble the in-memory ``AssetMetadata`` view from ``asset.json`` + current checks."""
    data = read_asset(asset_dir)
    ann = {k: v["value"] for k, v in data["annotations"].items()}
    checks = current_checks(asset_dir, data)
    urdf = asset_dir / "model.urdf"
    mesh_rec = checks.get("mesh_consistency")
    defaults = AssetMetadata()
    return AssetMetadata(
        friction=ann.get("friction", defaults.friction),
        size_cm=derive.size_cm(urdf),
        source=data["source"]["type"],
        role="fixture" if data["kind"] == "fixture" else "object",
        description=ann.get("description", ""),
        upright_verified=checks.get("llm_review", {}).get("result") == "pass",
        grasp_policy_bucket=(buckets or {}).get(data["id"]),
        mesh=dict(mesh_rec["metrics"]) if mesh_rec else {},
        handles=derive.handles(urdf, ann.get("handles", [])),
        place_targets=derive.place_targets(urdf, ann.get("place_targets", []), parse_urdf_joints(urdf)),
    )
