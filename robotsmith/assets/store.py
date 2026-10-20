"""On-disk asset model v1: ``asset.json`` annotations and append-only ``checks.jsonl``.

Every annotation is ``{value, by, note?}``. Check records bind to the asset
``revision`` and the hash of its build files; a record whose hash no longer
matches the files on disk does not count.
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


def new_asset(asset_dir: Path, kind: str, source_type: str, annotations: dict) -> dict:
    """Write a revision-1 ``asset.json`` covering every file already in ``asset_dir``."""
    files = sorted(p.name for p in asset_dir.iterdir() if p.is_file() and p.name not in (ASSET_FILE, CHECKS_FILE))
    build = [f for f in files if f not in SOURCE_FILES]
    data = {
        "schema_version": 1,
        "id": asset_dir.name,
        "revision": 1,
        "kind": kind,
        "source": {
            "type": source_type,
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


def current_checks(asset_dir: Path, data: dict) -> dict[str, dict]:
    """Latest record per check that is bound to the asset's current revision and build files."""
    path = asset_dir / CHECKS_FILE
    if not path.exists():
        return {}
    current_hash = build_hash(asset_dir, data["build"]["files"])
    latest: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["asset_revision"] == data["revision"] and rec["build_hash"] == current_hash:
            latest[rec["check"]] = rec
    return latest


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

    task_poses: dict = {}
    if "upright" in ann:
        upright = checks.get("upright")
        task_poses["upright"] = {
            "quat_object": ann["upright"],
            "verified": bool(upright and upright["result"] == "pass"),
        }
    urdf = asset_dir / "model.urdf"
    mesh_rec = checks.get("mesh_consistency")
    defaults = AssetMetadata()
    return AssetMetadata(
        friction=ann.get("friction", defaults.friction),
        size_cm=derive.size_cm(urdf),
        metric_scale=ann.get("metric_scale", defaults.metric_scale),
        source=data["source"]["type"],
        role="fixture" if data["kind"] == "fixture" else "object",
        description=ann.get("description", ""),
        task_poses=task_poses,
        grasp_policy_bucket=(buckets or {}).get(data["id"]),
        mesh=dict(mesh_rec["metrics"]) if mesh_rec else {},
        handles=derive.handles(urdf, ann.get("handles", [])),
        place_targets=derive.place_targets(urdf, ann.get("place_targets", []), parse_urdf_joints(urdf)),
    )
