"""Asset admission audit and metadata update.

This is an asset-level gate: it updates persistent metadata before scene build.
It does not do per-episode grasp candidate feasibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import trimesh.transformations as tf

from robotsmith.assets.schema import Asset

SUPPORT_ASSETS = {"table_simple", "plane"}


@dataclass(frozen=True)
class AuditConfig:
    # Conservative Franka 8cm gripper stroke limit.
    max_width: float = 0.075
    # Scale recommendation target with about 10% grasp-width margin.
    target_width: float = 0.0675
    # Lightweight visual/collision surface consistency check.
    consistency_sample_points: int = 800
    consistency_grid: int = 24
    consistency_min_coverage: float = 0.82


def _load_mesh(path: Path | None):
    if path is None or not path.exists():
        return None
    import trimesh

    mesh = trimesh.load(str(path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    if not isinstance(mesh, trimesh.Trimesh):
        return None
    return mesh


def _mesh_extents(path: Path | None) -> list[float] | None:
    mesh = _load_mesh(path)
    if mesh is None:
        return None
    return [float(v) for v in mesh.bounding_box.extents]


def _sample_surface_points(mesh, n_points: int, seed: int) -> Any:
    import numpy as np
    import trimesh

    state = np.random.get_state()
    try:
        np.random.seed(seed)
        points, _ = trimesh.sample.sample_surface(mesh, int(n_points))
    finally:
        np.random.set_state(state)
    return points


def _surface_voxel_mask(points, lo, span, grid: int):
    import numpy as np

    scaled = (points - lo) / span
    idx = np.floor(scaled * grid).astype(int)
    idx = np.clip(idx, 0, grid - 1)
    mask = np.zeros((grid, grid, grid), dtype=bool)
    mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return mask


def _dilate_voxels(mask):
    import numpy as np

    grid = mask.shape[0]
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    out = np.zeros_like(mask)
    for dx in range(3):
        for dy in range(3):
            for dz in range(3):
                out |= padded[dx:dx + grid, dy:dy + grid, dz:dz + grid]
    return out


def _visual_collision_consistency(
    visual_mesh,
    collision_mesh,
    config: AuditConfig,
) -> dict[str, Any]:
    """Compare visual/collision surfaces using a small voxelized point sample.

    This is intentionally a coarse admission check. It catches large structural
    mismatches such as a visual mug handle missing from the collision mesh,
    without requiring scipy/rtree proximity dependencies.
    """
    import numpy as np

    if visual_mesh is None or collision_mesh is None:
        return {"available": False, "ok": True}

    n = int(config.consistency_sample_points)
    grid = int(config.consistency_grid)
    v_pts = _sample_surface_points(visual_mesh, n, seed=17)
    c_pts = _sample_surface_points(collision_mesh, n, seed=23)
    all_pts = np.vstack([v_pts, c_pts])
    lo = all_pts.min(axis=0)
    hi = all_pts.max(axis=0)
    span = hi - lo
    if float(np.min(span)) <= 1e-9:
        return {"available": True, "ok": False, "issue": "degenerate_bounds"}

    v_mask = _surface_voxel_mask(v_pts, lo, span, grid)
    c_mask = _surface_voxel_mask(c_pts, lo, span, grid)
    v_dilated = _dilate_voxels(v_mask)
    c_dilated = _dilate_voxels(c_mask)

    v_count = max(int(np.count_nonzero(v_mask)), 1)
    c_count = max(int(np.count_nonzero(c_mask)), 1)
    visual_to_collision = float(np.count_nonzero(v_mask & c_dilated) / v_count)
    collision_to_visual = float(np.count_nonzero(c_mask & v_dilated) / c_count)
    intersection = int(np.count_nonzero(v_mask & c_mask))
    union = max(int(np.count_nonzero(v_mask | c_mask)), 1)
    iou = float(intersection / union)

    ok = (
        visual_to_collision >= config.consistency_min_coverage
        and collision_to_visual >= config.consistency_min_coverage
    )
    return {
        "available": True,
        "ok": bool(ok),
        "method": "surface_voxel_coverage",
        "sample_points": n,
        "grid": grid,
        "visual_to_collision_coverage": round(visual_to_collision, 4),
        "collision_to_visual_coverage": round(collision_to_visual, 4),
        "surface_voxel_iou": round(iou, 4),
        "min_coverage": config.consistency_min_coverage,
    }


def _upright_quat(asset: Asset) -> list[float]:
    pose = asset.metadata.task_poses.get("upright", {})
    quat = pose.get("quat_object") or pose.get("quat") or pose.get("quaternion")
    return [float(v) for v in (quat or [1.0, 0.0, 0.0, 0.0])]


def _extents(asset: Asset, scale: float, quat: list[float]) -> list[float]:
    import numpy as np

    sx, sy, sz = [float(v) / 100.0 * scale for v in asset.metadata.size_cm]
    rmat = tf.quaternion_matrix(quat)[:3, :3]
    corners = np.array(
        [
            [dx * sx / 2, dy * sy / 2, dz * sz / 2]
            for dx in (-1, 1)
            for dy in (-1, 1)
            for dz in (-1, 1)
        ]
    )
    rotated = corners @ rmat.T
    return list((rotated.max(axis=0) - rotated.min(axis=0)).astype(float))


def _grasp_xy(asset: Asset, scale: float, quat: list[float]) -> list[float]:
    ex, ey, _ = _extents(asset, scale, quat)
    return [float(ex), float(ey)]


def audit_mesh(asset: Asset, config: AuditConfig | None = None) -> dict[str, Any]:
    """Return compact mesh audit suitable for ``metadata.mesh``."""

    config = config or AuditConfig()
    primitive = asset.metadata.source == "builtin_primitive"
    visual_mesh = _load_mesh(asset.visual_mesh)
    collision_mesh = _load_mesh(asset.collision_mesh)
    v_ext = (
        [float(v) for v in visual_mesh.bounding_box.extents]
        if visual_mesh is not None else None
    )
    c_ext = (
        [float(v) for v in collision_mesh.bounding_box.extents]
        if collision_mesh is not None else None
    )
    issues: list[str] = []

    if not primitive:
        if v_ext is None:
            issues.append("no_visual")
        if c_ext is None:
            issues.append("no_collision")

    ratio = None
    size_m = [float(v) / 100.0 for v in asset.metadata.size_cm]
    if v_ext is not None and all(v > 1e-9 for v in size_m):
        ratio = [v_ext[i] / size_m[i] for i in range(3)]
        if max(ratio) > 1.35 or min(ratio) < 0.65:
            issues.append("size_mismatch")

    consistency = _visual_collision_consistency(visual_mesh, collision_mesh, config)
    if not primitive and consistency.get("available") and not consistency.get("ok"):
        issues.append("visual_collision_mismatch")

    return {
        "issues": issues,
        "visual_extents": [round(float(v), 5) for v in v_ext] if v_ext else None,
        "collision_extents": [round(float(v), 5) for v in c_ext] if c_ext else None,
        "visual_collision_consistency": consistency,
    }


def _audit_metric_scale(asset: Asset, config: AuditConfig | None = None) -> dict[str, Any]:
    """Return compact metric-scale audit and recommendation."""

    config = config or AuditConfig()
    metric_scale = float(asset.metadata.metric_scale)
    quat = _upright_quat(asset)
    xy = _grasp_xy(asset, metric_scale, quat)
    width = max(xy)
    rec = (
        min(1.0, metric_scale * config.target_width / width)
        if width > 1e-9
        else 1.0
    )
    return {
        "ok": width <= config.max_width,
        "xy": [round(float(v), 5) for v in xy],
        "rec": round(float(rec), 5),
    }


def audit_asset(asset: Asset, config: AuditConfig | None = None) -> dict[str, Any]:
    mesh = audit_mesh(asset, config)
    metric_scale = _audit_metric_scale(asset, config)
    return {
        "asset": asset.name,
        "mesh": mesh,
        "metric_scale": metric_scale,
    }


def audit_assets(
    assets: Iterable[Asset],
    config: AuditConfig | None = None,
) -> dict[str, Any]:
    rows = [audit_asset(asset, config) for asset in assets]
    return {
        "summary": {
            "n": len(rows),
            "mesh": [row["asset"] for row in rows if row["mesh"]["issues"]],
            "wide": [row["asset"] for row in rows if not row["metric_scale"]["ok"]],
        },
        "assets": rows,
    }


def audit_and_update(
    asset: Asset,
    config: AuditConfig | None = None,
) -> dict[str, Any]:
    """Audit one asset and persist ``metadata.mesh`` / ``metric_scale``."""

    row = audit_asset(asset, config)
    meta_path = asset.root_dir / "metadata.json"
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    data["mesh"] = row["mesh"]
    rec = float(row["metric_scale"]["rec"])
    data["metric_scale"] = rec
    meta_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    asset.metadata.mesh = row["mesh"]
    asset.metadata.metric_scale = float(data["metric_scale"])
    return row
