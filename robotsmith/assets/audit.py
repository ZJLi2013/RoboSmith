"""Asset admission audit and check records.

This is an asset-level gate: it records checks before scene build.
It does not do per-episode grasp candidate feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from robotsmith.assets import store
from robotsmith.assets.schema import Asset

SUPPORT_ASSETS = {"table_simple", "plane"}


@dataclass(frozen=True)
class AuditConfig:
    # Conservative Franka 8cm gripper stroke limit.
    max_width: float = 0.075
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


def audit_mesh(asset: Asset, config: AuditConfig | None = None) -> dict[str, Any]:
    """Return compact mesh audit, stored as ``mesh_consistency`` check metrics."""

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

    consistency = _visual_collision_consistency(visual_mesh, collision_mesh, config)
    if not primitive and consistency.get("available") and not consistency.get("ok"):
        issues.append("visual_collision_mismatch")

    return {
        "issues": issues,
        "visual_extents": [round(float(v), 5) for v in v_ext] if v_ext else None,
        "collision_extents": [round(float(v), 5) for v in c_ext] if c_ext else None,
        "visual_collision_consistency": consistency,
    }


def _audit_gripper_fit(asset: Asset, config: AuditConfig | None = None) -> dict[str, Any]:
    """Whether both horizontal extents fit the gripper stroke."""

    config = config or AuditConfig()
    xy = [float(v) / 100.0 for v in asset.metadata.size_cm[:2]]
    return {"ok": max(xy) <= config.max_width, "xy": [round(v, 5) for v in xy]}


def audit_asset(asset: Asset, config: AuditConfig | None = None) -> dict[str, Any]:
    return {
        "asset": asset.name,
        "mesh": audit_mesh(asset, config),
        "gripper_fit": _audit_gripper_fit(asset, config),
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
            "wide": [row["asset"] for row in rows if not row["gripper_fit"]["ok"]],
        },
        "assets": rows,
    }


def audit_and_update(
    asset: Asset,
    config: AuditConfig | None = None,
) -> dict[str, Any]:
    """Audit one asset: append ``mesh_consistency`` / ``gripper_fit`` check records."""

    row = audit_asset(asset, config)
    mesh = row["mesh"]
    fit = row["gripper_fit"]
    store.append_check(asset.root_dir, "mesh_consistency", "fail" if mesh["issues"] else "pass", mesh)
    store.append_check(
        asset.root_dir, "gripper_fit", "pass" if fit["ok"] else "fail",
        {"xy": fit["xy"]}, embodiment="franka",
    )
    asset.metadata.mesh = mesh
    return row
