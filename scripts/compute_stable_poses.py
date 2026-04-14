#!/usr/bin/env python3
"""Compute stable resting poses for assets and write to metadata.json.

Uses trimesh to estimate poses where an object can rest stably on a flat
surface. For primitive-only assets (no mesh file), computes trivial upright
poses from URDF geometry.

Usage:
    python scripts/compute_stable_poses.py                    # all assets
    python scripts/compute_stable_poses.py --name mug_01      # single asset
    python scripts/compute_stable_poses.py --dir assets/objects/mug_01
    python scripts/compute_stable_poses.py --dry-run           # preview only
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from robotsmith.assets.library import AssetLibrary
from robotsmith.assets.schema import AssetMetadata


def _stable_poses_from_mesh(mesh: trimesh.Trimesh, n_samples: int = 500) -> list[dict]:
    """Compute stable resting poses via trimesh."""
    transforms, probs = mesh.compute_stable_poses(n_samples=n_samples)
    poses = []
    for T, p in zip(transforms, probs):
        if p < 0.01:
            continue
        rot = T[:3, :3]
        pos = T[:3, 3]
        z = float(pos[2]) - float(mesh.bounds[0, 2])
        quat = trimesh.transformations.quaternion_from_matrix(T)
        poses.append({
            "z": round(z, 6),
            "quat": [round(float(q), 6) for q in quat],
            "probability": round(float(p), 4),
        })
    poses.sort(key=lambda x: x["probability"], reverse=True)
    return poses


def _stable_poses_from_urdf_primitive(urdf_path: Path) -> list[dict]:
    """Extract trivial upright pose from URDF primitive geometry."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for collision in root.iter("collision"):
        geom = collision.find("geometry")
        if geom is None:
            continue
        origin = collision.find("origin")
        oz = 0.0
        if origin is not None:
            xyz = origin.get("xyz", "0 0 0").split()
            oz = float(xyz[2])

        box = geom.find("box")
        if box is not None:
            sz = float(box.get("size").split()[2])
            z = sz / 2.0
            return [{"z": round(z, 6), "quat": [1.0, 0.0, 0.0, 0.0], "probability": 1.0}]

        cyl = geom.find("cylinder")
        if cyl is not None:
            length = float(cyl.get("length"))
            z = length / 2.0
            return [{"z": round(z, 6), "quat": [1.0, 0.0, 0.0, 0.0], "probability": 1.0}]

        sphere = geom.find("sphere")
        if sphere is not None:
            r = float(sphere.get("radius"))
            return [{"z": round(r, 6), "quat": [1.0, 0.0, 0.0, 0.0], "probability": 1.0}]

    return [{"z": 0.025, "quat": [1.0, 0.0, 0.0, 0.0], "probability": 1.0}]


MAX_FACES_FOR_STABLE_POSE = 10000


def compute_for_asset(asset_dir: Path, dry_run: bool = False) -> list[dict]:
    """Compute stable poses for a single asset directory."""
    mesh_candidates = ["collision.obj", "visual.obj", "visual.glb"]
    mesh_path = None
    for name in mesh_candidates:
        p = asset_dir / name
        if p.exists():
            mesh_path = p
            break

    if mesh_path is not None:
        print(f"  Using mesh: {mesh_path.name}")
        mesh = trimesh.load(mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            )
        if len(mesh.faces) > MAX_FACES_FOR_STABLE_POSE:
            hull = mesh.convex_hull
            print(f"  High-poly ({len(mesh.faces)} faces), using convex hull ({len(hull.faces)} faces)")
            mesh = hull
        poses = _stable_poses_from_mesh(mesh)
    else:
        urdf_path = asset_dir / "model.urdf"
        if urdf_path.exists():
            print(f"  No mesh file, using URDF primitive geometry")
            poses = _stable_poses_from_urdf_primitive(urdf_path)
        else:
            print(f"  WARNING: no mesh or URDF found, skipping")
            return []

    if not poses:
        print(f"  WARNING: no stable poses found, using default upright")
        poses = [{"z": 0.025, "quat": [1.0, 0.0, 0.0, 0.0], "probability": 1.0}]

    print(f"  Found {len(poses)} stable poses (top p={poses[0]['probability']:.2f})")

    if not dry_run:
        meta_path = asset_dir / "metadata.json"
        if meta_path.exists():
            meta = AssetMetadata.load(meta_path)
        else:
            meta = AssetMetadata()
        meta.stable_poses = poses
        meta.save(meta_path)
        print(f"  Updated {meta_path}")

    return poses


def main():
    parser = argparse.ArgumentParser(description="Compute stable poses for assets")
    parser.add_argument("--name", type=str, default=None, help="Process single asset by name")
    parser.add_argument("--dir", type=str, default=None, help="Process single asset directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't write")
    args = parser.parse_args()

    if args.dir:
        asset_dir = Path(args.dir)
        print(f"Computing stable poses for {asset_dir.name}...")
        compute_for_asset(asset_dir, dry_run=args.dry_run)
        return

    assets_root = REPO_ROOT / "assets"

    if args.name:
        lib = AssetLibrary(assets_root)
        asset = lib.get(args.name)
        if asset is None:
            print(f"ERROR: asset '{args.name}' not found")
            sys.exit(1)
        print(f"Computing stable poses for {asset.name}...")
        compute_for_asset(asset.root_dir, dry_run=args.dry_run)
        return

    objects_dir = assets_root / "objects"
    if not objects_dir.exists():
        print("ERROR: assets/objects/ not found")
        sys.exit(1)

    asset_dirs = sorted(d for d in objects_dir.iterdir() if d.is_dir())
    print(f"Computing stable poses for {len(asset_dirs)} assets in objects/...\n")
    for asset_dir in asset_dirs:
        print(f"[{asset_dir.name}]")
        compute_for_asset(asset_dir, dry_run=args.dry_run)
        print()

    print("Done!")


if __name__ == "__main__":
    main()
