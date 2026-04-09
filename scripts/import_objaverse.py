#!/usr/bin/env python3
"""Import high-quality tabletop assets from Objaverse to replace built-in primitives.

Downloads 10 common manipulation objects from Objaverse (via LVIS category
annotations + quality ranking), processes them through mesh_cleanup and
mesh_to_urdf, and saves them as curated built-in assets.

Usage:
    pip install objaverse trimesh numpy
    python scripts/import_objaverse.py                    # import all 10
    python scripts/import_objaverse.py --category mug     # import one
    python scripts/import_objaverse.py --dry-run           # preview candidates
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Target categories ──────────────────────────────────────────────────────
# Maps our asset name -> (LVIS categories to search, name keywords fallback,
#                         target size in meters, mass in kg, physical tags)

ASSET_SPECS: dict[str, dict] = {
    "mug": {
        "lvis": ["mug"],
        "keywords": ["mug", "coffee mug", "ceramic mug"],
        "target_size_m": 0.12,
        "mass_kg": 0.25,
        "friction": 0.6,
        "tags": ["mug", "cup", "red", "container", "grasp"],
        "description": "Ceramic mug (Objaverse)",
    },
    "bowl": {
        "lvis": ["bowl"],
        "keywords": ["bowl", "cereal bowl", "ceramic bowl"],
        "target_size_m": 0.15,
        "mass_kg": 0.30,
        "friction": 0.5,
        "tags": ["bowl", "white", "container", "grasp"],
        "description": "Ceramic bowl (Objaverse)",
    },
    "plate": {
        "lvis": ["plate"],
        "keywords": ["plate", "dinner plate", "ceramic plate"],
        "target_size_m": 0.22,
        "mass_kg": 0.35,
        "friction": 0.5,
        "tags": ["plate", "dish", "round", "flat", "grasp"],
        "description": "Round plate (Objaverse)",
    },
    "fork": {
        "lvis": ["fork"],
        "keywords": ["fork", "dinner fork", "metal fork"],
        "target_size_m": 0.19,
        "mass_kg": 0.04,
        "friction": 0.35,
        "tags": ["fork", "silver", "utensil", "thin", "grasp"],
        "description": "Metal fork (Objaverse)",
    },
    "spoon": {
        "lvis": ["spoon"],
        "keywords": ["spoon", "tablespoon", "metal spoon"],
        "target_size_m": 0.18,
        "mass_kg": 0.035,
        "friction": 0.35,
        "tags": ["spoon", "silver", "utensil", "thin", "grasp"],
        "description": "Metal spoon (Objaverse)",
    },
    "bottle": {
        "lvis": ["bottle"],
        "keywords": ["bottle", "water bottle", "glass bottle"],
        "target_size_m": 0.22,
        "mass_kg": 0.40,
        "friction": 0.4,
        "tags": ["bottle", "tall", "green", "container", "grasp", "pour"],
        "description": "Glass bottle (Objaverse)",
    },
    "can": {
        "lvis": ["can"],
        "keywords": ["can", "soda can", "tin can", "beverage can"],
        "target_size_m": 0.12,
        "mass_kg": 0.35,
        "friction": 0.45,
        "tags": ["can", "soda", "red", "cylinder", "grasp"],
        "description": "Soda can (Objaverse)",
    },
    "block": {
        "lvis": ["block_(wood)"],
        "keywords": ["wooden block", "toy block", "cube", "building block"],
        "target_size_m": 0.05,
        "mass_kg": 0.05,
        "friction": 0.6,
        "tags": ["block", "cube", "red", "stackable", "grasp"],
        "description": "Wooden block (Objaverse)",
    },
    "knife": {
        "lvis": ["knife"],
        "keywords": ["knife", "butter knife", "dinner knife", "table knife"],
        "target_size_m": 0.20,
        "mass_kg": 0.06,
        "friction": 0.3,
        "tags": ["knife", "silver", "utensil", "thin", "grasp"],
        "description": "Table knife (Objaverse)",
    },
    "pan": {
        "lvis": ["frying_pan"],
        "keywords": ["frying pan", "pan", "skillet"],
        "target_size_m": 0.28,
        "mass_kg": 0.80,
        "friction": 0.5,
        "tags": ["pan", "frying pan", "container", "grasp"],
        "description": "Frying pan (Objaverse)",
    },
}


def _score_candidate(ann: dict) -> float:
    """Rank an Objaverse annotation by quality for sim use.

    Prefers: high face count, many likes, no animations, downloadable,
    CC-BY license, reasonable file size.
    """
    if not ann.get("isDownloadable", False):
        return -1
    if ann.get("animationCount", 0) > 0:
        return -1

    score = 0.0

    face_count = ann.get("faceCount", 0) or 0
    if 500 < face_count < 200_000:
        score += min(face_count / 50_000, 1.0) * 40
    elif face_count >= 200_000:
        score += 20
    else:
        return -1

    score += min(ann.get("likeCount", 0), 50) * 0.5
    score += min(ann.get("viewCount", 0), 1000) * 0.01

    license_ = ann.get("license", "")
    if license_ in ("by", "by-sa", "cc0"):
        score += 10
    elif license_ in ("by-nd", "by-nc"):
        score += 5

    if ann.get("staffpickedAt"):
        score += 15

    return score


def find_candidates(
    category: str,
    spec: dict,
    lvis_annotations: dict,
    all_annotations: dict,
    top_k: int = 5,
) -> list[tuple[str, dict, float]]:
    """Find top-K candidate UIDs for a category."""
    candidate_uids: set[str] = set()

    for lvis_cat in spec["lvis"]:
        if lvis_cat in lvis_annotations:
            candidate_uids.update(lvis_annotations[lvis_cat])

    if len(candidate_uids) < 3:
        for uid, ann in all_annotations.items():
            name_lower = ann.get("name", "").lower()
            tags = [t["name"].lower() for t in ann.get("tags", [])]
            for kw in spec["keywords"]:
                if kw.lower() in name_lower or kw.lower() in tags:
                    candidate_uids.add(uid)
                    break
            if len(candidate_uids) >= 200:
                break

    scored: list[tuple[str, dict, float]] = []
    for uid in candidate_uids:
        ann = all_annotations.get(uid)
        if ann is None:
            continue
        s = _score_candidate(ann)
        if s > 0:
            scored.append((uid, ann, s))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]


def process_asset(
    uid: str,
    glb_path: str,
    category: str,
    spec: dict,
    output_dir: Path,
    ann: dict,
) -> Path:
    """Load GLB, cleanup, convert to URDF, write metadata."""
    import trimesh
    from robotsmith.gen.mesh_cleanup import cleanup_mesh
    from robotsmith.gen.mesh_to_urdf import mesh_to_urdf
    from robotsmith.assets.schema import AssetMetadata

    scene_or_mesh = trimesh.load(glb_path, force="mesh")
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in scene_or_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    else:
        mesh = scene_or_mesh

    mesh, stats = cleanup_mesh(mesh, remove_base=True, recenter=True)
    print(f"  Cleanup: {stats}")

    asset_dir = output_dir / category
    asset_dir.mkdir(parents=True, exist_ok=True)

    mesh_to_urdf(
        mesh,
        asset_dir,
        name=category,
        target_size_m=spec["target_size_m"],
        mass_kg=spec["mass_kg"],
        density_kg_m3=800.0,
        visual_format="glb",
    )

    meta = AssetMetadata(
        mass_kg=spec["mass_kg"],
        friction=spec["friction"],
        size_cm=[round(e * 100, 1) for e in mesh.bounding_box.extents],
        tags=spec["tags"],
        source="objaverse",
        description=f"{spec['description']} [uid={uid}]",
    )
    meta.save(asset_dir / "metadata.json")

    provenance = {
        "objaverse_uid": uid,
        "name": ann.get("name", ""),
        "license": ann.get("license", ""),
        "author": ann.get("user", {}).get("displayName", ""),
        "viewer_url": ann.get("viewerUrl", ""),
        "face_count_original": ann.get("faceCount", 0),
        "face_count_cleaned": len(mesh.faces),
    }
    (asset_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"  Saved to {asset_dir}")
    return asset_dir / "model.urdf"


def main():
    parser = argparse.ArgumentParser(description="Import Objaverse tabletop assets")
    parser.add_argument(
        "--category", type=str, default=None,
        help="Import only this category (e.g. 'mug'). Default: all 10.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory. Default: assets/objects/",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of candidates to evaluate per category.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only show candidates, don't download or process.",
    )
    parser.add_argument(
        "--keep-primitives", action="store_true",
        help="Don't remove old primitive URDF assets.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else REPO_ROOT / "assets" / "objects"
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = (
        {args.category: ASSET_SPECS[args.category]}
        if args.category
        else ASSET_SPECS
    )

    print("Loading Objaverse annotations (first run downloads ~200MB, cached after)...")
    import objaverse

    lvis = objaverse.load_lvis_annotations()
    print(f"  LVIS categories: {len(lvis)}")

    all_anns = objaverse.load_annotations()
    print(f"  Total annotations: {len(all_anns)}")

    results: dict[str, list] = {}
    for cat, spec in categories.items():
        print(f"\n{'='*60}")
        print(f"Category: {cat}")
        print(f"  LVIS search: {spec['lvis']}")

        candidates = find_candidates(cat, spec, lvis, all_anns, top_k=args.top_k)
        results[cat] = candidates

        if not candidates:
            print("  WARNING: No candidates found!")
            continue

        for i, (uid, ann, score) in enumerate(candidates):
            print(
                f"  #{i+1} score={score:.1f}  faces={ann.get('faceCount',0):,}  "
                f"likes={ann.get('likeCount',0)}  "
                f"license={ann.get('license','')}  "
                f"name={ann.get('name','')[:50]}"
            )

    if args.dry_run:
        print("\n[DRY RUN] Exiting without download.")
        return

    print(f"\n{'='*60}")
    print("Downloading and processing best candidates...")

    for cat, candidates in results.items():
        if not candidates:
            print(f"\n[SKIP] {cat}: no candidates")
            continue

        uid, ann, score = candidates[0]
        print(f"\n[{cat}] Downloading {uid} ({ann.get('name','')[:40]})...")

        objects = objaverse.load_objects(
            uids=[uid],
            download_processes=min(4, multiprocessing.cpu_count()),
        )
        glb_path = objects.get(uid)
        if not glb_path:
            print(f"  ERROR: download failed for {uid}")
            continue

        print(f"  Downloaded: {glb_path}")

        try:
            process_asset(uid, glb_path, cat, ASSET_SPECS[cat], output_dir, ann)
        except Exception as e:
            print(f"  ERROR processing {cat}: {e}")
            # Try next candidate
            for uid2, ann2, score2 in candidates[1:]:
                print(f"  Trying fallback: {uid2}...")
                try:
                    objects2 = objaverse.load_objects(uids=[uid2])
                    glb2 = objects2.get(uid2)
                    if glb2:
                        process_asset(uid2, glb2, cat, ASSET_SPECS[cat], output_dir, ann2)
                        break
                except Exception as e2:
                    print(f"  Fallback also failed: {e2}")
                    continue

    if not args.keep_primitives:
        print("\nCleaning up old primitive-only assets...")
        primitives_to_remove = [
            "mug_red", "bowl_white", "plate_round",
            "block_red", "block_blue", "block_green",
            "bottle_tall", "can_soda", "fork_silver", "spoon_silver",
        ]
        for name in primitives_to_remove:
            old_dir = output_dir / name
            if old_dir.exists():
                shutil.rmtree(old_dir)
                print(f"  Removed: {old_dir.name}")

    print("\nRebuilding catalog.json...")
    from robotsmith.assets.library import AssetLibrary
    lib = AssetLibrary(REPO_ROOT / "assets")
    lib.save_catalog()
    print(f"  Catalog: {len(lib)} assets")
    print("\nDone!")


if __name__ == "__main__":
    main()
