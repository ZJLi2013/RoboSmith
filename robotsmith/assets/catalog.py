"""Auto-catalog generated assets: write metadata and return a registerable Asset."""

from __future__ import annotations

import re
from pathlib import Path

from robotsmith.assets.schema import Asset, AssetMetadata

IDENTITY_4X4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def generated_canonical_frame() -> dict:
    """Canonical-frame contract expected from controlled generation backends."""
    return {
        "unit": "m",
        "up_axis": "+Z",
        "front_axis": "+X",
        "origin": "object_center",
        "T_object_mesh": IDENTITY_4X4,
        "source": "generator_import_contract",
        "verified": False,
    }


def generated_task_poses() -> dict:
    """Generated assets need visual QA before runtime task poses are written."""
    return {}


def name_from_prompt(prompt: str) -> str:
    """Generate a filesystem-safe name from a prompt."""
    clean = re.sub(r"[^a-zA-Z0-9\s]", "", prompt.lower().strip())
    parts = clean.split()[:4]
    return "_".join(parts) if parts else "unnamed"


def catalog_asset(
    output_dir: Path,
    prompt: str,
    mass_kg: float = 0.1,
    friction: float = 0.5,
) -> Asset:
    """Create metadata.json and return an Asset for a generated object."""
    name = output_dir.name

    import trimesh

    visual_glb = output_dir / "visual.glb"
    visual_obj = output_dir / "visual.obj"
    visual_path = visual_glb if visual_glb.exists() else visual_obj

    if visual_path.exists():
        mesh = trimesh.load(str(visual_path), force="mesh")
        extents = mesh.bounding_box.extents
        size_cm = [round(e * 100, 1) for e in extents]
    else:
        size_cm = [5.0, 5.0, 5.0]

    metadata = AssetMetadata(
        mass_kg=mass_kg,
        friction=friction,
        size_cm=size_cm,
        source="generated",
        description=f"Generated from prompt: {prompt}",
        canonical_frame=generated_canonical_frame(),
        task_poses=generated_task_poses(),
    )
    metadata.save(output_dir / "metadata.json")

    urdf_path = output_dir / "model.urdf"
    return Asset(
        name=name,
        root_dir=output_dir,
        urdf_path=urdf_path,
        metadata=metadata,
        visual_mesh=visual_path if visual_path.exists() else None,
        collision_mesh=output_dir / "collision.obj" if (output_dir / "collision.obj").exists() else None,
    )
