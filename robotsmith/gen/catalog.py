"""Auto-catalog generated assets: extract tags from prompt, write metadata, register."""

from __future__ import annotations

import re
from pathlib import Path

from robotsmith.assets.schema import Asset, AssetMetadata

KEYWORD_TAGS: dict[str, list[str]] = {
    "mug": ["mug", "cup", "container"],
    "cup": ["cup", "mug", "container"],
    "bowl": ["bowl", "container"],
    "plate": ["plate", "dish"],
    "fork": ["fork", "utensil"],
    "spoon": ["spoon", "utensil"],
    "bottle": ["bottle", "container"],
    "can": ["can", "container"],
    "block": ["block", "cube", "stackable"],
    "cube": ["cube", "block", "stackable"],
    "teapot": ["teapot", "container", "pour"],
    "pot": ["pot", "container"],
    "knife": ["knife", "utensil"],
    "pan": ["pan", "container"],
    "drawer": ["drawer", "furniture"],
    "table": ["table", "furniture"],
    "chair": ["chair", "furniture"],
    "box": ["box", "container"],
    "sphere": ["sphere", "ball"],
    "ball": ["ball", "sphere"],
}

COLOR_TAGS = {"red", "blue", "green", "yellow", "white", "black", "silver", "gold", "brown", "orange", "pink", "purple"}


def tags_from_prompt(prompt: str) -> list[str]:
    """Extract relevant tags from a generation prompt."""
    tags: set[str] = set()
    words = re.split(r"[\s,;\-_]+", prompt.lower().strip())

    for w in words:
        if w in KEYWORD_TAGS:
            tags.update(KEYWORD_TAGS[w])
        if w in COLOR_TAGS:
            tags.add(w)

    tags.add("generated")
    tags.add("grasp")
    return sorted(tags)


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
    tags = tags_from_prompt(prompt)
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
        tags=tags,
        source="generated",
        description=f"Generated from prompt: {prompt}",
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
