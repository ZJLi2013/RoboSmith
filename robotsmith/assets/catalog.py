"""Auto-catalog generated assets: write asset.json and return a registerable Asset."""

from __future__ import annotations

import re
from pathlib import Path

from robotsmith.assets import store
from robotsmith.assets.schema import Asset


def name_from_prompt(prompt: str) -> str:
    """Generate a filesystem-safe name from a prompt."""
    clean = re.sub(r"[^a-zA-Z0-9\s]", "", prompt.lower().strip())
    parts = clean.split()[:4]
    return "_".join(parts) if parts else "unnamed"


def catalog_asset(
    output_dir: Path,
    prompt: str,
    friction: float = 0.5,
) -> Asset:
    """Create asset.json and return an Asset for a generated object.

    No upright annotation is written: generated assets need visual QA first.
    """
    visual_glb = output_dir / "visual.glb"
    visual_path = visual_glb if visual_glb.exists() else output_dir / "visual.obj"

    store.new_asset(output_dir, "rigid", "generated", {
        "description": store.annotation(f"Generated from prompt: {prompt}", "harness"),
        "friction": store.annotation(friction, "harness"),
    })

    return Asset(
        name=output_dir.name,
        root_dir=output_dir,
        urdf_path=output_dir / "model.urdf",
        metadata=store.load_metadata(output_dir),
        visual_mesh=visual_path if visual_path.exists() else None,
        collision_mesh=output_dir / "collision.obj" if (output_dir / "collision.obj").exists() else None,
    )
