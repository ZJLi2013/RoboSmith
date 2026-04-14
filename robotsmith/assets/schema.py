"""Asset and metadata schema for sim-ready objects."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class AssetMetadata:
    """Physics and catalog metadata for a sim-ready asset."""

    mass_kg: float = 0.1
    friction: float = 0.5
    restitution: float = 0.1
    density_kg_m3: float = 1000.0
    size_cm: list[float] = field(default_factory=lambda: [5.0, 5.0, 5.0])
    tags: list[str] = field(default_factory=list)
    source: str = "builtin"
    description: str = ""
    stable_poses: list[dict] = field(default_factory=list)
    """Pre-computed stable resting poses on a flat surface.
    Each entry: {"z": float, "quat": [w, x, y, z]}.
    Empty list means not yet computed (fallback: upright with z = half-height)."""

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> AssetMetadata:
        data = json.loads(path.read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Asset:
    """A sim-ready asset with URDF, meshes, and metadata."""

    name: str
    root_dir: Path
    urdf_path: Path
    metadata: AssetMetadata
    visual_mesh: Optional[Path] = None
    collision_mesh: Optional[Path] = None

    @property
    def tags(self) -> list[str]:
        return self.metadata.tags

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "root_dir": str(self.root_dir),
            "urdf_path": str(self.urdf_path),
            "visual_mesh": str(self.visual_mesh) if self.visual_mesh else None,
            "collision_mesh": str(self.collision_mesh) if self.collision_mesh else None,
            "metadata": asdict(self.metadata),
        }

    def __repr__(self) -> str:
        tags = ", ".join(self.tags[:5])
        return f"Asset({self.name!r}, tags=[{tags}], mass={self.metadata.mass_kg}kg)"
