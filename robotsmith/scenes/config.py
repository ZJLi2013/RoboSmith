"""Scene configuration dataclass and loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Franka Panda effective tabletop workspace (conservative rectangle).
# Robot base at origin; arm reach ~0.855m.
DEFAULT_WORKSPACE_XY = [[0.35, -0.25], [0.70, 0.25]]


@dataclass
class ObjectPlacement:
    """Placement specification for one object in a scene."""

    asset_query: str
    count: int = 1
    position_range: Optional[list[list[float]]] = None
    """XY(Z) sampling bounds. If None, falls back to scene-level workspace_xy."""
    rotation_range: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    fixed_position: Optional[list[float]] = None
    scale: float = 1.0
    """Uniform scale applied to the URDF at load time (e.g. 0.25 to shrink a
    14 cm asset to ~3.5 cm).  Also applied to collision mesh during placement."""
    name_override: Optional[str] = None
    """Override the asset name in PlacedObject for skill-target matching.
    Useful when you need a specific name like 'cube' or 'bowl' regardless
    of the underlying asset (e.g. 'block_red')."""


@dataclass
class SceneConfig:
    """A scene layout configuration (simulator-agnostic)."""

    name: str
    description: str = ""
    objects: list[ObjectPlacement] = field(default_factory=list)

    table_size: list[float] = field(default_factory=lambda: [1.2, 0.8, 0.05])
    table_height: float = 0.75
    robot: str = "franka_panda"

    workspace_xy: list[list[float]] = field(
        default_factory=lambda: list(DEFAULT_WORKSPACE_XY)
    )
    """Robot-reachable XY area. Objects sample (x,y) within this rectangle."""

    collision_margin: float = 0.02
    """Minimum distance (m) between placed objects."""
    max_placement_retries: int = 100

    gravity: list[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    time_step: float = 1.0 / 240.0

    camera_position: list[float] = field(default_factory=lambda: [1.5, 0.0, 1.2])
    camera_target: list[float] = field(default_factory=lambda: [0.4, 0.0, 0.75])

    def to_dict(self) -> dict:
        """Serialize for JSON / Genesis / PyBullet consumption."""
        return {
            "name": self.name,
            "description": self.description,
            "table": {"size": self.table_size, "height": self.table_height},
            "robot": self.robot,
            "workspace_xy": self.workspace_xy,
            "objects": [
                {
                    "query": o.asset_query,
                    "count": o.count,
                    "position_range": o.position_range,
                    "fixed_position": o.fixed_position,
                    "scale": o.scale,
                    "name_override": o.name_override,
                }
                for o in self.objects
            ],
            "physics": {"gravity": self.gravity, "time_step": self.time_step},
            "camera": {"position": self.camera_position, "target": self.camera_target},
        }
