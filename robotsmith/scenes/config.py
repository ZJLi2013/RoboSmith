"""Scene configuration dataclass and loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ObjectPlacement:
    """Placement specification for one object in a scene."""

    asset_query: str
    count: int = 1
    position_range: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    rotation_range: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    fixed_position: Optional[list[float]] = None


@dataclass
class SceneConfig:
    """A scene layout configuration (simulator-agnostic)."""

    name: str
    description: str = ""
    objects: list[ObjectPlacement] = field(default_factory=list)

    table_size: list[float] = field(default_factory=lambda: [0.8, 0.6, 0.05])
    table_height: float = 0.75
    robot: str = "franka_panda"

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
            "objects": [
                {
                    "query": o.asset_query,
                    "count": o.count,
                    "position_range": o.position_range,
                    "fixed_position": o.fixed_position,
                }
                for o in self.objects
            ],
            "physics": {"gravity": self.gravity, "time_step": self.time_step},
            "camera": {"position": self.camera_position, "target": self.camera_target},
        }
