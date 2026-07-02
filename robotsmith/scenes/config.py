"""Simulator-agnostic declarative scene schema.

SceneConfig describes what should be placed on the table. It intentionally does
not expose simulator runtime knobs; adapters such as genesis_loader own those.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Franka Panda effective tabletop workspace (conservative rectangle).
# Robot base at origin; arm reach ~0.855m.
DEFAULT_WORKSPACE_XY = [[0.35, -0.25], [0.70, 0.25]]
TASK_WORKSPACE_XY = [[0.40, -0.20], [0.70, 0.20]]
"""Default workspace for task/data-collection scenes with extra table margin."""


@dataclass
class ObjectPlacement:
    """Placement specification for one object in a scene."""

    asset_query: str
    count: int = 1
    position_range: Optional[list[list[float]]] = None
    """XY(Z) sampling bounds. If None, falls back to scene-level workspace_xy."""
    fixed_position: Optional[list[float]] = None
    """Explicit world position. Trusted as-is and bypasses XY sampling/retry."""
    name_override: Optional[str] = None
    """Override the asset name in PlacedObject for skill-target matching.
    Useful when you need a specific name like 'cube' or 'bowl' regardless
    of the underlying asset (e.g. 'block_blue')."""
    joint_init: Optional[dict] = None
    """Articulated assets only: per-scenario initial joint state {joint: qpos}
    overriding the asset metadata default on reset. None = use asset default."""
    yaw_deg: float = 0.0
    """Rotation (degrees) about world +Z applied on top of the asset upright
    pose. Positive = CCW seen from above, negative = clockwise. 0 = upright."""


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

    camera_position: list[float] = field(default_factory=lambda: [1.5, 0.0, 1.2])
    camera_target: list[float] = field(default_factory=lambda: [0.4, 0.0, 0.75])
