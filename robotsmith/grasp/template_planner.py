"""TemplateGraspPlanner — per-category human-defined grasp templates.

Each category (block, bowl, mug, ...) has one GraspTemplate that specifies
approach direction, EE orientation, finger width, etc.  The planner converts
a template + runtime object pose into a concrete GraspPlan.

The block/cube template reproduces the exact behavior of the old
TrajectoryParams defaults so that pick_cube remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robotsmith.grasp.plan import GraspPlan
from robotsmith.grasp.planner import GraspPlanner


@dataclass
class GraspTemplate:
    """Per-category grasp affordance — define once, reuse for all variants."""

    category: str
    grasp_type: str                     # "top_down", "side", "rim", ...
    approach_axis: np.ndarray           # unit vec, EE approach direction (world frame)
    ee_quat: np.ndarray                 # EE quaternion at grasp (wxyz)
    finger_open: float
    finger_closed: float

    grasp_z: float = 0.135              # absolute Z of grasp point (above z_offset)
    hover_z: float = 0.25               # absolute Z of pre-grasp hover point
    retreat_z: float = 0.30             # absolute Z of post-grasp retreat

    place_z: float = 0.15              # absolute Z of place point (for pick_and_place)

    requires_scale: bool = False
    scale_range: tuple[float, float] = (1.0, 1.0)


# ---------------------------------------------------------------------------
# Built-in templates (block reproduces legacy hardcoded behaviour)
# ---------------------------------------------------------------------------

_TOP_DOWN_QUAT = np.array([0, 1, 0, 0], dtype=np.float32)
_DOWN = np.array([0.0, 0.0, -1.0])

GRASP_TEMPLATES: dict[str, GraspTemplate] = {}


def _register(t: GraspTemplate) -> GraspTemplate:
    GRASP_TEMPLATES[t.category] = t
    return t


_register(GraspTemplate(
    category="block",
    grasp_type="top_down",
    approach_axis=_DOWN.copy(),
    ee_quat=_TOP_DOWN_QUAT.copy(),
    finger_open=0.04,
    finger_closed=0.01,
    grasp_z=0.135,
    hover_z=0.25,
    retreat_z=0.30,
    place_z=0.15,
))

_register(GraspTemplate(
    category="cube",
    grasp_type="top_down",
    approach_axis=_DOWN.copy(),
    ee_quat=_TOP_DOWN_QUAT.copy(),
    finger_open=0.04,
    finger_closed=0.01,
    grasp_z=0.135,
    hover_z=0.25,
    retreat_z=0.30,
    place_z=0.15,
))


class TemplateGraspPlanner(GraspPlanner):
    """Look up per-category template and emit a GraspPlan.

    z_offset shifts all absolute Z values (e.g. when table surface != 0).
    """

    def __init__(
        self,
        templates: dict[str, GraspTemplate] | None = None,
        z_offset: float = 0.0,
    ):
        self._templates = templates or GRASP_TEMPLATES
        self._z_offset = z_offset

    def plan(
        self,
        object_pos: np.ndarray,
        object_quat: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        *,
        category: str = "block",
        asset: Any = None,
    ) -> list[GraspPlan]:
        cat = self._resolve_category(asset, category)
        template = self._templates.get(cat, self._templates.get("block"))
        if template is None:
            template = next(iter(self._templates.values()))

        cx, cy = float(object_pos[0]), float(object_pos[1])
        zo = self._z_offset

        grasp_pos = np.array([cx, cy, template.grasp_z + zo])
        pre_grasp_pos = np.array([cx, cy, template.hover_z + zo])
        retreat_pos = np.array([cx, cy, template.retreat_z + zo])

        return [GraspPlan(
            grasp_pos=grasp_pos,
            grasp_quat=template.ee_quat.copy(),
            pre_grasp_pos=pre_grasp_pos,
            pre_grasp_quat=template.ee_quat.copy(),
            retreat_pos=retreat_pos,
            retreat_quat=template.ee_quat.copy(),
            finger_open=template.finger_open,
            finger_closed=template.finger_closed,
            quality=1.0,
            metadata={"source": "template", "category": cat},
        )]

    def plan_place(
        self,
        place_pos: np.ndarray,
        *,
        category: str = "block",
        place_z_override: float | None = None,
    ) -> GraspPlan:
        """Build a place-target GraspPlan for pick_and_place tasks."""
        template = self._templates.get(category, self._templates.get("block"))
        if template is None:
            template = next(iter(self._templates.values()))

        px, py = float(place_pos[0]), float(place_pos[1])
        zo = self._z_offset
        pz = (place_z_override if place_z_override is not None
              else template.place_z)

        pre_place_pos = np.array([px, py, template.retreat_z + zo])
        place_target = np.array([px, py, pz + zo])
        retreat_pos = np.array([px, py, template.retreat_z + zo])

        return GraspPlan(
            grasp_pos=place_target,
            grasp_quat=template.ee_quat.copy(),
            pre_grasp_pos=pre_place_pos,
            pre_grasp_quat=template.ee_quat.copy(),
            retreat_pos=retreat_pos,
            retreat_quat=template.ee_quat.copy(),
            finger_open=template.finger_open,
            finger_closed=template.finger_closed,
            quality=1.0,
            metadata={"source": "template_place", "category": category},
        )

    def _resolve_category(self, asset: Any, fallback: str) -> str:
        if asset is not None and hasattr(asset, "metadata"):
            for tag in getattr(asset.metadata, "tags", []):
                if tag in self._templates:
                    return tag
        return fallback
