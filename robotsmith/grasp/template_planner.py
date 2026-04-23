"""TemplateGraspPlanner — per-category human-defined grasp templates.

Each category (block, bowl, mug, ...) has one GraspTemplate that specifies
approach direction, EE orientation, finger width, and key Z-heights.
The planner converts a template + runtime object pose into a concrete GraspPlan.
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

    # Absolute Z mode (default — block/cube)
    grasp_z: float = 0.135              # absolute Z of grasp point (above z_offset)
    hover_z: float = 0.25               # absolute Z of pre-grasp hover point
    retreat_z: float = 0.30             # absolute Z of post-grasp retreat

    place_z: float = 0.15              # absolute Z of place point (for pick_and_place)

    # Relative Z mode (bowl, etc.) — Z values computed from object height at runtime.
    # grasp_z = z_offset + object_height + ee_above_object (EE height above object top)
    grasp_z_mode: str = "absolute"      # "absolute" | "relative"
    ee_above_object: float = 0.10       # relative: EE Z = table + obj_height + this offset
    hover_clearance: float = 0.12       # relative: hover_z = grasp_z + clearance
    retreat_clearance: float = 0.17     # relative: retreat_z = grasp_z + clearance

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

_register(GraspTemplate(
    category="bowl",
    grasp_type="top_down",
    approach_axis=_DOWN.copy(),
    ee_quat=_TOP_DOWN_QUAT.copy(),
    finger_open=0.04,
    finger_closed=0.01,
    grasp_z_mode="relative",
    ee_above_object=0.095,
    hover_clearance=0.12,
    retreat_clearance=0.17,
    requires_scale=True,
    scale_range=(0.45, 0.55),
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
        object_height: float | None = None,
    ) -> list[GraspPlan]:
        cat = self._resolve_category(asset, category)
        template = self._templates.get(cat, self._templates.get("block"))
        if template is None:
            template = next(iter(self._templates.values()))

        cx, cy = float(object_pos[0]), float(object_pos[1])
        zo = self._z_offset

        if template.grasp_z_mode == "relative":
            h = object_height
            if h is None and asset is not None and hasattr(asset, "metadata"):
                h = getattr(asset.metadata, "size_cm", [0, 0, 0])[2] / 100.0
            if h is None:
                raise ValueError(
                    f"Template '{cat}' uses relative Z mode but no object_height "
                    f"provided (pass object_height= or asset with metadata.size_cm)"
                )
            gz = zo + h + template.ee_above_object
            hz = gz + template.hover_clearance
            rz = gz + template.retreat_clearance
        else:
            gz = template.grasp_z + zo
            hz = template.hover_z + zo
            rz = template.retreat_z + zo

        grasp_pos = np.array([cx, cy, gz])
        pre_grasp_pos = np.array([cx, cy, hz])
        retreat_pos = np.array([cx, cy, rz])

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
