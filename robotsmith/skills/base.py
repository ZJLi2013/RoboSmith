"""Core skill types shared by the action primitives and the registry.

``Skill`` is the authored action; ``SkillContext`` is everything a primitive
needs to plan + execute one skill. Primitives read/write the carry-over state
(``held_category`` / ``held_place_z``) on the context so the registry loop stays
generic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from robotsmith.grasp.planner import GraspPlanner
from robotsmith.execution.executor import MotionExecutor
from robotsmith.motion.params import MotionParams
from robotsmith.motion.planner import MotionPlanner


@dataclass
class Skill:
    """Single atomic manipulation action.

    ``target`` is the logical scene object name. ``category`` is an optional
    semantic hint; when omitted, the registry resolves it from the target asset
    metadata. Learned grasping should not require manual category registration
    for every new asset.
    """

    name: str  # "pick" | "place" | "open" | "close" | ...
    target: str  # object name — key into scene_state["positions"]
    category: str | None = None
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = {
            "name": self.name,
            "target": self.target,
            "params": dict(self.params),
        }
        if self.category is not None:
            data["category"] = self.category
        return data

    @classmethod
    def from_dict(cls, d: dict) -> Skill:
        return cls(
            name=d["name"],
            target=d["target"],
            category=d.get("category"),
            params=d.get("params", {}),
        )


@dataclass
class SkillContext:
    """Everything a primitive needs to plan + execute one skill.

    Carry-over state (``held_category`` / ``held_place_z``) is read and written
    by primitives so the registry loop stays generic.
    """

    skill: Skill
    index: int
    skills: list[Skill]
    planner: GraspPlanner
    executor: MotionExecutor
    motion_planner: MotionPlanner
    params: MotionParams
    scene_state: dict
    qpos: np.ndarray
    category: str
    held_category: str | None = None
    held_place_z: float | None = None
    phase: str = "all"

    @property
    def obj_pos(self):
        return self.scene_state["positions"][self.skill.target]

    def trace(self, entry: dict) -> None:
        self.scene_state.setdefault("skill_traces", []).append(entry)


def _category_from_asset(asset, fallback: str) -> str:
    """Derive a grasp category label from the asset itself.

    Assets follow the ``{category}_{NN}`` import naming convention (e.g.
    ``mug_01``, ``die_03``), so the variant suffix is stripped to recover the
    category. Names without that suffix are used as-is. The category is a
    trace label in the learned grasp path.
    """
    if asset is None:
        return fallback
    name = getattr(asset, "name", None)
    if not name:
        return fallback
    m = re.match(r"^(.+?)_\d+$", str(name))
    return m.group(1) if m else str(name)


def _resolve_skill_category(
    skill: Skill,
    assets: dict,
    held_category: str | None = None,
) -> str:
    if skill.category:
        return skill.category
    if skill.name == "place" and held_category:
        return held_category
    return _category_from_asset(assets.get(skill.target), skill.target)
