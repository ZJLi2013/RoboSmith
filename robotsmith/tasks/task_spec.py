"""TaskSpec: declarative task definition dataclass.

Serializable — no callables, only registry keys.
``motion_type`` replaces the old ``ik_strategy`` field (Part 3 refactor).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field


_IK_TO_MOTION = {
    "pick": "pick",
    "pick_and_place": "pick_and_place",
    "stack": "pick_and_place",
}


@dataclass
class TaskSpec:
    name: str
    instruction: str
    scene: str
    contact_objects: list[str] = field(default_factory=list)
    success_fn: str = ""
    success_params: dict = field(default_factory=dict)
    motion_type: str = "pick"
    episode_length: int = 200
    dart_sigma: float = 0.0
    grasp_planner: str = "template"
    is_stack: bool = False
    n_stack: int = 3

    @property
    def ik_strategy(self) -> str:
        """Deprecated — use motion_type instead."""
        warnings.warn(
            "TaskSpec.ik_strategy is deprecated; use motion_type",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.is_stack:
            return "stack"
        return self.motion_type

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "instruction": self.instruction,
            "scene": self.scene,
            "contact_objects": self.contact_objects,
            "success_fn": self.success_fn,
            "success_params": dict(self.success_params),
            "motion_type": self.motion_type,
            "episode_length": self.episode_length,
            "dart_sigma": self.dart_sigma,
            "grasp_planner": self.grasp_planner,
            "is_stack": self.is_stack,
            "n_stack": self.n_stack,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TaskSpec:
        d = dict(d)
        if "ik_strategy" in d and "motion_type" not in d:
            old = d.pop("ik_strategy")
            d["motion_type"] = _IK_TO_MOTION.get(old, "pick")
            if old == "stack":
                d["is_stack"] = True
        elif "ik_strategy" in d:
            d.pop("ik_strategy")
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
