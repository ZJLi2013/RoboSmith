"""TaskSpec: declarative task definition dataclass.

Serializable — no callables, only registry keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotsmith.orchestration.skills import Skill


@dataclass
class TaskSpec:
    name: str
    instruction: str
    scene: str
    contact_objects: list[str] = field(default_factory=list)
    success_fn: str = ""
    success_params: dict = field(default_factory=dict)
    skills: list[Skill] = field(default_factory=list)
    episode_length: int = 200
    dart_sigma: float = 0.0
    grasp_planner: str = "template"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "instruction": self.instruction,
            "scene": self.scene,
            "contact_objects": self.contact_objects,
            "success_fn": self.success_fn,
            "success_params": dict(self.success_params),
            "skills": [s.to_dict() for s in self.skills],
            "episode_length": self.episode_length,
            "dart_sigma": self.dart_sigma,
            "grasp_planner": self.grasp_planner,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TaskSpec:
        from robotsmith.orchestration.skills import Skill
        data = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "skills" in data and data["skills"]:
            data["skills"] = [Skill.from_dict(s) for s in data["skills"]]
        return cls(**data)
