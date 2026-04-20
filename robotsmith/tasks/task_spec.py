"""TaskSpec: declarative task definition dataclass.

Phase 1 design (design.md §2.2). Serializable — no callables, only registry keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskSpec:
    name: str
    instruction: str
    scene: str
    contact_objects: list[str] = field(default_factory=list)
    success_fn: str = ""
    success_params: dict = field(default_factory=dict)
    ik_strategy: str = "pick"
    episode_length: int = 200
    dart_sigma: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "instruction": self.instruction,
            "scene": self.scene,
            "contact_objects": self.contact_objects,
            "success_fn": self.success_fn,
            "success_params": dict(self.success_params),
            "ik_strategy": self.ik_strategy,
            "episode_length": self.episode_length,
            "dart_sigma": self.dart_sigma,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TaskSpec:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
