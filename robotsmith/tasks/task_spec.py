"""TaskSpec: declarative task definition dataclass.

Serializable — no callables, only registry keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotsmith.skills import Skill


@dataclass
class SuccessNode:
    """Serializable success-condition tree.

    op == "leaf": a single registered predicate (predicate name + params).
    op in {"all", "any"}: composite over terms (AND / OR).
    op == "not": negation of a single term (terms[0]).

    The leaf still references PREDICATE_REGISTRY by name, so the tree stays
    serializable with no callables.
    """

    op: str
    predicate: str = ""
    params: dict = field(default_factory=dict)
    terms: list["SuccessNode"] = field(default_factory=list)

    @classmethod
    def leaf(cls, predicate: str, params: dict | None = None) -> "SuccessNode":
        return cls(op="leaf", predicate=predicate, params=dict(params or {}))

    def to_dict(self) -> dict:
        if self.op == "leaf":
            return {
                "op": "leaf",
                "predicate": self.predicate,
                "params": dict(self.params),
            }
        return {"op": self.op, "terms": [t.to_dict() for t in self.terms]}

    @classmethod
    def from_dict(cls, d: dict) -> "SuccessNode":
        op = d["op"]
        if op == "leaf":
            return cls(
                op="leaf",
                predicate=d.get("predicate", ""),
                params=dict(d.get("params", {})),
            )
        return cls(op=op, terms=[cls.from_dict(t) for t in d.get("terms", [])])


@dataclass
class TaskSpec:
    name: str
    instruction: str
    scene: str
    contact_objects: list[str] = field(default_factory=list)
    success: SuccessNode = field(default_factory=lambda: SuccessNode(op="leaf"))
    skills: list[Skill] = field(default_factory=list)
    episode_length: int = 200
    dart_sigma: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "instruction": self.instruction,
            "scene": self.scene,
            "contact_objects": self.contact_objects,
            "success": self.success.to_dict(),
            "skills": [s.to_dict() for s in self.skills],
            "episode_length": self.episode_length,
            "dart_sigma": self.dart_sigma,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TaskSpec:
        from robotsmith.skills import Skill
        data = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "skills" in data and data["skills"]:
            data["skills"] = [Skill.from_dict(s) for s in data["skills"]]
        if isinstance(data.get("success"), dict):
            data["success"] = SuccessNode.from_dict(data["success"])
        return cls(**data)
