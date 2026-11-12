"""Validation for scenario tasks and object ids; plan steps are checked by ``load.steps_to_skills``."""

from __future__ import annotations

from robotsmith.scenes.config import SceneConfig
from robotsmith.tasks.predicates import PREDICATE_REGISTRY
from robotsmith.tasks.task_spec import SuccessNode, TaskSpec


class ValidationError(ValueError):
    """Raised when a scenario authoring spec is invalid."""


def check_object_ids(ids: list[str]) -> None:
    _require_unique(ids, "object names")


def check_task(task: TaskSpec, scene: SceneConfig, target_names) -> None:
    """A task's name and instruction are non-empty, and its success tree names registered predicates and only
    objects of ``scene``, ``target_names`` or the table."""

    _require(task.name.strip(), "TaskSpec.name must be non-empty")
    _require(task.instruction.strip(), "TaskSpec.instruction must be non-empty")
    _validate_success(task.success)

    allowed_refs = {o.name_override for o in scene.objects} | set(target_names) | {"table"}
    missing = [ref for ref in _refs(task.success) if ref not in allowed_refs]
    _require(not missing, f"success predicate references unknown names: {missing}")


def _validate_success(node: SuccessNode) -> None:
    if node.op == "leaf":
        _require(node.predicate in PREDICATE_REGISTRY, f"unknown success predicate '{node.predicate}'")
        return
    _require(node.op in ("all", "any", "not"), f"unsupported success op: {node.op!r}")
    _require(len(node.terms) == 1 if node.op == "not" else node.terms,
             f"{node.op} requires {'one term' if node.op == 'not' else 'at least one term'}")
    for term in node.terms:
        _validate_success(term)


def _refs(node: SuccessNode) -> list[str]:
    """Scene names a success tree reads: each leaf's ``object`` / ``objects`` / ``parent`` / ``reference`` params."""

    if node.op != "leaf":
        return [ref for term in node.terms for ref in _refs(term)]
    p = node.params
    return ([p["object"]] if "object" in p else []) + list(p.get("objects", ())) + [
        p[k] for k in ("parent", "reference") if k in p]


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _require_unique(values: list[str], label: str) -> None:
    duplicates = sorted({v for v in values if values.count(v) > 1})
    _require(not duplicates, f"duplicate {label}: {duplicates}")
