"""Validation for scenario authoring specs."""

from __future__ import annotations

from robotsmith.assets import AssetLibrary
from robotsmith.scenario.intents import SkillIntentSequenceSpec
from robotsmith.scenario.specs import SceneSpec
from robotsmith.tasks.predicates import PREDICATE_REGISTRY
from robotsmith.tasks.task_spec import SuccessNode, TaskSpec

SUPPORTED_SKILL_INTENTS = {"pick", "place", "open", "close"}


class ValidationError(ValueError):
    """Raised when a scenario authoring spec is invalid."""


def validate_scene(scene: SceneSpec, *, asset_library: AssetLibrary | None = None) -> None:
    _require(scene.name.strip(), "SceneSpec.name must be non-empty")

    object_names = [o.name for o in scene.objects]
    _require_unique(object_names, "object names")
    _require(scene.objects, "SceneSpec.objects must contain at least one object")

    target_names = [t.name for t in scene.target_positions]
    _require_unique(target_names, "target position names")
    overlap = set(object_names) & set(target_names)
    _require(not overlap, f"object and target names must not overlap: {sorted(overlap)}")

    for name, region in scene.layout.regions.items():
        _require(name == region.name, f"region key '{name}' must match RegionSpec.name")
        _validate_xy_bounds(region.xy_bounds, f"region '{name}'")

    for obj in scene.objects:
        _require(obj.name.strip(), "ObjectSpec.name must be non-empty")
        _require(obj.asset.strip(), f"ObjectSpec.asset must be non-empty for '{obj.name}'")
        _validate_one_placement(
            region=obj.region,
            fixed_position=obj.fixed_position,
            region_names=scene.layout.regions,
            label=f"object '{obj.name}'",
        )
        if asset_library is not None:
            _require(
                asset_library.get(obj.asset) is not None,
                f"asset '{obj.asset}' not found in AssetLibrary",
            )

    object_names = {obj.name for obj in scene.objects}
    for target in scene.target_positions:
        _require(target.name.strip(), "TargetPositionSpec.name must be non-empty")
        if target.anchor is not None:
            _require(
                target.region is None and target.fixed_position is None,
                f"target position '{target.name}' must define exactly one of "
                "region/fixed_position/anchor",
            )
            _require(
                target.anchor.parent in object_names,
                f"target position '{target.name}' anchor references unknown "
                f"object '{target.anchor.parent}'",
            )
            continue
        _validate_one_placement(
            region=target.region,
            fixed_position=target.fixed_position,
            region_names=scene.layout.regions,
            label=f"target position '{target.name}'",
        )


def check_task(task: TaskSpec, scene: SceneSpec) -> None:
    """A task's name and instruction are non-empty, and its success tree names registered predicates and only
    objects or targets of ``scene`` (or the table)."""

    validate_scene(scene)
    _require(task.name.strip(), "TaskSpec.name must be non-empty")
    _require(task.instruction.strip(), "TaskSpec.instruction must be non-empty")
    _validate_success(task.success)

    allowed_refs = {o.name for o in scene.objects} | {t.name for t in scene.target_positions} | {"table"}
    missing = [ref for ref in _refs(task.success) if ref not in allowed_refs]
    _require(not missing, f"success predicate references unknown names: {missing}")


def validate_skill_intent_sequence(
    intents: SkillIntentSequenceSpec,
    scene: SceneSpec,
) -> None:
    """Validate task-level skill intents against the scene's object and target names."""

    validate_scene(scene)
    _require(intents.intents, "SkillIntentSequenceSpec.intents must be non-empty")

    object_names = {obj.name for obj in scene.objects}
    target_names = {target.name for target in scene.target_positions}
    available_place_targets = object_names | target_names

    for intent in intents.intents:
        _require(intent.name in SUPPORTED_SKILL_INTENTS, f"unknown skill intent '{intent.name}'")
        _require(intent.target.strip(), f"{intent.name} intent target must be non-empty")
        if intent.name == "pick":
            _require(
                intent.target in object_names,
                f"pick intent references unknown object '{intent.target}'",
            )
        elif intent.name == "place":
            _require(
                intent.target in available_place_targets,
                f"place intent references unknown object or target '{intent.target}'",
            )
        elif intent.name in ("open", "close"):
            _require(
                intent.target in object_names,
                f"{intent.name} intent references unknown object '{intent.target}'",
            )


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


def _validate_one_placement(
    *,
    region: str | None,
    fixed_position: tuple[float, float, float] | None,
    region_names: dict,
    label: str,
) -> None:
    has_region = region is not None
    has_fixed = fixed_position is not None
    _require(has_region ^ has_fixed, f"{label} must define exactly one of region/fixed_position")
    if has_region:
        _require(region in region_names, f"{label} references unknown region '{region}'")
    if has_fixed:
        _require(len(fixed_position) == 3, f"{label} fixed_position must be xyz")


def _validate_xy_bounds(
    bounds: tuple[tuple[float, float], tuple[float, float]],
    label: str,
) -> None:
    _require(len(bounds) == 2, f"{label} xy_bounds must contain min/max corners")
    lower, upper = bounds
    _require(len(lower) == 2 and len(upper) == 2, f"{label} xy_bounds must be 2D")
    _require(lower[0] < upper[0], f"{label} x bounds must increase")
    _require(lower[1] < upper[1], f"{label} y bounds must increase")


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _require_unique(values: list[str], label: str) -> None:
    duplicates = sorted({v for v in values if values.count(v) > 1})
    _require(not duplicates, f"duplicate {label}: {duplicates}")
