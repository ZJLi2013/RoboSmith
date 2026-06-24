"""Validation for CAP authoring specs."""

from __future__ import annotations

from robotsmith.assets import AssetLibrary
from robotsmith.cap.intents import SkillIntentSequenceSpec
from robotsmith.cap.specs import AllOf, AnyOf, Not, SceneSpec, SuccessSpec, TaskSpec, TaskSuccessCfg
from robotsmith.tasks.predicates import PREDICATE_REGISTRY

SUPPORTED_SKILL_INTENTS = {"pick", "place", "open", "close"}


class ValidationError(ValueError):
    """Raised when a CAP authoring spec is invalid."""


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
        if region.min_distance is not None:
            _require(region.min_distance >= 0, f"region '{name}' min_distance must be >= 0")
        if region.keep_out_radius is not None:
            _require(
                region.keep_out_radius >= 0,
                f"region '{name}' keep_out_radius must be >= 0",
            )

    for obj in scene.objects:
        _require(obj.name.strip(), "ObjectSpec.name must be non-empty")
        _require(obj.asset.strip(), f"ObjectSpec.asset must be non-empty for '{obj.name}'")
        _require(obj.pose == "upright", f"unsupported pose '{obj.pose}' for '{obj.name}'")
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


def validate_task(task: TaskSpec, *, asset_library: AssetLibrary | None = None) -> None:
    validate_scene(task.scene, asset_library=asset_library)
    _require(task.name.strip(), "TaskSpec.name must be non-empty")
    _require(task.instruction.strip(), "TaskSpec.instruction must be non-empty")
    _validate_success(task.success)

    refs = success_refs(task.success)
    available_refs = {o.name for o in task.scene.objects} | {
        t.name for t in task.scene.target_positions
    }
    allowed_refs = available_refs | {"table"}
    missing = [ref for ref in refs if ref not in allowed_refs]
    _require(not missing, f"success predicate references unknown names: {missing}")


def validate_skill_intent_sequence(
    intents: SkillIntentSequenceSpec,
    task: TaskSpec,
) -> None:
    """Validate task-level skill intents against a CAP task's scene names."""

    validate_task(task)
    _require(intents.intents, "SkillIntentSequenceSpec.intents must be non-empty")

    object_names = {obj.name for obj in task.scene.objects}
    target_names = {target.name for target in task.scene.target_positions}
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


def _validate_success(spec: SuccessSpec) -> None:
    """Recursively validate a success spec tree."""

    if isinstance(spec, TaskSuccessCfg):
        _require(
            spec.predicate in PREDICATE_REGISTRY,
            f"unknown success predicate '{spec.predicate}'",
        )
        return
    if isinstance(spec, (AllOf, AnyOf)):
        _require(spec.terms, f"{type(spec).__name__} requires at least one term")
        for term in spec.terms:
            _validate_success(term)
        return
    if isinstance(spec, Not):
        _validate_success(spec.term)
        return
    raise ValidationError(f"unsupported success spec: {type(spec)!r}")


def success_refs(spec: SuccessSpec) -> list[str]:
    """Union of object/target references across all leaves of a success spec.

    Each leaf self-describes its refs (TaskSuccessCfg.refs), declared by the
    authoring helper. Composite nodes recurse and union.
    """

    if isinstance(spec, TaskSuccessCfg):
        return list(spec.refs)
    if isinstance(spec, (AllOf, AnyOf)):
        return [ref for term in spec.terms for ref in success_refs(term)]
    if isinstance(spec, Not):
        return success_refs(spec.term)
    return []


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
