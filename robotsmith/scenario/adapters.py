"""Adapters from scenario authoring specs to the current RoboSmith runtime.

These adapters lower scenario authoring objects to the runtime contract:
SceneConfig and TaskSpec(success: SuccessNode). The scenario success tree
(TaskSuccessCfg / AllOf / AnyOf / Not) lowers to a runtime SuccessNode tree.
"""

from __future__ import annotations

from robotsmith.scenario.intents import SkillIntentSequenceSpec
from robotsmith.scenario.relations import SolveError, split_ref
from robotsmith.scenario.specs import (
    AllOf,
    AnyOf,
    FrameRef,
    Not,
    SceneSpec,
    SuccessSpec,
    TargetPositionSpec,
    TaskSpec,
    TaskSuccessCfg,
)
from robotsmith.scenario.validators import (
    success_refs,
    validate_scene,
    validate_skill_intent_sequence,
    validate_task,
)
from robotsmith.skills import Skill
from robotsmith.scenes.config import ObjectPlacement, SceneConfig
from robotsmith.tasks.task_spec import SuccessNode, TaskSpec as LegacyTaskSpec


def to_legacy_scene_config(scene: SceneSpec) -> SceneConfig:
    """Lower a scenario SceneSpec to the current simulator-agnostic SceneConfig.

    Example:
        scenario input:
            cap_scene = SceneSpec(
                name="cap_pick_die",
                layout=LayoutSpec(
                    regions={
                        "left_reachable": RegionSpec(
                            name="left_reachable",
                            xy_bounds=((0.42, -0.14), (0.50, -0.06)),
                        ),
                    },
                ),
                objects=(
                    ObjectSpec(
                        name="die",
                        asset="die_01",
                        region="left_reachable",
                    ),
                ),
            )

            legacy_scene = to_legacy_scene_config(cap_scene)

        Legacy output:
            SceneConfig(
                name="cap_pick_die",
                objects=[
                    ObjectPlacement(
                        asset_query="die_01",
                        position_range=[[0.42, -0.14, 0.0], [0.50, -0.06, 0.0]],
                        name_override="die",
                    ),
                ],
            )

    The scenario scene is the authoring object; SceneConfig/ObjectPlacement is the
    legacy runtime contract consumed by the existing resolver and Genesis path.
    Task semantics are lowered separately with to_legacy_task_spec().
    """

    validate_scene(scene)
    camera_overrides = {}
    if scene.camera_position is not None:
        camera_overrides["camera_position"] = list(scene.camera_position)
    if scene.camera_target is not None:
        camera_overrides["camera_target"] = list(scene.camera_target)
    if scene.table_size is not None:
        camera_overrides["table_size"] = list(scene.table_size)
    if scene.table_height is not None:
        camera_overrides["table_height"] = scene.table_height
    return SceneConfig(
        name=scene.name,
        description="Generated from scenario SceneSpec",
        objects=[
            ObjectPlacement(
                asset_query=obj.asset,
                count=1,
                position_range=_region_position_range(scene, obj.region)
                if obj.region is not None
                else None,
                fixed_position=list(obj.fixed_position)
                if obj.fixed_position is not None
                else None,
                name_override=obj.name,
                joint_init=dict(obj.joint_init) if obj.joint_init else None,
                yaw_deg=obj.yaw_deg,
            )
            for obj in scene.objects
        ],
        **camera_overrides,
    )


def to_legacy_task_spec(task: TaskSpec) -> LegacyTaskSpec:
    """Lower a scenario TaskSpec to the current serializable runtime TaskSpec.

    Example:
        scenario input:
            TaskSpec(
                name="cap_pick_die",
                scene=cap_scene,
                instruction="Pick up the die",
                success=TaskSuccessCfg(
                    predicate="object_above",
                    params={
                        "object": "die",
                        "reference": "table",
                        "z_margin": 0.05,
                    },
                ),
            )

        Legacy output:
            LegacyTaskSpec(
                name="cap_pick_die",
                scene="cap_pick_die",
                contact_objects=["die", "table"],
                success=SuccessNode(op="leaf", predicate="object_above", params={...}),
                skills=[],
            )

    skills=[] is intentional: scenario task authoring defines task semantics, not
    expert action sequences.
    """

    validate_task(task)
    return LegacyTaskSpec(
        name=task.name,
        instruction=task.instruction,
        scene=task.scene.name,
        contact_objects=derive_contact_objects(task),
        success=_lower_success(task.success),
        skills=[],
    )


def _lower_success(spec: SuccessSpec) -> SuccessNode:
    """Lower a scenario success spec tree to a runtime SuccessNode tree."""

    if isinstance(spec, TaskSuccessCfg):
        return SuccessNode(op="leaf", predicate=spec.predicate, params=dict(spec.params))
    if isinstance(spec, AllOf):
        return SuccessNode(op="all", terms=[_lower_success(t) for t in spec.terms])
    if isinstance(spec, AnyOf):
        return SuccessNode(op="any", terms=[_lower_success(t) for t in spec.terms])
    if isinstance(spec, Not):
        return SuccessNode(op="not", terms=[_lower_success(spec.term)])
    raise TypeError(f"unsupported success spec: {type(spec)!r}")


def to_legacy_skills(
    intents: SkillIntentSequenceSpec,
    task: TaskSpec,
) -> list[Skill]:
    """Lower scenario task-level skill intents to current runtime Skills.

    This is the Feature 2 boundary: scenario intent selects an existing scripted
    expert primitive, while planner, IK, MotionExecutor, and recorder stay in
    the runtime.
    """

    validate_skill_intent_sequence(intents, task)
    return [
        Skill(
            name=intent.name,
            target=intent.target,
            category=intent.category,
            params=dict(intent.params),
        )
        for intent in intents.intents
    ]


def _placement_relations(spec: SuccessSpec) -> tuple[list[dict], bool]:
    """Positive on / inside relations of a success tree, and whether the tree is a pure conjunction."""
    if isinstance(spec, TaskSuccessCfg):
        return ([dict(spec.params)] if spec.predicate == "object_in_box" else []), True
    if isinstance(spec, AllOf):
        found, conj = [], True
        for term in spec.terms:
            rels, term_conj = _placement_relations(term)
            found += rels
            conj &= term_conj
        return found, conj
    return [], False


def _unchecked(reason: str) -> dict:
    return {"status": "unchecked", "reason": reason}


def to_step_bindings(intents: SkillIntentSequenceSpec, success: SuccessSpec) -> list[dict]:
    """Bind each teacher step to the completion condition it can be judged by, one entry per step.

    Only ``place`` binds for now: the object placed is the one the preceding ``pick`` took, and its
    relation is the success tree's single positive on / inside relation for that object. A relation whose
    parent or target disagrees with the place raises ``SolveError``; trees with ``any_of`` / ``not``, an
    object placed twice, or several candidate relations stay ``unchecked`` rather than guessed.
    """
    relations, conjunction = _placement_relations(success)
    bindings, held, place_count = [], None, {}
    for intent in intents.intents:
        if intent.name == "pick":
            held = intent.target
        elif intent.name == "place" and held is not None:
            place_count[held] = place_count.get(held, 0) + 1
            held = None
    held = None
    for i, intent in enumerate(intents.intents):
        if intent.name == "pick":
            held = intent.target
            bindings.append(_unchecked("not migrated"))
            continue
        if intent.name != "place":
            bindings.append(_unchecked("not migrated"))
            continue
        obj, held = held, None
        parent, target = split_ref(intent.target)
        candidates = [r for r in relations if r["object"] == obj]
        if obj is None:
            bindings.append(_unchecked("no preceding pick"))
        elif not conjunction:
            bindings.append(_unchecked("success tree is not a conjunction"))
        elif place_count.get(obj, 0) > 1:
            bindings.append(_unchecked("object placed more than once"))
        elif len(candidates) != 1:
            bindings.append(_unchecked(f"{len(candidates)} success relations for {obj}"))
        else:
            rel = candidates[0]
            if (rel["parent"], rel.get("target")) != (parent, target):
                raise SolveError(f"teacher step {i} places {obj} at {intent.target}, but success wants "
                                 f"{rel['mode']} {rel['parent']}.{rel.get('target')}")
            bindings.append({"status": "bound", "predicate": "object_in_box", "params": rel})
    return bindings


def derive_contact_objects(task: TaskSpec) -> list[str]:
    """Derive the legacy tracking list from success predicate references.

    contact_objects is a runtime tracking contract. Authors should reference
    logical object/target names in predicates instead of writing this list.
    """

    refs = success_refs(task.success)
    ordered = []
    for ref in refs + ["table"]:
        if ref not in ordered:
            ordered.append(ref)
    return ordered


def resolved_target_positions(scene: SceneSpec) -> dict[str, tuple[float, float, float]]:
    """Resolve static target positions to xyz points for predicate/success eval.

    Region-based targets resolve to the region center; fixed targets are trusted
    as provided. Articulated-anchored targets are motion anchors resolved live by
    the runtime (see ``target_frames``), not static success markers, so they are
    excluded here. This does not create physical scene objects.
    """

    validate_scene(scene)
    return {
        target.name: _target_position(scene, target)
        for target in scene.target_positions
        if target.anchor is None
    }


def target_frames(scene: SceneSpec) -> dict[str, FrameRef]:
    """Self-describing frame for every target, for live runtime resolution.

    Static targets become ``world`` frames; anchored targets pass their FrameRef
    through. The runtime resolves each against the live world per segment so a
    point attached to a moving part (e.g. a drawer tray) tracks it.
    """

    validate_scene(scene)
    frames: dict[str, FrameRef] = {}
    for target in scene.target_positions:
        if target.anchor is not None:
            frames[target.name] = target.anchor
        else:
            frames[target.name] = FrameRef(
                kind="world", xyz=_target_position(scene, target)
            )
    return frames


def _region_position_range(
    scene: SceneSpec,
    region_name: str | None,
) -> list[list[float]]:
    if region_name is None:
        raise ValueError("region_name is required")
    region = scene.layout.regions[region_name]
    z = 0.0 if region.z is None else region.z
    lower, upper = region.xy_bounds
    return [[lower[0], lower[1], z], [upper[0], upper[1], z]]


def _target_position(
    scene: SceneSpec,
    target: TargetPositionSpec,
) -> tuple[float, float, float]:
    if target.fixed_position is not None:
        return target.fixed_position
    if target.region is None:
        raise ValueError(f"target position '{target.name}' has no placement")
    region = scene.layout.regions[target.region]
    lower, upper = region.xy_bounds
    z = 0.0 if region.z is None else region.z
    return ((lower[0] + upper[0]) / 2.0, (lower[1] + upper[1]) / 2.0, z)
