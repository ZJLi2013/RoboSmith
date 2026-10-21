"""Scenario authoring: YAML scenarios lowered to the scene / task runtime."""

from robotsmith.scenario.adapters import (
    derive_contact_objects,
    resolved_target_positions,
    target_frames,
    to_legacy_scene_config,
    to_legacy_skills,
    to_legacy_task_spec,
)
from robotsmith.scenario.intents import (
    SkillIntentSequenceSpec,
    SkillIntentSpec,
    intent_sequence,
    pick,
    place,
)
from robotsmith.scenario.specs import (
    FrameRef,
    LayoutSpec,
    ObjectSpec,
    RegionSpec,
    SceneSpec,
    TargetPositionSpec,
    TaskSpec,
    TaskSuccessCfg,
)
from robotsmith.scenario.validators import (
    ValidationError,
    validate_scene,
    validate_skill_intent_sequence,
    validate_task,
)

__all__ = [
    "derive_contact_objects",
    "resolved_target_positions",
    "target_frames",
    "to_legacy_scene_config",
    "to_legacy_skills",
    "to_legacy_task_spec",
    "SkillIntentSequenceSpec",
    "SkillIntentSpec",
    "intent_sequence",
    "pick",
    "place",
    "FrameRef",
    "LayoutSpec",
    "ObjectSpec",
    "RegionSpec",
    "SceneSpec",
    "TargetPositionSpec",
    "TaskSpec",
    "TaskSuccessCfg",
    "ValidationError",
    "validate_scene",
    "validate_skill_intent_sequence",
    "validate_task",
]
