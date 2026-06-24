"""Code-as-Policy authoring API."""

from robotsmith.cap.adapters import (
    derive_contact_objects,
    resolved_target_positions,
    target_frames,
    to_legacy_scene_config,
    to_legacy_skills,
    to_legacy_task_spec,
)
from robotsmith.cap.intents import (
    SkillIntentSequenceSpec,
    SkillIntentSpec,
    intent_sequence,
    pick,
    place,
)
from robotsmith.cap.specs import (
    FrameRef,
    LayoutSpec,
    ObjectSpec,
    RegionSpec,
    SceneSpec,
    TargetPositionSpec,
    TaskSpec,
    TaskSuccessCfg,
)
from robotsmith.cap.validators import (
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
