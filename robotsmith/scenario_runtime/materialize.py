"""Materialize authoring-ready scenarios into current runtime objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from robotsmith.cap.adapters import (
    to_legacy_scene_config,
    to_legacy_skills,
    to_legacy_task_spec,
)
from robotsmith.cap.intents import SkillIntentSequenceSpec
from robotsmith.cap.specs import SceneSpec, TaskSpec
from robotsmith.scenario_runtime.loader import ScenarioCandidate, load_candidate
from robotsmith.scenes.config import SceneConfig
from robotsmith.tasks.task_spec import TaskSpec as LegacyTaskSpec


@dataclass(frozen=True)
class ScenarioRuntimePackage:
    """Both CAP specs and legacy runtime objects for one scenario run."""

    source_path: Path
    scenario_scene: SceneSpec
    scenario_task: TaskSpec
    scenario_intents: SkillIntentSequenceSpec
    legacy_scene: SceneConfig
    legacy_task: LegacyTaskSpec


def materialize_candidate(candidate_or_path: ScenarioCandidate | str | Path) -> ScenarioRuntimePackage:
    """Reconstruct runtime objects from an authoring-ready scenario candidate."""

    candidate = (
        load_candidate(candidate_or_path)
        if isinstance(candidate_or_path, (str, Path))
        else candidate_or_path
    )
    legacy_scene = to_legacy_scene_config(candidate.scene)
    legacy_task = to_legacy_task_spec(candidate.task)
    legacy_task.skills = to_legacy_skills(candidate.intents, candidate.task)
    return ScenarioRuntimePackage(
        source_path=candidate.source_path,
        scenario_scene=candidate.scene,
        scenario_task=candidate.task,
        scenario_intents=candidate.intents,
        legacy_scene=legacy_scene,
        legacy_task=legacy_task,
    )
