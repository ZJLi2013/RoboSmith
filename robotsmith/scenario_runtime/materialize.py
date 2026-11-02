"""Materialize scenarios into current runtime objects."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from robotsmith.scenario.adapters import (
    to_legacy_scene_config,
    to_legacy_skills,
    to_legacy_task_spec,
    to_step_bindings,
)
from robotsmith.assets.library import AssetLibrary
from robotsmith.scenario.intents import SkillIntentSequenceSpec
from robotsmith.scenario.load import DEFAULT_ASSETS, all_place_target_frames
from robotsmith.scenario.specs import SceneSpec, TaskSpec
from robotsmith.scenario_runtime.loader import ScenarioCandidate, load_candidate
from robotsmith.scenes.config import SceneConfig
from robotsmith.tasks.task_spec import TaskSpec as LegacyTaskSpec


@dataclass(frozen=True)
class ScenarioRuntimePackage:
    """Both scenario specs and legacy runtime objects for one scenario run."""

    source_path: Path
    scenario_scene: SceneSpec
    scenario_task: TaskSpec
    scenario_intents: SkillIntentSequenceSpec | None
    legacy_scene: SceneConfig
    legacy_task: LegacyTaskSpec


def materialize_candidate(
    candidate_or_path: ScenarioCandidate | str | Path,
    *,
    seed: int | None = None,
    agent_plan: bool = False,
) -> ScenarioRuntimePackage:
    """Reconstruct runtime objects; a path is loaded with ``seed``. With ``agent_plan`` the teacher file is
    ignored and every place target gets a frame, since an agent writes the steps at run time."""

    candidate = (
        load_candidate(candidate_or_path, seed=seed)
        if isinstance(candidate_or_path, (str, Path))
        else candidate_or_path
    )
    library = AssetLibrary(DEFAULT_ASSETS)
    assets = {o.name: library.get(o.asset) for o in candidate.scene.objects}
    scene, task, intents = candidate.scene, candidate.task, candidate.intents
    if agent_plan:
        scene = replace(scene, target_positions=tuple(all_place_target_frames(assets).values()))
        task, intents = replace(task, scene=scene), None
    legacy_scene = to_legacy_scene_config(scene)
    legacy_task = to_legacy_task_spec(task)
    if intents is not None:
        legacy_task.skills = to_legacy_skills(intents, task)
        legacy_task.step_bindings = to_step_bindings(intents, task.success, assets)
    return ScenarioRuntimePackage(
        source_path=candidate.source_path,
        scenario_scene=scene,
        scenario_task=task,
        scenario_intents=intents,
        legacy_scene=legacy_scene,
        legacy_task=legacy_task,
    )
