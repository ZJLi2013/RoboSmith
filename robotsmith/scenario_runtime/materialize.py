"""Materialize scenarios into current runtime objects."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from pathlib import Path

from robotsmith.scenario.adapters import (
    to_legacy_scene_config,
    to_legacy_skills,
    to_step_bindings,
)
from robotsmith.assets.library import AssetLibrary
from robotsmith.scenario.intents import SkillIntentSequenceSpec
from robotsmith.scenario.load import DEFAULT_ASSETS, all_place_target_frames
from robotsmith.scenario.specs import SceneSpec
from robotsmith.scenario_runtime.loader import ScenarioCandidate, load_candidate
from robotsmith.scenes.config import SceneConfig
from robotsmith.tasks.task_spec import TaskSpec


@dataclass(frozen=True)
class ScenarioRuntimePackage:
    """Both scenario specs and legacy runtime objects for one scenario run."""

    source_path: Path
    scenario_scene: SceneSpec
    scenario_intents: SkillIntentSequenceSpec | None
    legacy_scene: SceneConfig
    legacy_task: TaskSpec


def materialize_candidate(
    candidate_or_path: ScenarioCandidate | str | Path,
    *,
    seed: int | None = None,
    agent_plan: bool = False,
) -> ScenarioRuntimePackage:
    """Reconstruct runtime objects; a path is loaded with ``seed``. With ``agent_plan`` the plan file is
    ignored and every place target gets a frame, since an agent writes the steps at run time."""

    candidate = (
        load_candidate(candidate_or_path, seed=seed)
        if isinstance(candidate_or_path, (str, Path))
        else candidate_or_path
    )
    library = AssetLibrary(DEFAULT_ASSETS)
    assets = {o.name: library.get(o.asset) for o in candidate.scene.objects}
    scene, intents = candidate.scene, candidate.intents
    if agent_plan:
        scene = replace(scene, target_positions=tuple(all_place_target_frames(assets).values()))
        intents = None
    legacy_scene = to_legacy_scene_config(scene)
    # The candidate's task is shared by every materialization of it; this run's skills go on a copy.
    legacy_task = copy.deepcopy(candidate.task)
    if intents is not None:
        legacy_task.skills = to_legacy_skills(intents, scene)
        legacy_task.step_bindings = to_step_bindings(intents, legacy_task.success, assets)
    return ScenarioRuntimePackage(
        source_path=candidate.source_path,
        scenario_scene=scene,
        scenario_intents=intents,
        legacy_scene=legacy_scene,
        legacy_task=legacy_task,
    )
