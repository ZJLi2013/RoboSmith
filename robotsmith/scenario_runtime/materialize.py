"""Materialize scenarios into current runtime objects."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

from robotsmith.scenario.adapters import to_step_bindings
from robotsmith.assets.library import AssetLibrary
from robotsmith.scenario.load import DEFAULT_ASSETS, all_place_target_frames
from robotsmith.scenario.specs import FrameRef
from robotsmith.scenario_runtime.loader import ScenarioCandidate, load_candidate
from robotsmith.scenes.config import SceneConfig
from robotsmith.tasks.task_spec import TaskSpec


@dataclass(frozen=True)
class ScenarioRuntimePackage:
    """The runtime scene, its target frames and the task (with the plan's skills) for one scenario run."""

    source_path: Path
    legacy_scene: SceneConfig
    target_frames: dict[str, FrameRef]
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
    assets = {o.name_override: library.get(o.asset_query) for o in candidate.scene.objects}
    frames, skills = dict(candidate.target_frames), candidate.skills
    if agent_plan:
        frames = all_place_target_frames(assets)
        skills = None
    # The candidate's scene, task and skills are shared by every materialization of it; each run gets its own copies.
    legacy_task = copy.deepcopy(candidate.task)
    if skills is not None:
        legacy_task.skills = copy.deepcopy(skills)
        legacy_task.step_bindings = to_step_bindings(legacy_task.skills, legacy_task.success, assets)
    return ScenarioRuntimePackage(
        source_path=candidate.source_path,
        legacy_scene=copy.deepcopy(candidate.scene),
        target_frames=frames,
        legacy_task=legacy_task,
    )
