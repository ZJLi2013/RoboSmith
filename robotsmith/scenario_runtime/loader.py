"""Load YAML scenario files into the runtime scene, target frames, task and plan skills for one seed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from robotsmith.scenario.load import load_scenario
from robotsmith.scenario.specs import FrameRef
from robotsmith.scenes.config import SceneConfig
from robotsmith.skills import Skill
from robotsmith.tasks.task_spec import TaskSpec


@dataclass(frozen=True)
class ScenarioCandidate:
    """Scene, plan-file target frames, task and optional plan-file skills solved for one seed."""

    source_path: Path
    scene: SceneConfig
    target_frames: dict[str, FrameRef]
    task: TaskSpec
    skills: list[Skill] | None


def load_candidate(path: str | Path, *, seed: int) -> ScenarioCandidate:
    scene, frames, task, skills = load_scenario(path, seed=seed)
    return ScenarioCandidate(source_path=Path(path), scene=scene, target_frames=frames, task=task, skills=skills)
