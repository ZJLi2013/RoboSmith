"""Load YAML scenario files into the runtime scene, target frames, task and plan intents for one seed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from robotsmith.scenario.intents import SkillIntentSequenceSpec
from robotsmith.scenario.load import load_scenario
from robotsmith.scenario.specs import FrameRef
from robotsmith.scenes.config import SceneConfig
from robotsmith.tasks.task_spec import TaskSpec


@dataclass(frozen=True)
class ScenarioCandidate:
    """Scene, plan-file target frames, task and optional plan-file intents solved for one seed."""

    source_path: Path
    scene: SceneConfig
    target_frames: dict[str, FrameRef]
    task: TaskSpec
    intents: SkillIntentSequenceSpec | None


def load_candidate(path: str | Path, *, seed: int) -> ScenarioCandidate:
    scene, frames, task, intents = load_scenario(path, seed=seed)
    return ScenarioCandidate(source_path=Path(path), scene=scene, target_frames=frames, task=task, intents=intents)
