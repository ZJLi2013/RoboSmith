"""Load YAML scenario files into scenario specs for one seed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from robotsmith.scenario.intents import SkillIntentSequenceSpec
from robotsmith.scenario.load import load_scenario
from robotsmith.scenario.specs import SceneSpec, TaskSpec


@dataclass(frozen=True)
class ScenarioCandidate:
    """Scene, task and optional teacher intents solved for one seed."""

    source_path: Path
    scene: SceneSpec
    task: TaskSpec
    intents: SkillIntentSequenceSpec | None


def load_candidate(path: str | Path, *, seed: int) -> ScenarioCandidate:
    scene, task, intents = load_scenario(path, seed=seed)
    return ScenarioCandidate(source_path=Path(path), scene=scene, task=task, intents=intents)
