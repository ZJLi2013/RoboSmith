"""Load authoring-ready scenario candidate files."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path

from robotsmith.cap.intents import SkillIntentSequenceSpec
from robotsmith.cap.specs import SceneSpec, TaskSpec


@dataclass(frozen=True)
class ScenarioCandidate:
    """CAP specs returned by a scenario candidate's stable build() entrypoint."""

    source_path: Path
    scene: SceneSpec
    task: TaskSpec
    intents: SkillIntentSequenceSpec


def load_candidate(path: str | Path) -> ScenarioCandidate:
    """Load a generated scenario candidate and call its build() entrypoint."""

    source_path = Path(path)
    spec = importlib.util.spec_from_file_location(
        f"robotsmith_generated_scenario_{source_path.stem}",
        source_path,
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import scenario candidate: {source_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    build = getattr(module, "build", None)
    if not callable(build):
        raise ValueError(f"scenario candidate must define callable build(): {source_path}")

    result = build()
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError(
            "build() must return (scenario_scene, scenario_task, scenario_intents)"
        )

    scenario_scene, scenario_task, scenario_intents = result
    return ScenarioCandidate(
        source_path=source_path,
        scene=scenario_scene,
        task=scenario_task,
        intents=scenario_intents,
    )
