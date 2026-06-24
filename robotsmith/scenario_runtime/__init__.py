"""Runtime consumption helpers for authoring-ready RoboSmith scenarios."""

from robotsmith.scenario_runtime.loader import ScenarioCandidate, load_candidate
from robotsmith.scenario_runtime.materialize import (
    ScenarioRuntimePackage,
    materialize_candidate,
)

__all__ = [
    "ScenarioCandidate",
    "ScenarioRuntimePackage",
    "load_candidate",
    "materialize_candidate",
]
