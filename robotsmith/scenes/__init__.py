"""scenes — the runtime scene: ``SceneConfig`` / ``ObjectPlacement`` are what the simulator and skill
pipeline consume; :mod:`robotsmith.scenario.load` builds them from a YAML scenario."""

from robotsmith.scenes.config import SceneConfig, ObjectPlacement  # noqa: F401
from robotsmith.scenes.backend import (  # noqa: F401
    PlacedObject,
    ResolvedScene,
    ProgrammaticSceneBackend,
)
