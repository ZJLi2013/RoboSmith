"""scenes — runtime scene contract (the lowered-to target of scenario authoring).

This is the **runtime** side of the authoring↔runtime split: ``SceneConfig`` /
``ObjectPlacement`` are what the simulator + skill pipeline consume. The authoring
side (``SceneSpec``) lives in :mod:`robotsmith.scenario` and is lowered here via
``scenario.adapters``. Don't confuse ``scenes.SceneConfig`` (runtime) with
``scenario.SceneSpec`` (authoring).
"""

from robotsmith.scenes.config import SceneConfig, ObjectPlacement  # noqa: F401
from robotsmith.scenes.backend import (  # noqa: F401
    PlacedObject,
    ResolvedScene,
    ProgrammaticSceneBackend,
)
