"""scenes — runtime scene contract (the lowered-to target of CAP authoring).

This is the **runtime** side of the authoring↔runtime split: ``SceneConfig`` /
``ObjectPlacement`` are what the simulator + skill pipeline consume. The authoring
side (``SceneSpec``) lives in :mod:`robotsmith.cap` and is lowered here via
``cap.adapters``. Don't confuse ``scenes.SceneConfig`` (runtime) with
``cap.SceneSpec`` (authoring).
"""

from robotsmith.scenes.config import SceneConfig, ObjectPlacement  # noqa: F401
from robotsmith.scenes.backend import (  # noqa: F401
    PlacedObject,
    ResolvedScene,
    ProgrammaticSceneBackend,
)
from robotsmith.scenes.pose_utils import (  # noqa: F401
    task_pose_quat,
    task_pose_verified,
    task_upright_quat,
)
