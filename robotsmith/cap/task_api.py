"""Small authoring helper for CAP task specs."""

from __future__ import annotations

from robotsmith.cap.specs import SceneSpec, SuccessSpec, TaskSpec


def task(
    name: str,
    *,
    scene: SceneSpec,
    instruction: str,
    success: SuccessSpec,
) -> TaskSpec:
    """Create a CAP task from scene, instruction, and success config.

    Example:
        task(
            "cap_pick_die",
            scene=pick_die_scene,
            instruction="Pick up the die",
            success=object_lifted("die"),
        )

    This only defines what the task is; expert action sequences such as
    pick/place steps belong to a separate policy/action layer.
    """

    return TaskSpec(
        name=name,
        scene=scene,
        instruction=instruction,
        success=success,
    )
