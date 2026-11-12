"""Authoring-level success predicate helpers.

Leaf predicates: lifted, joint open / closed, alignment, stacking; YAML\nscenarios lower ``on`` / ``inside`` in ``scenario.load``. These
compose via all_of / any_of into a success tree. Still not a full
TerminationCfg system: multiple done terms, timeouts, subtasks, contact /
gripper-detached / local-frame containment checks are future extensions.
"""

from __future__ import annotations

from robotsmith.tasks.task_spec import SuccessNode


def object_lifted(object: str, *, z_margin: float = 0.05) -> SuccessNode:
    """Define lifted-object success without executing the predicate.

    Example:
        object_lifted("die", z_margin=0.05)

    The runtime predicate is "object_above".
    """

    return SuccessNode.leaf(
        "object_above", {"object": object, "reference": "table", "z_margin": z_margin}
    )


def joint_opened(object: str, joint: str, *, open_position: float) -> SuccessNode:
    """Define joint-opened success for an articulated object.

    Example:
        joint_opened("drawer", "drawer_slide", open_position=0.25)

    ``open_position`` is the absolute joint position (meters for prismatic,
    radians for revolute) past which the joint counts as open.
    """

    return SuccessNode.leaf(
        "joint_opened", {"object": object, "joint": joint, "threshold": open_position}
    )


def joint_closed(
    object: str, joint: str, *, closed_position: float = 0.02
) -> SuccessNode:
    """Define joint-closed success for an articulated object.

    Example:
        joint_closed("drawer", "drawer_slide")
    """

    return SuccessNode.leaf(
        "joint_closed", {"object": object, "joint": joint, "threshold": closed_position}
    )


def objects_aligned(
    objects: list[str] | tuple[str, ...],
    *,
    axis: str = "y",
    xy_threshold: float = 0.06,
) -> SuccessNode:
    """Define alignment success for a group of named objects.

    Example:
        objects_aligned(["die", "mug", "apple"], axis="x", xy_threshold=0.12)
    """

    return SuccessNode.leaf(
        "objects_aligned",
        {"objects": list(objects), "axis": axis, "xy_threshold": xy_threshold},
    )


def stacked(
    objects: list[str] | tuple[str, ...],
    *,
    z_tolerance: float = 0.02,
) -> SuccessNode:
    """Define stacking success for ordered named objects.

    Example:
        stacked(["block_a", "block_b", "block_c"])
    """

    return SuccessNode.leaf("stacked", {"objects": list(objects), "z_tolerance": z_tolerance})


def all_of(*terms: SuccessNode) -> SuccessNode:
    """Success when all terms hold (AND).

    Example:
        all_of(
            object_lifted("apple"),
            object_lifted("banana"),
        )
    """

    return SuccessNode(op="all", terms=list(terms))


def any_of(*terms: SuccessNode) -> SuccessNode:
    """Success when any term holds (OR).

    Example:
        any_of(
            object_lifted("die"),
            object_lifted("cube"),
        )
    """

    return SuccessNode(op="any", terms=list(terms))
