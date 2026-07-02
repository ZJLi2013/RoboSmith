"""Authoring-level success predicate helpers.

Leaf predicates: lifted, in-container/at-target, alignment, stacking. These
compose via all_of / any_of / negate into a success tree. Still not a full
TerminationCfg system: multiple done terms, timeouts, subtasks, contact /
gripper-detached / local-frame containment checks are future extensions.
"""

from __future__ import annotations

from robotsmith.cap.specs import AllOf, AnyOf, Not, SuccessSpec, TaskSuccessCfg


def object_lifted(object: str, *, z_margin: float = 0.05) -> TaskSuccessCfg:
    """Define lifted-object success without executing the predicate.

    Example:
        object_lifted("die", z_margin=0.05)

    This lowers to the legacy runtime predicate "object_above".
    """

    return TaskSuccessCfg(
        predicate="object_above",
        params={"object": object, "reference": "table", "z_margin": z_margin},
        refs=(object, "table"),
    )


def object_in_container(
    object: str,
    container: str,
    *,
    xy_threshold: float = 0.05,
    z_margin: float = 0.0,
    z_tol: float | None = None,
) -> TaskSuccessCfg:
    """Define object-in-container or object-at-target success.

    Example:
        object_in_container("die", "goal", xy_threshold=0.02)

    The container argument may reference a physical object or a
    TargetPositionSpec name materialized by the legacy adapter.

    For target markers whose Z distinguishes the goal (e.g. stacked shelves
    sharing XY), pass ``z_tol`` to enforce a two-sided band around the marker's
    resolved world height. ``z_tol=None`` keeps the legacy XY-only check.
    """

    return TaskSuccessCfg(
        predicate="object_in_container",
        params={
            "object": object,
            "container": container,
            "xy_threshold": xy_threshold,
            "z_margin": z_margin,
            "z_tol": z_tol,
        },
        refs=(object, container),
    )


def joint_opened(object: str, joint: str, *, open_position: float) -> TaskSuccessCfg:
    """Define joint-opened success for an articulated object.

    Example:
        joint_opened("drawer", "drawer_slide", open_position=0.25)

    ``open_position`` is the absolute joint position (meters for prismatic,
    radians for revolute) past which the joint counts as open. Lowers to the
    runtime predicate "joint_opened".
    """

    return TaskSuccessCfg(
        predicate="joint_opened",
        params={"object": object, "joint": joint, "threshold": open_position},
        refs=(object,),
    )


def joint_closed(
    object: str, joint: str, *, closed_position: float = 0.02
) -> TaskSuccessCfg:
    """Define joint-closed success for an articulated object.

    Example:
        joint_closed("drawer", "drawer_slide")

    Lowers to the runtime predicate "joint_closed".
    """

    return TaskSuccessCfg(
        predicate="joint_closed",
        params={"object": object, "joint": joint, "threshold": closed_position},
        refs=(object,),
    )


def objects_aligned(
    objects: list[str] | tuple[str, ...],
    *,
    axis: str = "y",
    xy_threshold: float = 0.06,
) -> TaskSuccessCfg:
    """Define alignment success for a group of named objects.

    Example:
        objects_aligned(["die", "mug", "apple"], axis="x", xy_threshold=0.12)
    """

    return TaskSuccessCfg(
        predicate="objects_aligned",
        params={
            "objects": list(objects),
            "axis": axis,
            "xy_threshold": xy_threshold,
        },
        refs=tuple(objects),
    )


def stacked(
    objects: list[str] | tuple[str, ...],
    *,
    z_tolerance: float = 0.02,
) -> TaskSuccessCfg:
    """Define stacking success for ordered named objects.

    Example:
        stacked(["block_a", "block_b", "block_c"])
    """

    return TaskSuccessCfg(
        predicate="stacked",
        params={"objects": list(objects), "z_tolerance": z_tolerance},
        refs=tuple(objects),
    )


def all_of(*terms: SuccessSpec) -> AllOf:
    """Success when all terms hold (AND).

    Example:
        all_of(
            object_in_container("apple", "bowl"),
            object_in_container("banana", "bowl"),
        )
    """

    return AllOf(terms=tuple(terms))


def any_of(*terms: SuccessSpec) -> AnyOf:
    """Success when any term holds (OR).

    Example:
        any_of(
            object_in_container("die", "bowl"),
            object_in_container("die", "plate"),
        )
    """

    return AnyOf(terms=tuple(terms))


def negate(term: SuccessSpec) -> Not:
    """Success when the inner term does not hold (NOT)."""

    return Not(term=term)
