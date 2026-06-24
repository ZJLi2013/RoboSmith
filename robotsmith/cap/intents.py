"""Task-level skill intent helpers for CAP authoring.

These intents describe which existing scripted expert primitive to use. They do
not define waypoints, gripper commands, IK, or rollout execution details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SkillIntentSpec:
    """Typed task-level intent that lowers to a legacy runtime Skill.

    ``params`` carries optional primitive knobs (e.g. ``place_z`` for dropping
    into a raised container) and lowers verbatim to ``Skill.params``.
    """

    name: str
    target: str
    category: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillIntentSequenceSpec:
    """Ordered task-level skill intents for scripted expert rollout."""

    intents: tuple[SkillIntentSpec, ...]


def pick(target: str, *, category: str | None = None) -> SkillIntentSpec:
    """Create a pick intent over a physical scene object.

    Example:
        pick("object", category="fruit")
    """

    return SkillIntentSpec(name="pick", target=target, category=category)


def place(
    target: str,
    *,
    category: str | None = None,
    place_z: float | None = None,
) -> SkillIntentSpec:
    """Create a place intent over an object or target position.

    Release height defaults to the held object's selected grasp at runtime.
    Pass ``place_z`` (height above the table) to drop into a raised container
    such as an open drawer tray, where the tabletop pick height is too low.

    Example:
        place("goal", category="fruit")
        place("drawer_open_slot", place_z=0.12)
    """

    params: dict[str, Any] = {}
    if place_z is not None:
        params["place_z"] = place_z
    return SkillIntentSpec(name="place", target=target, category=category, params=params)


def open_(target: str, *, category: str | None = None) -> SkillIntentSpec:
    """Create an open intent over an articulated scene object.

    Drives the object's task joint open by grabbing its handle and dragging
    along the joint axis (e.g. pulling a drawer out).

    Example:
        open_("drawer")
    """

    return SkillIntentSpec(name="open", target=target, category=category)


def close_(target: str, *, category: str | None = None) -> SkillIntentSpec:
    """Create a close intent over an articulated scene object.

    Example:
        close_("drawer")
    """

    return SkillIntentSpec(name="close", target=target, category=category)


def intent_sequence(
    intents: list[SkillIntentSpec] | tuple[SkillIntentSpec, ...],
) -> SkillIntentSequenceSpec:
    """Create an ordered skill intent sequence."""

    return SkillIntentSequenceSpec(intents=tuple(intents))
