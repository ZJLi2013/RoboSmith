"""Skill registry.

A task is an ordered list of ``Skill``s. ``EpisodeRun`` executes them
sequentially through the ``SKILL_DEFS`` registry (one definition per action:
its primitive and, once migrated, its completion condition). Adding a new action
means dropping a module with an ``@register("name")`` primitive into this
package — the episode loop stays generic (no if/elif growth).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from robotsmith.execution.gate import VerifiedMotion
from robotsmith.skills.base import Skill, SkillContext

@dataclass(frozen=True)
class SkillDef:
    """An action's primitive and, once migrated, the completion condition judged after its last phase.

    ``post(skill, binding)`` returns the ``SuccessNode`` to judge, or ``None`` when the step's load-time
    binding is unchecked; ``settle_s`` is the recorded hold before judging it.
    """

    primitive: Callable[[SkillContext], VerifiedMotion]
    post: Callable[[Skill, dict], object] | None = None
    settle_s: float = 0.0


SKILL_DEFS: dict[str, SkillDef] = {}


def register(name: str, *, post: Callable[[Skill, dict], object] | None = None,
             settle_s: float = 0.0) -> Callable[[Callable], Callable]:
    """Register a primitive implementation (and optionally its completion condition) under an action ``name``.

    Used as a decorator on the per-action ``run_*`` functions so the registry is
    populated by simply importing the action module (done in this package's
    ``__init__``).
    """

    def deco(fn: Callable[[SkillContext], VerifiedMotion]):
        if name in SKILL_DEFS:
            raise ValueError(f"skill primitive {name!r} already registered")
        SKILL_DEFS[name] = SkillDef(fn, post, settle_s)
        return fn

    return deco


def bound_leaf(skill: Skill, binding: dict):
    """The step's load-time binding as one predicate leaf; ``None`` unless the binding is ``bound``."""
    from robotsmith.tasks.task_spec import SuccessNode

    if binding.get("status") != "bound":
        return None
    return SuccessNode.leaf(binding["predicate"], dict(binding["params"]))


def skill_phases(skill: Skill, target_frame=None) -> list[str]:
    """Execution phases for a skill, gated by anchor dynamism.

    Splitting a skill into phases lets the runtime re-sense + re-anchor at each
    phase boundary (step a phase into the sim → re-read the world → plan the
    next). Only ``place`` onto a *dynamic* (articulated) anchor needs it: the
    drawer recoils during the free ``transport``, so ``descend`` must re-read the
    drawer's settled pose before going vertical or it lands on the stale front
    edge. Static (``world``) anchors and every other skill stay a single ``"all"``
    phase — i.e. the legacy open-loop behavior, no extra plan/step cost.
    """
    if skill.name == "place" and getattr(target_frame, "kind", None) in (
        "articulated",
        "articulated_opening",
    ):
        return ["transport", "descend"]
    return ["all"]

