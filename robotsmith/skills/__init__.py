"""Skill (action) layer: pluggable manipulation primitives behind a registry.

A task is an ordered list of :class:`Skill`s. ``run_skills`` / ``plan_segment``
dispatch each one through the ``SKILL_PRIMITIVES`` registry. Adding a new action
is a new module here whose ``run_*`` function carries an ``@register("name")``
decorator — the generic loop never grows an if/elif. Importing this package
imports the action modules so they self-register.
"""

from robotsmith.skills.base import Skill, SkillContext
from robotsmith.skills.registry import (
    SKILL_PRIMITIVES,
    plan_segment,
    register,
    run_skills,
    skill_phases,
)
from robotsmith.skills.frames import resolve_frame

# Import the action modules for their registration side effects.
from robotsmith.skills import open_close, pick, place  # noqa: F401,E402

__all__ = [
    "Skill",
    "SkillContext",
    "SKILL_PRIMITIVES",
    "register",
    "run_skills",
    "plan_segment",
    "skill_phases",
    "resolve_frame",
]
