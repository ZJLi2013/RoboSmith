"""Skill primitive registry + the generic execution loop.

A task is an ordered list of ``Skill``s. ``run_skills`` executes them
sequentially through the ``SKILL_PRIMITIVES`` registry (one implementation per
action). Adding a new action means dropping a module with an ``@register("name")``
primitive into this package — ``run_skills`` / ``plan_segment`` stay generic loops
(no if/elif growth).
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from robotsmith.grasp.planner import GraspPlanner
from robotsmith.execution.executor import MotionExecutor
from robotsmith.motion.params import MotionParams
from robotsmith.motion.planner import GenesisBackend, MotionPlanner
from robotsmith.skills.base import Skill, SkillContext, _resolve_skill_category

logger = logging.getLogger(__name__)

SKILL_PRIMITIVES: dict[str, Callable[[SkillContext], list[np.ndarray]]] = {}


def register(name: str) -> Callable[[Callable], Callable]:
    """Register a primitive implementation under an action ``name``.

    Used as a decorator on the per-action ``run_*`` functions so the registry is
    populated by simply importing the action module (done in this package's
    ``__init__``).
    """

    def deco(fn: Callable[[SkillContext], list[np.ndarray]]):
        if name in SKILL_PRIMITIVES:
            raise ValueError(f"skill primitive {name!r} already registered")
        SKILL_PRIMITIVES[name] = fn
        return fn

    return deco


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


def plan_segment(
    skill: Skill,
    index: int,
    skills: list[Skill],
    planner: GraspPlanner,
    executor: MotionExecutor,
    solve_ik: Callable,
    scene_state: dict,
    params: MotionParams,
    qpos: np.ndarray,
    held_category: str | None = None,
    held_place_z: float | None = None,
    motion_planner: MotionPlanner | None = None,
    phase: str = "all",
) -> tuple[list[np.ndarray], str | None, float | None]:
    """Plan one skill's joint-target segment from the given (live) scene_state.

    Returns ``(segment, held_category, held_place_z)``; the carry-over state is
    threaded into the next segment. The segment-level runtime re-reads object and
    joint poses into ``scene_state`` before each call so each subtask anchors to
    the world's actual current state (e.g. ``close`` to where the drawer really
    is); ``run_skills`` calls it over a static scene_state.
    """
    primitive = SKILL_PRIMITIVES.get(skill.name)
    if primitive is None:
        raise ValueError(f"Unknown skill: {skill.name!r}")
    # Single motion seam: wrap the bare Genesis ``solve_ik`` in the collision-blind
    # GenesisBackend when no collision-aware planner is wired, so primitives/executor
    # always go through one MotionPlanner instead of a raw callable.
    ctx = SkillContext(
        skill=skill,
        index=index,
        skills=skills,
        planner=planner,
        executor=executor,
        motion_planner=motion_planner or GenesisBackend(solve_ik),
        params=params,
        scene_state=scene_state,
        qpos=qpos,
        category=_resolve_skill_category(
            skill, scene_state.get("assets", {}), held_category
        ),
        held_category=held_category,
        held_place_z=held_place_z,
        phase=phase,
    )
    logger.debug(
        "[skills] subtask %d %s(%s) phase=%s; live joint_positions=%s",
        index, skill.name, skill.target, phase,
        scene_state.get("joint_positions", {}),
    )
    seg = primitive(ctx)
    return seg, ctx.held_category, ctx.held_place_z


def run_skills(
    skills: list[Skill],
    planner: GraspPlanner,
    executor: MotionExecutor,
    solve_ik: Callable,
    scene_state: dict,
    params: MotionParams,
) -> list[np.ndarray]:
    """Plan a full skill sequence open-loop over one static ``scene_state``.

    Thin loop over ``plan_segment``. Genesis-free tests use this; the simulator
    runtime calls ``plan_segment`` per subtask with a live-refreshed scene_state.

    ``scene_state`` must contain ``"home_qpos"`` and ``"positions"``; optional
    ``"object_heights"``, ``"assets"``, ``"object_quats"``, ``"object_scales"``,
    ``"table_surface_z"``, ``"joint_positions"``.
    """
    traj: list[np.ndarray] = []
    qpos = scene_state["home_qpos"].copy()
    held_category: str | None = None
    held_place_z: float | None = None

    for i, skill in enumerate(skills):
        seg, held_category, held_place_z = plan_segment(
            skill, i, skills, planner, executor, solve_ik, scene_state, params,
            qpos, held_category, held_place_z,
        )
        traj += seg
        qpos = traj[-1]

    return traj
