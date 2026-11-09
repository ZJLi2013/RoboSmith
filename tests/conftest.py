from contextlib import contextmanager

import numpy as np
import pytest

from robotsmith.execution.gate import ExecutionGate
from robotsmith.motion.planner import GenesisBackend
from robotsmith.scenario.load import pick_part, place_part
from robotsmith.skills import SKILL_DEFS, Skill, SkillContext
from robotsmith.skills.base import _resolve_skill_category


class ClearEnv:
    """A scene where nothing touches anything; ``writes`` stands in for the real-write count."""

    writes = 0
    poses = ()
    joint_limits = (np.full(9, -10.0), np.full(9, 10.0))
    joined_links = frozenset()

    def contacts_at(self, qpos, held=None, placed=None):
        return []

    def lever_arms(self, held=None):
        return np.r_[np.full(7, 0.01), 1.0, 1.0]

    def hand_pose(self, qpos):
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    def object_pose(self, name):
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    @contextmanager
    def posed(self, qpos, objects):
        outer = self.poses
        self.poses = (*outer, (np.asarray(qpos), dict(objects)))
        try:
            yield
        finally:
            self.poses = outer

    def state_fingerprint(self):
        return f"{self.writes}:posed{len(self.poses)}" if self.poses else str(self.writes)


@pytest.fixture
def gate():
    return ExecutionGate(ClearEnv())


def pick_skill(obj):
    return Skill("pick", obj, params={"actions": pick_part(obj)})


def place_skill(surface, obj="die"):
    return Skill("place", surface, params={"actions": place_part(obj, surface)})


def plan_skills(skills, planner, executor, solve_ik, scene_state, params, *, gate):
    """Each skill's primitive in turn over one static ``scene_state``, as ``EpisodeRun`` calls them, stepping
    nothing: only the first motion is current against the scene."""
    motions, qpos = [], scene_state["home_qpos"].copy()
    held = held_category = held_place_z = None
    for i, skill in enumerate(skills):
        skill_def = SKILL_DEFS.get(skill.name)
        if skill_def is None:
            raise ValueError(f"Unknown skill: {skill.name!r}")
        ctx = SkillContext(skill=skill, index=i, skills=skills, planner=planner, executor=executor,
                           motion_planner=GenesisBackend(solve_ik), params=params, scene_state=scene_state,
                           qpos=qpos, category=_resolve_skill_category(skill, scene_state.get("assets", {}),
                                                                       held_category),
                           gate=gate, held=held, held_category=held_category, held_place_z=held_place_z)
        motions.append(skill_def.primitive(ctx))
        held_category, held_place_z = ctx.held_category, ctx.held_place_z
        if motions[-1].trajectory:
            qpos = motions[-1].trajectory[-1]
        held = skill.target if skill.name == "pick" else None if skill.name == "place" else held
    return motions
