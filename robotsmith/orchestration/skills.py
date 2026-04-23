"""Skill — atomic manipulation primitive for task orchestration.

A task is defined as an ordered list of Skills. The generic runner
`run_skills()` executes them sequentially, calling into GraspPlanner
and MotionExecutor for each skill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from robotsmith.grasp.planner import GraspPlanner
from robotsmith.motion.executor import MotionExecutor
from robotsmith.motion.params import MotionParams


@dataclass
class Skill:
    """Single atomic manipulation action."""

    name: str               # "pick" | "place"
    target: str             # object name — key into scene_state["positions"]
    category: str           # GraspTemplate category
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "target": self.target,
            "category": self.category,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Skill:
        return cls(
            name=d["name"],
            target=d["target"],
            category=d["category"],
            params=d.get("params", {}),
        )


def run_skills(
    skills: list[Skill],
    planner: GraspPlanner,
    executor: MotionExecutor,
    solve_ik: Callable,
    scene_state: dict,
    params: MotionParams,
) -> list[np.ndarray]:
    """Execute a skill sequence, returning the concatenated joint trajectory.

    ``scene_state`` must contain:
      - ``"home_qpos"``: np.ndarray — robot home joint positions
      - ``"positions"``: dict[str, np.ndarray] — object name → world-frame pos

    Optional:
      - ``"object_heights"``: dict[str, float] — object name → height in meters
    """
    traj: list[np.ndarray] = []
    qpos = scene_state["home_qpos"].copy()
    heights = scene_state.get("object_heights", {})
    n = len(skills)

    for i, skill in enumerate(skills):
        obj_pos = scene_state["positions"][skill.target]

        if skill.name == "pick":
            obj_h = (
                skill.params.get("object_height")
                or heights.get(skill.target)
            )
            plan = planner.plan(
                obj_pos,
                category=skill.category,
                object_height=obj_h,
            )[0]
            next_is_place = (i + 1 < n and skills[i + 1].name == "place")
            pick_params = MotionParams(
                approach_steps=params.approach_steps,
                descend_steps=params.descend_steps,
                grasp_hold_steps=params.grasp_hold_steps,
                lift_steps=params.lift_steps,
                lift_hold_steps=0 if next_is_place else params.lift_hold_steps,
                transport_steps=params.transport_steps,
                place_descend_steps=params.place_descend_steps,
                release_steps=params.release_steps,
                retreat_steps=params.retreat_steps,
            )
            seg = executor.pick(plan, solve_ik, qpos, pick_params)

        elif skill.name == "place":
            place_plan = planner.plan_place(
                obj_pos,
                category=skill.category,
                place_z_override=skill.params.get("place_z"),
            )
            seg = executor.place(place_plan, solve_ik, qpos, params)

        else:
            raise ValueError(f"Unknown skill: {skill.name!r}")

        traj += seg
        qpos = traj[-1]

    return traj
