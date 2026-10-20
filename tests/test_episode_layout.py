import random

import numpy as np

from robotsmith.rollout.episode_layout import (
    build_episode_positions,
    derive_skill_targets,
)
from robotsmith.skills import Skill
from robotsmith.tasks.task_spec import SuccessNode, TaskSpec


class FakeEnv:
    x_range = (0.40, 0.70)
    y_range = (-0.20, 0.20)

    def get_initial_z(self, name: str) -> float:
        return {"a": 0.03, "b": 0.04, "target": 0.05}.get(name, 0.02)


def test_derive_skill_targets_preserves_first_seen_order():
    task = TaskSpec(
        name="layout",
        instruction="test",
        scene="scene",
        skills=[
            Skill("pick", "a"),
            Skill("place", "target"),
            Skill("pick", "a"),
            Skill("pick", "b"),
            Skill("place", "target"),
        ],
    )

    pick_names, place_names = derive_skill_targets(task)

    assert pick_names == ["a", "b"]
    assert place_names == ["target"]


def test_build_episode_positions_uses_env_z_values():
    positions = build_episode_positions(
        FakeEnv(),
        random.Random(1),
        ["a", "b"],
        [],
    )

    assert set(positions) == {"a", "b"}
    assert positions["a"][2] == 0.03
    assert positions["b"][2] == 0.04


def test_build_episode_positions_aligns_line_task_targets():
    task = TaskSpec(
        name="line",
        instruction="line up",
        scene="scene",
        success=SuccessNode.leaf("objects_aligned", {"axis": "y"}),
        skills=[],
    )

    positions = build_episode_positions(
        FakeEnv(),
        random.Random(3),
        ["a"],
        ["target_0", "target_1", "target_2"],
        task_spec=task,
    )

    targets = [positions[f"target_{i}"] for i in range(3)]
    np.testing.assert_allclose([pos[0] for pos in targets], targets[0][0])
    np.testing.assert_allclose(np.diff([pos[1] for pos in targets]), [0.12, 0.12])
    assert all(pos[2] == 0.03 for pos in targets)


def test_build_episode_positions_samples_collision_free_place_targets():
    positions = build_episode_positions(
        FakeEnv(),
        random.Random(4),
        ["a"],
        ["target"],
    )

    pick_xy = positions["a"][:2]
    place_xy = positions["target"][:2]
    assert float(np.linalg.norm(place_xy - pick_xy)) >= 0.12
