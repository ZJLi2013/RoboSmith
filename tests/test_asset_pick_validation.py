"""Rollout wiring for asset-pick validation.

The scenario -> legacy adapter shapes (position_range, success leaf, skill dict)
are covered in test_cap_authoring. Here we only assert the rollout-specific
wiring in _make_scene / _make_task: the fixed "object" name and the
name-derived grasp category.
"""

import random

import numpy as np

from robotsmith.rollout import asset_pick_validation as apv
from robotsmith.rollout.asset_pick_validation import _make_scene, _make_task


def test_make_scene_wires_single_named_object():
    scene = _make_scene("die_01")

    assert scene.name == "pick_die_01"
    assert len(scene.objects) == 1
    obj = scene.objects[0]
    assert obj.asset_query == "die_01"
    assert obj.name_override == "object"


def test_make_task_derives_category_from_asset_name():
    task = _make_task("apple_02")

    assert task.name == "pick_apple_02"
    assert task.success.predicate == "object_above"
    assert [skill.to_dict() for skill in task.skills] == [
        {"name": "pick", "target": "object", "params": {}, "category": "apple"},
    ]


def test_object_xy_pins_the_object_before_settling(monkeypatch):
    monkeypatch.setattr(apv, "build_episode_positions",
                        lambda *a, **k: {"object": np.array([0.55, 0.10, 0.03])})
    monkeypatch.setattr(apv, "run_skills", lambda *a, **k: [])
    reset_xy = {}

    class Env:
        entity_map = {"object": object()}
        object_heights, asset_map, object_quats, table_surface_z = {}, {}, {}, 0.0
        planner = solve_ik = None

        def reset(self, xy_map, settle_steps):
            reset_xy.update(xy_map)
            return {"object": np.array([*xy_map["object"], 0.025])}

    plan = apv._prepare_asset_pick_episode(Env(), _make_task("die_01"), rng=random.Random(0), executor=None,
                                           motion_params=None, settle_steps=1, object_xy=(0.42, -0.27))
    assert reset_xy["object"] == (0.42, -0.27)
    np.testing.assert_allclose(plan.initial_positions["object"], [0.42, -0.27, 0.025])


def test_trial_selection_replays_from_the_start_and_resets_before_recording(monkeypatch):
    monkeypatch.setattr(apv, "build_episode_positions",
                        lambda *a, **k: {"object": np.array([0.55, 0.10, 0.03])})
    resets, trials = [], []

    def skills(task_skills, planner, executor, solve_ik, scene_state, params):
        trials.append(scene_state["grasp_trial"](["q"], "object"))
        return ["q"]

    monkeypatch.setattr(apv, "run_skills", skills)

    class Env:
        entity_map = {"object": object()}
        object_heights, asset_map, object_quats, table_surface_z = {}, {}, {}, 0.0
        planner = solve_ik = None

        def reset(self, xy_map, settle_steps):
            resets.append(dict(xy_map))
            return {"object": np.array([*xy_map["object"], 0.025])}

        def trial_trajectory(self, start, trajectory, target, settle_steps):
            resets.append(dict(start))
            return {"lifted": True, "lift_m": 0.1, "seconds": 0.1}

    apv._prepare_asset_pick_episode(Env(), _make_task("die_01"), rng=random.Random(0), executor=None,
                                    motion_params=None, settle_steps=1, object_xy=(0.42, -0.27),
                                    grasp_selection="trial")
    assert trials == [{"lifted": True, "lift_m": 0.1, "seconds": 0.1}]
    assert resets == [{"object": (0.42, -0.27)}] * 3
