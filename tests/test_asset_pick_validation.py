"""Asset-pick validation rollout: _make_scene / _make_task (the fixed "object"
name, the name-derived grasp category), the object pinned at object_xy before
settling, and the pick run through an EpisodeRun.
"""

import random
from types import SimpleNamespace

import numpy as np

from conftest import ClearEnv
from robotsmith.motion.params import MotionParams
from robotsmith.motion.plan_state import Attach
from robotsmith.rollout import asset_pick_validation as apv
from robotsmith.rollout.asset_pick_validation import _make_scene, _make_task
from robotsmith.scenario.load import pick_part


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
        {"name": "pick", "target": "object", "params": {"actions": pick_part("object")}, "category": "apple"},
    ]


def test_object_xy_pins_the_object_before_settling(monkeypatch):
    monkeypatch.setattr(apv, "build_episode_positions",
                        lambda *a, **k: {"object": np.array([0.55, 0.10, 0.03])})
    reset_xy = {}

    class Env:
        entity_map = {"object": object()}

        def reset(self, xy_map, settle_steps):
            reset_xy.update(xy_map)
            return {"object": np.array([*xy_map["object"], 0.025])}

    initial = apv._reset_asset_pick_episode(Env(), _make_task("die_01"), rng=random.Random(0), settle_steps=1,
                                            object_xy=(0.42, -0.27))
    assert reset_xy["object"] == (0.42, -0.27)
    np.testing.assert_allclose(initial["object"], [0.42, -0.27, 0.025])


def test_asset_validation_runs_its_pick_through_an_episode_run():
    quat = np.array([0.0, 1.0, 0.0, 0.0])
    candidate = SimpleNamespace(grasp_pos=np.array([0.5, 0.0, 0.12]), grasp_quat=quat,
                                pre_grasp_pos=np.array([0.5, 0.0, 0.2]), pre_grasp_quat=quat,
                                finger_open=0.04, finger_closed=0.0, metadata={"candidate_index": 0})

    class Env(ClearEnv):
        entity_map = {"object": SimpleNamespace(get_pos=lambda: np.array([0.5, 0.0, 0.1]))}
        object_heights, asset_map, object_quats, table_surface_z = {"object": 0.02}, {}, {}, 0.0
        planner = SimpleNamespace(plan=lambda *a, **k: [candidate])

        @staticmethod
        def solve_ik(pos, quat, finger, init_qpos=None):
            return np.r_[np.full(7, pos[0]), 0.0, 0.0]

    lists = []

    class Executor:
        last_motion_trace = None

        def run(self, name, actions, motion_planner, start, params, **k):
            lists.append(actions)
            approach = [np.asarray(q) for q in actions[1].path]
            return SimpleNamespace(trajectory=approach + [np.full(9, 1.0)], segments=[
                {"label": "approach", "span": [0, len(approach)], "holding": False, "allow": None},
                {"label": "lift", "span": [len(approach), len(approach) + 1], "holding": False, "allow": None}])

    recorded = []
    run = apv._run_pick(Env(), _make_task("die_01"), executor=Executor(), motion_params=MotionParams(), dataset=None,
                        record_fn=lambda env, dataset, seg, task, **k: recorded.append(len(seg)) or [],
                        camera_names=("up",), grasp_selection="trial")
    whole = lists[-1]
    assert [a.label if not isinstance(a, Attach) else "attach" for a in whole] == [
        "open", "approach", "descend", "close", "attach", "lift"]
    np.testing.assert_allclose(whole[-1].pos, candidate.grasp_pos + [0.0, 0.0, 0.10])
    assert run.stopped is None and recorded == [run.frame_count] and run.frame_count > 1
    assert apv._pick_trace(run.skill_traces)["selected_candidate_index"] == 0
