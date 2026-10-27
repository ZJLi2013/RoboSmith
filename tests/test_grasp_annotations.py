import numpy as np
import pytest

from robotsmith.assets import store
from robotsmith.grasp import annotations

YAW_90 = [0.707107, 0.0, 0.0, 0.707107]
REPAIR = {"tip_xyz": [0.42, 0.2745, 1.0085], "approach": [0.0, 0.0, -1.0], "finger_axis": [0.0, 1.0, 0.0],
          "ik_seed": None}


@pytest.fixture
def asset(tmp_path):
    d = tmp_path / "radio_01"
    d.mkdir()
    (d / "model.urdf").write_text("<robot name='radio_01'/>")
    store.new_asset(d, "rigid", "test", {}, license="MIT")
    annotations.write_grasp(d, REPAIR, [0.42, 0.27, 0.9047], [1.0, 0.0, 0.0, 0.0], lift_m=0.1663,
                            by="agent:test", run={"object_xy": [0.42, 0.27], "seed": 0})
    return d


def test_written_grasp_is_read_back_in_the_object_frame(asset):
    (c,) = annotations.current_grasps(asset)
    tip, approach, finger = annotations.to_world(c, [0.42, 0.27, 0.9047], [1.0, 0.0, 0.0, 0.0])
    assert tip == pytest.approx(REPAIR["tip_xyz"], abs=1e-4)
    assert approach == pytest.approx(REPAIR["approach"], abs=1e-4)
    assert finger == pytest.approx(REPAIR["finger_axis"], abs=1e-4)


def test_object_frame_follows_the_settled_pose():
    local = annotations.to_local(REPAIR["tip_xyz"], REPAIR["approach"], REPAIR["finger_axis"], [0.42, 0.27, 0.9], YAW_90)
    tip, _, finger = annotations.to_world(local, [0.5, 0.0, 0.9], YAW_90)
    assert tip == pytest.approx([0.5, 0.0045, 1.0085], abs=1e-4)
    assert finger == pytest.approx([0.0, 1.0, 0.0], abs=1e-4)


def test_rebuilt_asset_drops_the_grasp(asset):
    (asset / "model.urdf").write_text("<robot name='radio_01_v2'/>")
    data = store.read_asset(asset)
    data["build"]["hash"] = store.build_hash(asset, data["build"]["files"])
    store.write_asset(asset, data)
    assert annotations.current_grasps(asset) == []


def test_other_embodiment_does_not_use_the_grasp(asset):
    assert annotations.current_grasps(asset, embodiment="ur5") == []


def test_new_revision_drops_the_grasp(asset):
    data = store.read_asset(asset)
    data["revision"] = 2
    store.write_asset(asset, data)
    assert annotations.current_grasps(asset) == []


def test_asset_without_asset_json_has_no_grasps(tmp_path):
    assert annotations.current_grasps(tmp_path) == []
