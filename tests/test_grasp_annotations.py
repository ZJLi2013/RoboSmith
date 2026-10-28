import pytest

from robotsmith.assets import store
from robotsmith.grasp import annotations

UP = [1.0, 0.0, 0.0, 0.0]
YAW_90 = [0.707107, 0.0, 0.0, 0.707107]
POSE = [0.42, 0.27, 0.9047]
REPAIR = {"tip_xyz": [0.42, 0.2745, 1.0085], "approach": [0.0, 0.0, -1.0], "finger_axis": [0.0, 1.0, 0.0],
          "ik_seed": None}


@pytest.fixture
def asset(tmp_path):
    d = tmp_path / "radio_01"
    d.mkdir()
    (d / "model.urdf").write_text("<robot name='radio_01'/>")
    store.new_asset(d, "rigid", "test", {}, license="MIT")
    annotations.write_grasp(d, REPAIR, POSE, UP, lift_m=0.1663, by="agent:test", run={"object_xy": [0.42, 0.27], "seed": 0})
    return d


def test_written_grasp_comes_back_in_the_object_frame(asset):
    (c,) = annotations.current_grasps(asset)
    tip, approach, finger = annotations.to_world(c, POSE, UP)
    assert tip == pytest.approx(REPAIR["tip_xyz"], abs=1e-4)
    assert approach == pytest.approx(REPAIR["approach"], abs=1e-4)
    assert finger == pytest.approx(REPAIR["finger_axis"], abs=1e-4)

    tip, _, finger = annotations.to_world(annotations.to_local(tip, approach, finger, POSE, YAW_90),
                                          [0.5, 0.0, POSE[2]], YAW_90)
    assert tip == pytest.approx([0.5, 0.0045, 1.0085], abs=1e-4)
    assert finger == pytest.approx([0.0, 1.0, 0.0], abs=1e-4)


def _rebuild(d):
    (d / "model.urdf").write_text("<robot name='radio_01_v2'/>")
    data = store.read_asset(d)
    data["build"]["hash"] = store.build_hash(d, data["build"]["files"])
    store.write_asset(d, data)


def _bump_revision(d):
    data = store.read_asset(d)
    data["revision"] = 2
    store.write_asset(d, data)


@pytest.mark.parametrize("make_stale, embodiment", [
    (_rebuild, "panda"),
    (_bump_revision, "panda"),
    (lambda d: None, "ur5"),
    (lambda d: (d / store.ASSET_FILE).unlink(), "panda"),
])
def test_grasp_is_dropped_once_its_binding_no_longer_holds(asset, make_stale, embodiment):
    make_stale(asset)
    assert annotations.current_grasps(asset, embodiment=embodiment) == []
