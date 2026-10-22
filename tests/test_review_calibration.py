"""Review calibration cases: sources exist, and the mutations make the defect they claim."""

import shutil

import pytest
import trimesh
from test_blender_item import _cabinet, _part

from robotsmith.assets import store
from robotsmith.assets.producers.blender_item import compile_item
from robotsmith.assets.review import calibration
from robotsmith.assets.schema import parse_urdf_collision_boxes


def _hash_is_current(asset_dir):
    data = store.read_asset(asset_dir)
    return store.build_hash(asset_dir, data["build"]["files"]) == data["build"]["hash"]


def test_cases_name_their_sources():
    for case in calibration.load_cases():
        assert case["expect"] in ("accept", "reject")
        assert ("build" in case) != ("library" in case)
        assert (calibration.HERE / case.get("build", case["request"])).exists()
        assert (calibration.HERE / case["request"]).exists()
        assert "library" not in case or (calibration.LIBRARY / case["library"]).is_dir()
        assert case.get("mutate", "solid_tray") in calibration.MUTATIONS


def test_solid_tray_fills_the_tray_with_one_box(tmp_path):
    item = _cabinet(tmp_path)
    _part(item / "drawer.glb", {
        "tray_floor": ([-0.12, -0.17, 0.03], [0.14, 0.17, 0.04]),
        "tray_back": ([-0.13, -0.17, 0.03], [-0.12, 0.17, 0.12]),
        "front_panel": ([0.14, -0.2, 0.03], [0.15, 0.2, 0.18]),
    })
    asset = tmp_path / "cab"
    compile_item(item, asset)

    calibration.solid_tray(asset)

    trays = [b for b in parse_urdf_collision_boxes(asset / "model.urdf") if b.name.startswith("tray_")]
    assert len(trays) == 1 and trays[0].size == pytest.approx((0.27, 0.34, 0.09))
    assert _hash_is_current(asset)


def test_tipped_lays_the_mug_on_its_side(tmp_path):
    asset = tmp_path / "mug"
    shutil.copytree(calibration.LIBRARY / "mug_02", asset, ignore=shutil.ignore_patterns("checks.jsonl", "visual.obj"))
    calibration._refresh(asset)
    col_x, col_y, col_z = trimesh.load(str(asset / "collision.obj"), force="mesh").extents
    vis_x, vis_y, vis_z = trimesh.load(str(asset / "visual.glb")).extents

    calibration.tipped(asset)

    col = trimesh.load(str(asset / "collision.obj"), force="mesh")
    assert col.extents == pytest.approx([col_x, col_z, col_y], abs=1e-4)
    assert col.bounds[0][2] == pytest.approx(0.0, abs=1e-6)
    assert trimesh.load(str(asset / "visual.glb")).extents == pytest.approx([vis_x, vis_z, vis_y], abs=1e-4)
    assert _hash_is_current(asset)
