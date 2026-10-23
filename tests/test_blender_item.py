"""Blender item -> asset directory: named collision boxes carry handles and openings."""

import json

import numpy as np
import pytest
import trimesh

from robotsmith.assets import store
from robotsmith.assets.library import AssetLibrary
from robotsmith.assets.producers.blender_item import compile_item
from robotsmith.assets.review.package import admit, readings

Z_UP_TO_Y_UP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)


def _part(path, boxes):
    """Write a glb the way Blender exports it: named objects, +Y up."""
    scene = trimesh.Scene()
    for name, (lo, hi) in boxes.items():
        box = trimesh.creation.box(bounds=np.array([lo, hi], dtype=float))
        box.apply_transform(Z_UP_TO_Y_UP)
        scene.add_geometry(box, node_name=name, geom_name=name)
    scene.export(str(path))


def _cabinet(tmp_path):
    item = tmp_path / "item"
    item.mkdir()
    _part(item / "carcass.glb", {"bottom": ([-0.15, -0.2, 0.0], [0.15, 0.2, 0.02])})
    _part(item / "drawer.glb", {
        "tray_floor": ([-0.12, -0.17, 0.03], [0.14, 0.17, 0.04]),
        "front_panel": ([0.14, -0.2, 0.03], [0.15, 0.2, 0.18]),
    })
    _part(item / "handle.glb", {
        "handle_neck": ([0.15, -0.006, 0.1], [0.19, 0.006, 0.112]),
        "handle_knob": ([0.19, -0.017, 0.089], [0.224, 0.017, 0.123]),
    })
    (item / "item.json").write_text(json.dumps({
        "name": "drawer_cabinet", "kind": "articulated",
        "parts": [
            {"name": "carcass", "mesh": "carcass.glb", "mass_kg": 5.0},
            {"name": "drawer", "mesh": "drawer.glb", "mass_kg": 0.5},
            {"name": "handle", "mesh": "handle.glb", "mass_kg": 0.02, "parent": "drawer"},
        ],
        "joints": [{"name": "drawer_slide", "type": "prismatic", "parent": "carcass", "child": "drawer",
                    "axis": [1, 0, 0], "origin": [0.15, 0, 0.1], "lower": 0.0, "upper": 0.2}],
        "handles": [{"part": "handle", "object": "handle_neck"}],
        "place_targets": [{"name": "drawer_opening", "part": "drawer", "surface": "tray_floor"}],
    }), encoding="utf-8")
    return item


def test_drawer_cabinet_compiles_to_loadable_asset(tmp_path):
    compile_item(_cabinet(tmp_path), tmp_path / "assets" / "cab")
    asset = AssetLibrary(tmp_path / "assets").get("cab")

    assert asset.is_fixture and [j.name for j in asset.movable_joints] == ["drawer_slide"]
    assert asset.metadata.size_cm == pytest.approx([37.4, 40.0, 18.0])
    assert asset.metadata.handles[0]["local_pos"] == pytest.approx([0.17, 0.0, 0.106])
    opening = asset.metadata.place_targets[0]
    assert opening["joint"] == "drawer_slide"
    assert np.allclose(opening["box"], [[-0.12, -0.17, 0.03], [0.14, 0.17, 0.04]])
    assert opening["lip_local"] == pytest.approx([0.14, 0.0, 0.04])


def test_rigid_part_collides_as_its_mesh(tmp_path):
    item = tmp_path / "item"
    item.mkdir()
    ball = trimesh.creation.icosphere(radius=0.035)
    ball.apply_translation([0, 0, 0.035])
    ball.apply_transform(Z_UP_TO_Y_UP)
    scene = trimesh.Scene()
    scene.add_geometry(ball, node_name="apple_body", geom_name="apple_body")
    scene.export(str(item / "apple.glb"))
    (item / "item.json").write_text(json.dumps({
        "name": "apple", "kind": "rigid",
        "parts": [{"name": "apple", "mesh": "apple.glb", "mass_kg": 0.15, "collision": "mesh"}],
    }), encoding="utf-8")
    asset_dir = tmp_path / "apple"
    compile_item(item, asset_dir)

    r = readings(asset_dir)
    assert r["kind"] == "rigid" and r["collision_boxes"] == []
    assert [m["file"] for m in r["collision_meshes"]] == ["apple.obj"]
    assert np.allclose(r["collision_meshes"][0]["bounds_at_rest"], [[-0.035, -0.035, 0.0], [0.035, 0.035, 0.07]], atol=1e-3)


def test_review_verdict_gates_library_admission(tmp_path):
    asset_dir = tmp_path / "assets" / "cab"
    compile_item(_cabinet(tmp_path), asset_dir)
    r = readings(asset_dir)
    assert {b["name"] for b in r["collision_boxes"]} >= {"tray_floor", "handle_neck"}
    assert r["joints"][0]["upper"] == 0.2 and r["place_targets"][0]["joint"] == "drawer_slide"

    claims = [{"claim": "drawer stays in the carcass", "verdict": "fail", "evidence": "probes/01.out"}]
    assert admit(asset_dir, {"decision": "reject", "claims": claims}, by="agent:t") == ["llm_review failed"]

    store.append_check(asset_dir, "upright", "pass", {"method": "manual_visual_qa"}, by="human")
    assert not store.load_metadata(asset_dir).upright_verified, "human_upright_is_not_enough"

    claims[0]["verdict"] = "pass"
    assert admit(asset_dir, {"decision": "accept", "claims": claims}, by="agent:t") == []
    assert AssetLibrary(tmp_path / "assets").get("cab").metadata.upright_verified
    checks = [json.loads(l) for l in (asset_dir / "checks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(c["check"], c["result"]) for c in checks] == [
            ("llm_review", "fail"), ("dedup_check", "none"), ("upright", "pass"), ("llm_review", "pass")]
