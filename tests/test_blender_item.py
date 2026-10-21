"""Blender item -> asset directory: named collision boxes carry handles and openings."""

import json

import numpy as np
import pytest
import trimesh

from robotsmith.assets.library import AssetLibrary
from robotsmith.assets.producers.blender_item import compile_item

Z_UP_TO_Y_UP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)


def _part(path, boxes):
    """Write a glb the way Blender exports it: named objects, +Y up."""
    scene = trimesh.Scene()
    for name, (lo, hi) in boxes.items():
        box = trimesh.creation.box(bounds=np.array([lo, hi], dtype=float))
        box.apply_transform(Z_UP_TO_Y_UP)
        scene.add_geometry(box, node_name=name, geom_name=name)
    scene.export(str(path))


def test_drawer_cabinet_compiles_to_loadable_asset(tmp_path):
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

    compile_item(item, tmp_path / "assets" / "generated" / "cab")
    asset = AssetLibrary(tmp_path / "assets").get("cab")

    assert asset.is_fixture and [j.name for j in asset.movable_joints] == ["drawer_slide"]
    assert asset.metadata.size_cm == pytest.approx([37.4, 40.0, 18.0])
    assert asset.metadata.handles[0]["local_pos"] == pytest.approx([0.17, 0.0, 0.106])
    opening = asset.metadata.place_targets[0]
    assert opening["joint"] == "drawer_slide"
    assert np.allclose(opening["box"], [[-0.12, -0.17, 0.03], [0.14, 0.17, 0.04]])
    assert opening["lip_local"] == pytest.approx([0.14, 0.0, 0.04])
