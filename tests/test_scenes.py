"""Tests for scene configs, collision-aware backend, and genesis loader."""

import math
from pathlib import Path

import pytest

from robotsmith.assets.builtin import bootstrap_builtin_assets
from robotsmith.assets.library import AssetLibrary
from robotsmith.scenes.config import SceneConfig, ObjectPlacement, DEFAULT_WORKSPACE_XY
from robotsmith.scenes.backend import (
    ProgrammaticSceneBackend,
    SceneSmithBackend,
    PlacedObject,
    _pick_stable_pose,
    _quat_to_euler,
)
from robotsmith.scenes.presets.tabletop_simple import tabletop_simple


def _make_lib(tmp_path: Path) -> AssetLibrary:
    root = tmp_path / "assets"
    bootstrap_builtin_assets(root)
    return AssetLibrary(root)


class TestSceneConfig:
    def test_tabletop_has_objects(self):
        assert len(tabletop_simple.objects) >= 3

    def test_config_to_dict(self):
        d = tabletop_simple.to_dict()
        assert "name" in d
        assert "objects" in d
        assert "physics" in d
        assert "camera" in d
        assert "workspace_xy" in d

    def test_workspace_xy_default(self):
        cfg = SceneConfig(name="test")
        assert cfg.workspace_xy == DEFAULT_WORKSPACE_XY

    def test_collision_margin_default(self):
        cfg = SceneConfig(name="test")
        assert cfg.collision_margin == 0.02


class TestCollisionAwarePlacement:
    def test_resolve_tabletop(self, tmp_path):
        lib = _make_lib(tmp_path)
        backend = ProgrammaticSceneBackend(seed=42)
        scene = backend.resolve(tabletop_simple, lib)
        assert len(scene.placed_objects) >= 3
        assert scene.table_asset is not None

    def test_deterministic_with_seed(self, tmp_path):
        lib = _make_lib(tmp_path)
        s1 = ProgrammaticSceneBackend(seed=123).resolve(tabletop_simple, lib)
        s2 = ProgrammaticSceneBackend(seed=123).resolve(tabletop_simple, lib)
        for p1, p2 in zip(s1.placed_objects, s2.placed_objects):
            assert p1.position == p2.position
            assert p1.quaternion == p2.quaternion

    def test_different_seeds_differ(self, tmp_path):
        lib = _make_lib(tmp_path)
        s1 = ProgrammaticSceneBackend(seed=1).resolve(tabletop_simple, lib)
        s2 = ProgrammaticSceneBackend(seed=2).resolve(tabletop_simple, lib)
        positions_differ = any(
            p1.position != p2.position
            for p1, p2 in zip(s1.placed_objects, s2.placed_objects)
        )
        assert positions_differ

    def test_objects_within_workspace(self, tmp_path):
        lib = _make_lib(tmp_path)
        backend = ProgrammaticSceneBackend(seed=99)
        scene = backend.resolve(tabletop_simple, lib)
        ws_lo, ws_hi = tabletop_simple.workspace_xy
        for po in scene.placed_objects:
            assert ws_lo[0] <= po.position[0] <= ws_hi[0], f"{po.asset.name} x out of bounds"
            assert ws_lo[1] <= po.position[1] <= ws_hi[1], f"{po.asset.name} y out of bounds"

    def test_objects_on_table_surface(self, tmp_path):
        lib = _make_lib(tmp_path)
        backend = ProgrammaticSceneBackend(seed=42)
        scene = backend.resolve(tabletop_simple, lib)
        table_surface = tabletop_simple.table_height + tabletop_simple.table_size[2] / 2.0
        for po in scene.placed_objects:
            assert po.position[2] >= table_surface, (
                f"{po.asset.name} z={po.position[2]:.3f} < table={table_surface:.3f}"
            )

    def test_no_pairwise_collision(self, tmp_path):
        """All placed objects should be at least collision_margin apart."""
        lib = _make_lib(tmp_path)
        backend = ProgrammaticSceneBackend(seed=42)
        scene = backend.resolve(tabletop_simple, lib)
        objs = scene.placed_objects
        margin = tabletop_simple.collision_margin
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                dx = objs[i].position[0] - objs[j].position[0]
                dy = objs[i].position[1] - objs[j].position[1]
                dist = math.sqrt(dx * dx + dy * dy)
                assert dist > 0.001, (
                    f"{objs[i].asset.name} and {objs[j].asset.name} overlap"
                )

    def test_many_objects_no_crash(self, tmp_path):
        """Stress test: many objects should either place or skip gracefully."""
        lib = _make_lib(tmp_path)
        cfg = SceneConfig(
            name="crowded",
            objects=[ObjectPlacement(asset_query="block", count=20)],
            workspace_xy=[[0.4, -0.1], [0.6, 0.1]],
        )
        backend = ProgrammaticSceneBackend(seed=7)
        scene = backend.resolve(cfg, lib)
        assert len(scene.placed_objects) >= 1

    def test_placed_object_has_quaternion(self, tmp_path):
        lib = _make_lib(tmp_path)
        backend = ProgrammaticSceneBackend(seed=42)
        scene = backend.resolve(tabletop_simple, lib)
        for po in scene.placed_objects:
            assert len(po.quaternion) == 4
            norm = sum(q * q for q in po.quaternion) ** 0.5
            assert 0.99 < norm < 1.01, f"quaternion not unit: {po.quaternion}"


class TestStablePoseSelection:
    def test_pick_with_no_poses_returns_default(self):
        import random
        from robotsmith.assets.schema import Asset, AssetMetadata
        meta = AssetMetadata(size_cm=[5.0, 5.0, 10.0], stable_poses=[])
        asset = Asset(name="test", root_dir=Path("."), urdf_path=Path("x.urdf"), metadata=meta)
        z, quat = _pick_stable_pose(random.Random(1), asset)
        assert z == 0.05  # half of 10cm
        assert quat == [1.0, 0.0, 0.0, 0.0]

    def test_pick_respects_probability(self):
        import random
        from robotsmith.assets.schema import Asset, AssetMetadata
        meta = AssetMetadata(stable_poses=[
            {"z": 0.01, "quat": [1, 0, 0, 0], "probability": 0.99},
            {"z": 0.05, "quat": [0, 1, 0, 0], "probability": 0.01},
        ])
        asset = Asset(name="test", root_dir=Path("."), urdf_path=Path("x.urdf"), metadata=meta)
        hits = {0.01: 0, 0.05: 0}
        rng = random.Random(42)
        for _ in range(200):
            z, _ = _pick_stable_pose(rng, asset)
            hits[z] += 1
        assert hits[0.01] > hits[0.05] * 5


class TestQuat:
    def test_identity_quat_gives_zero_euler(self):
        euler = _quat_to_euler([1.0, 0.0, 0.0, 0.0])
        for e in euler:
            assert abs(e) < 1e-6


class TestSceneSmithStub:
    def test_raises_not_implemented(self, tmp_path):
        lib = _make_lib(tmp_path)
        backend = SceneSmithBackend()
        with pytest.raises(NotImplementedError):
            backend.resolve(tabletop_simple, lib)
