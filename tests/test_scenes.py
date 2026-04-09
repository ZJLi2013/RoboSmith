"""Tests for scene configs and programmatic backend."""

from pathlib import Path

from robotsmith.assets.builtin import bootstrap_builtin_assets
from robotsmith.assets.library import AssetLibrary
from robotsmith.scenes.backend import ProgrammaticSceneBackend, SceneSmithBackend
from robotsmith.scenes.presets.tabletop_simple import tabletop_simple
import pytest


def _make_lib(tmp_path: Path) -> AssetLibrary:
    root = tmp_path / "assets"
    bootstrap_builtin_assets(root)
    return AssetLibrary(root)


class TestScenePresets:
    def test_tabletop_has_objects(self):
        assert len(tabletop_simple.objects) >= 3

    def test_config_to_dict(self):
        d = tabletop_simple.to_dict()
        assert "name" in d
        assert "objects" in d
        assert "physics" in d
        assert "camera" in d


class TestProgrammaticBackend:
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

    def test_different_seeds_differ(self, tmp_path):
        lib = _make_lib(tmp_path)
        s1 = ProgrammaticSceneBackend(seed=1).resolve(tabletop_simple, lib)
        s2 = ProgrammaticSceneBackend(seed=2).resolve(tabletop_simple, lib)
        positions_differ = any(
            p1.position != p2.position
            for p1, p2 in zip(s1.placed_objects, s2.placed_objects)
        )
        assert positions_differ


class TestSceneSmithStub:
    def test_raises_not_implemented(self, tmp_path):
        lib = _make_lib(tmp_path)
        backend = SceneSmithBackend()
        with pytest.raises(NotImplementedError):
            backend.resolve(tabletop_simple, lib)
