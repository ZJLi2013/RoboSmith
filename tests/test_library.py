"""Tests for AssetLibrary, search, and builtin assets."""

from pathlib import Path
import tempfile
import shutil

from robotsmith.assets.builtin import bootstrap_builtin_assets
from robotsmith.assets.library import AssetLibrary
from robotsmith.assets.schema import Asset, AssetMetadata
from robotsmith.assets.search import search_assets


def _make_lib(tmp_path: Path) -> AssetLibrary:
    assets_root = tmp_path / "assets"
    bootstrap_builtin_assets(assets_root)
    return AssetLibrary(assets_root)


class TestBootstrap:
    def test_creates_12_assets(self, tmp_path):
        assets = bootstrap_builtin_assets(tmp_path / "assets")
        assert len(assets) == 12

    def test_urdf_files_exist(self, tmp_path):
        assets = bootstrap_builtin_assets(tmp_path / "assets")
        for a in assets:
            assert a.urdf_path.exists(), f"{a.name} URDF missing"
            content = a.urdf_path.read_text()
            assert "<robot" in content
            assert "<link" in content

    def test_metadata_files_exist(self, tmp_path):
        assets = bootstrap_builtin_assets(tmp_path / "assets")
        for a in assets:
            meta_path = a.root_dir / "metadata.json"
            assert meta_path.exists(), f"{a.name} metadata missing"


class TestAssetLibrary:
    def test_loads_all_assets(self, tmp_path):
        lib = _make_lib(tmp_path)
        assert len(lib) == 12

    def test_search_cup(self, tmp_path):
        lib = _make_lib(tmp_path)
        results = lib.search("cup")
        assert len(results) >= 1
        assert results[0].name == "mug_red"

    def test_search_block(self, tmp_path):
        lib = _make_lib(tmp_path)
        results = lib.search("block")
        names = {a.name for a in results}
        assert "block_red" in names
        assert "block_blue" in names

    def test_search_fork(self, tmp_path):
        lib = _make_lib(tmp_path)
        results = lib.search("fork")
        assert any(a.name == "fork_silver" for a in results)

    def test_search_no_match(self, tmp_path):
        lib = _make_lib(tmp_path)
        results = lib.search("xyznonexistent")
        assert len(results) == 0

    def test_get_by_name(self, tmp_path):
        lib = _make_lib(tmp_path)
        a = lib.get("mug_red")
        assert a is not None
        assert a.name == "mug_red"

    def test_get_missing(self, tmp_path):
        lib = _make_lib(tmp_path)
        assert lib.get("does_not_exist") is None

    def test_add_asset(self, tmp_path):
        lib = _make_lib(tmp_path)
        n = len(lib)
        new_asset = Asset(
            name="test_obj",
            root_dir=tmp_path,
            urdf_path=tmp_path / "model.urdf",
            metadata=AssetMetadata(tags=["test"]),
        )
        lib.add(new_asset)
        assert len(lib) == n + 1
        assert lib.get("test_obj") is not None


class TestSearch:
    def test_tag_overlap_scoring(self, tmp_path):
        lib = _make_lib(tmp_path)
        results = lib.search("red block")
        assert results[0].name == "block_red"

    def test_compound_query(self, tmp_path):
        lib = _make_lib(tmp_path)
        results = lib.search("red cup container")
        assert len(results) >= 1
        assert results[0].name == "mug_red"
