"""Tests for AssetLibrary, search, and builtin assets."""

from pathlib import Path

from robotsmith.assets.builtin import bootstrap_builtin_assets
from robotsmith.assets.library import AssetLibrary
from robotsmith.assets.schema import Asset, AssetMetadata
from robotsmith.assets.search import search_assets

BUILTIN_COUNT = 9  # 3 blocks + 2 boxes + 2 L-blocks + table + plane


def _make_lib(tmp_path: Path) -> AssetLibrary:
    assets_root = tmp_path / "assets"
    bootstrap_builtin_assets(assets_root)
    return AssetLibrary(assets_root)


class TestBootstrap:
    def test_creates_expected_assets(self, tmp_path):
        assets = bootstrap_builtin_assets(tmp_path / "assets")
        assert len(assets) == BUILTIN_COUNT

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

    def test_lblock_has_two_links(self, tmp_path):
        assets = bootstrap_builtin_assets(tmp_path / "assets")
        lblocks = [a for a in assets if a.name.startswith("lblock_")]
        assert len(lblocks) == 2
        for a in lblocks:
            content = a.urdf_path.read_text()
            assert content.count("<link") == 2
            assert "fixed" in content


class TestAssetLibrary:
    def test_loads_all_assets(self, tmp_path):
        lib = _make_lib(tmp_path)
        assert len(lib) == BUILTIN_COUNT

    def test_search_block(self, tmp_path):
        lib = _make_lib(tmp_path)
        results = lib.search("block")
        names = {a.name for a in results}
        assert "block_red" in names
        assert "block_blue" in names

    def test_search_box(self, tmp_path):
        lib = _make_lib(tmp_path)
        results = lib.search("box rectangular")
        names = {a.name for a in results}
        assert "box_small" in names or "box_large" in names

    def test_search_lblock(self, tmp_path):
        lib = _make_lib(tmp_path)
        results = lib.search("lblock non-convex")
        assert len(results) >= 1
        assert any("lblock" in a.name for a in results)

    def test_search_no_match(self, tmp_path):
        lib = _make_lib(tmp_path)
        results = lib.search("xyznonexistent")
        assert len(results) == 0

    def test_get_by_name(self, tmp_path):
        lib = _make_lib(tmp_path)
        a = lib.get("block_red")
        assert a is not None
        assert a.name == "block_red"

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
        results = lib.search("red cube stackable")
        assert len(results) >= 1
        assert results[0].name == "block_red"
