"""Tests for AssetLibrary loading + mutation."""

import shutil
from pathlib import Path

from robotsmith.assets.library import AssetLibrary
from robotsmith.assets.schema import Asset, AssetMetadata

REPO_ASSETS = Path(__file__).resolve().parents[1] / "assets"
SUPPORT_ASSETS = ("die_01", "table_simple", "plane")
BUILTIN_COUNT = len(SUPPORT_ASSETS)


def _make_lib(tmp_path: Path) -> AssetLibrary:
    """Writable library seeded with the committed support assets."""
    for name in SUPPORT_ASSETS:
        shutil.copytree(REPO_ASSETS / name, tmp_path / "assets" / name)
    return AssetLibrary(tmp_path / "assets")


class TestAssetLibrary:
    def test_loads_all_assets(self, tmp_path):
        lib = _make_lib(tmp_path)
        assert len(lib) == BUILTIN_COUNT

    def test_get_by_name(self, tmp_path):
        lib = _make_lib(tmp_path)
        a = lib.get("die_01")
        assert a is not None
        assert a.name == "die_01"
        assert a.urdf_path.exists()

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
            metadata=AssetMetadata(),
        )
        lib.add(new_asset)
        assert len(lib) == n + 1
        assert lib.get("test_obj") is not None
