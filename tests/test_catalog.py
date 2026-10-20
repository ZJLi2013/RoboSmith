"""Tests for asset cataloging (onboarding generated/imported assets)."""

import trimesh

from robotsmith.assets.catalog import name_from_prompt, catalog_asset


class TestCatalog:
    def test_name_from_prompt(self):
        name = name_from_prompt("a blue teapot on table")
        assert "blue" in name
        assert "teapot" in name

    def test_catalog_asset(self, tmp_path):
        mesh = trimesh.primitives.Cylinder(radius=0.04, height=0.12)
        out = tmp_path / "test_gen"
        out.mkdir()
        mesh.export(str(out / "visual.obj"))
        (out / "model.urdf").write_text(
            '<robot name="r"><link name="l"><visual><geometry>'
            '<mesh filename="visual.obj"/></geometry></visual></link></robot>'
        )

        asset = catalog_asset(out, "red mug")
        assert asset.name == "test_gen"
        assert asset.metadata.description == "Generated from prompt: red mug"
        assert asset.metadata.size_cm == [8.0, 8.0, 12.0]
        assert not asset.metadata.upright_verified
        assert asset.metadata.source == "generated"
        assert (out / "asset.json").exists()
