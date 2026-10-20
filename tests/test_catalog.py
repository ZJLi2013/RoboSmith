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
        (out / "model.urdf").write_text("<robot/>")

        asset = catalog_asset(out, "red mug", mass_kg=0.25)
        assert asset.name == "test_gen"
        assert asset.metadata.description == "Generated from prompt: red mug"
        assert asset.metadata.canonical_frame["up_axis"] == "+Z"
        assert asset.metadata.task_poses == {}
        assert (out / "metadata.json").exists()
