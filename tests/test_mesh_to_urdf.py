"""Tests for mesh_to_urdf conversion."""

from pathlib import Path
import trimesh
import numpy as np

from robotsmith.gen.mesh_to_urdf import mesh_to_urdf
from robotsmith.gen.catalog import tags_from_prompt, name_from_prompt, catalog_asset


class TestMeshToURDF:
    def test_box_mesh(self, tmp_path):
        mesh = trimesh.primitives.Box(extents=[0.1, 0.1, 0.1])
        urdf = mesh_to_urdf(mesh, tmp_path / "box_test", name="test_box")
        assert urdf.exists()
        content = urdf.read_text()
        assert "<robot" in content
        assert 'name="test_box"' in content
        assert "visual.obj" in content
        assert "collision.obj" in content
        assert (tmp_path / "box_test" / "visual.obj").exists()
        assert (tmp_path / "box_test" / "collision.obj").exists()

    def test_sphere_mesh(self, tmp_path):
        mesh = trimesh.primitives.Sphere(radius=0.05)
        urdf = mesh_to_urdf(mesh, tmp_path / "sphere_test", name="test_sphere")
        assert urdf.exists()
        content = urdf.read_text()
        assert "mass" in content

    def test_scaling(self, tmp_path):
        mesh = trimesh.primitives.Box(extents=[1.0, 0.5, 0.3])
        mesh_to_urdf(mesh, tmp_path / "scaled", name="scaled", target_size_m=0.1)
        visual = trimesh.load(str(tmp_path / "scaled" / "visual.obj"), force="mesh")
        max_ext = max(visual.bounding_box.extents)
        assert abs(max_ext - 0.1) < 0.01

    def test_mass_override(self, tmp_path):
        mesh = trimesh.primitives.Box(extents=[0.1, 0.1, 0.1])
        urdf = mesh_to_urdf(mesh, tmp_path / "mass_test", name="mass_test", mass_kg=0.5)
        content = urdf.read_text()
        assert 'value="0.5' in content


class TestCatalog:
    def test_tags_from_prompt(self):
        tags = tags_from_prompt("red mug")
        assert "mug" in tags
        assert "red" in tags
        assert "generated" in tags

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
        assert "mug" in asset.tags
        assert (out / "metadata.json").exists()
