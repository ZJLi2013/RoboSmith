"""Tests for mesh_to_urdf conversion and produce-time prompt constraints."""

import trimesh

from robotsmith.assets.producers import mesh_to_urdf
from robotsmith.assets.producers.produce import _with_canonical_asset_constraint


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


class TestProducePrompt:
    def test_generation_prompt_adds_canonical_constraint(self):
        prompt = _with_canonical_asset_constraint("red mug")
        assert "red mug" in prompt
        assert "+Z is semantic up" in prompt
