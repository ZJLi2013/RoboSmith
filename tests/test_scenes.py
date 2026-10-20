"""Tests for the collision-aware scene backend and upright pose resolution."""

import math
import shutil
from pathlib import Path

import pytest

from robotsmith.assets.library import AssetLibrary
from robotsmith.assets.schema import Asset, AssetMetadata
from robotsmith.scenes.config import SceneConfig, ObjectPlacement
from robotsmith.scenes.backend import ProgrammaticSceneBackend, _resolve_upright_pose
from robotsmith.scenes.pose_utils import task_pose_quat, task_pose_verified, task_upright_quat

tabletop_simple = SceneConfig(
    name="tabletop_simple",
    description="Simple tabletop with several blocks",
    objects=[ObjectPlacement(asset_query="die_01", count=5)],
    table_size=[1.2, 0.8, 0.05],
    table_height=0.75,
    workspace_xy=[[0.35, -0.20], [0.65, 0.20]],
)

REPO_OBJECTS = Path(__file__).resolve().parents[1] / "assets" / "objects"


@pytest.fixture
def lib(tmp_path: Path) -> AssetLibrary:
    """Writable library seeded with the committed support assets."""
    objects = tmp_path / "assets" / "objects"
    objects.mkdir(parents=True)
    for name in ("die_01", "table_simple", "plane"):
        shutil.copytree(REPO_OBJECTS / name, objects / name)
    return AssetLibrary(tmp_path / "assets")


def _asset(root: Path, name: str = "test", **meta) -> Asset:
    return Asset(name=name, root_dir=root, urdf_path=root / "model.urdf",
                 metadata=AssetMetadata(**meta))


def _resolve_single(lib, tmp_path, name, **meta):
    asset_dir = tmp_path / name
    asset_dir.mkdir()
    lib.add(_asset(asset_dir, name, **meta))
    cfg = SceneConfig(name=name, objects=[ObjectPlacement(asset_query=name, count=1)])
    return ProgrammaticSceneBackend(seed=42).resolve(cfg, lib)


class TestCollisionAwarePlacement:
    def test_resolved_tabletop_is_valid(self, lib):
        scene = ProgrammaticSceneBackend(seed=42).resolve(tabletop_simple, lib)
        objs = scene.placed_objects
        assert len(objs) >= 3
        assert scene.table_asset is not None

        (x_lo, y_lo), (x_hi, y_hi) = tabletop_simple.workspace_xy
        table_surface = tabletop_simple.table_height + tabletop_simple.table_size[2] / 2.0
        for po in objs:
            assert x_lo <= po.position[0] <= x_hi and y_lo <= po.position[1] <= y_hi
            assert po.position[2] >= table_surface
            assert math.isclose(sum(q * q for q in po.quaternion), 1.0, abs_tol=0.02)
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                dx = objs[i].position[0] - objs[j].position[0]
                dy = objs[i].position[1] - objs[j].position[1]
                assert math.hypot(dx, dy) > 0.001

    def test_seed_controls_placement(self, lib):
        def positions(seed):
            scene = ProgrammaticSceneBackend(seed=seed).resolve(tabletop_simple, lib)
            return [(p.position, p.quaternion) for p in scene.placed_objects]

        assert positions(123) == positions(123)
        assert positions(1) != positions(2)

    def test_many_objects_no_crash(self, lib):
        """Many objects should either place or skip gracefully."""
        cfg = SceneConfig(
            name="crowded",
            objects=[ObjectPlacement(asset_query="die_01", count=20)],
            workspace_xy=[[0.4, -0.1], [0.6, 0.1]],
        )
        scene = ProgrammaticSceneBackend(seed=7).resolve(cfg, lib)
        assert len(scene.placed_objects) >= 1

    def test_mesh_issue_asset_is_skipped(self, lib, tmp_path):
        scene = _resolve_single(
            lib, tmp_path, "bad_mug",
            size_cm=[8.0, 8.0, 10.0], mesh={"issues": ["visual_collision_mismatch"]},
        )
        assert scene.placed_objects == []

    def test_unverified_upright_pose_asset_is_skipped(self, lib, tmp_path):
        scene = _resolve_single(
            lib, tmp_path, "candidate_mug",
            source="objaverse", size_cm=[8.0, 8.0, 10.0], mesh={"issues": []},
            task_poses={"upright": {"quat_object": [1.0, 0.0, 0.0, 0.0], "verified": False}},
        )
        assert scene.placed_objects == []


class TestUprightPoseResolution:
    def test_task_pose_verified_requires_explicit_true(self):
        asset = _asset(Path("."), task_poses={
            "upright": {"quat_object": [1, 0, 0, 0], "verified": False},
        })
        assert not task_pose_verified(asset, "upright")

    def test_task_pose_quat_reads_metadata_aliases(self):
        asset = _asset(Path("."), task_poses={
            "upright": {"quat": [0.0, 0.0, 1.0, 0.0], "verified": True},
        })
        assert task_pose_quat(asset, "upright") == [0.0, 0.0, 1.0, 0.0]
        assert task_pose_verified(asset, "upright")
        assert task_upright_quat(asset).tolist() == [0.0, 0.0, 1.0, 0.0]

    def test_resolve_upright_pose_uses_metric_scale(self):
        asset = _asset(Path("."), size_cm=[8.0, 6.0, 10.0], metric_scale=0.8)
        z, quat = _resolve_upright_pose(asset, asset.metadata.metric_scale)
        assert z == pytest.approx(0.04)
        assert quat == [1.0, 0.0, 0.0, 0.0]

    def test_resolve_upright_pose_uses_task_pose_metadata(self):
        asset = _asset(Path("."), size_cm=[8.0, 6.0, 10.0],
                       task_poses={"upright": {"quat_object": [0.0, 1.0, 0.0, 0.0]}})
        z, quat = _resolve_upright_pose(asset, 1.0)
        assert z == pytest.approx(0.05)
        assert quat == [0.0, 1.0, 0.0, 0.0]
