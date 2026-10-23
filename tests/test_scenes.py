"""Tests for the collision-aware scene backend and upright pose resolution."""

import math
import shutil
from pathlib import Path

import pytest

from robotsmith.assets.library import AssetLibrary
from robotsmith.assets.schema import Asset, AssetMetadata
from robotsmith.scenes.config import SceneConfig, ObjectPlacement
from robotsmith.scenes.backend import ProgrammaticSceneBackend

tabletop_simple = SceneConfig(
    name="tabletop_simple",
    description="Simple tabletop with several blocks",
    objects=[ObjectPlacement(asset_query="die_01", count=5)],
    table_size=[1.2, 0.8, 0.05],
    table_height=0.75,
    workspace_xy=[[0.35, -0.20], [0.65, 0.20]],
)

REPO_ASSETS = Path(__file__).resolve().parents[1] / "assets"


@pytest.fixture
def lib(tmp_path: Path) -> AssetLibrary:
    """Writable library seeded with the committed support assets."""
    for name in ("die_01",):
        shutil.copytree(REPO_ASSETS / name, tmp_path / "assets" / name)
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
            upright_verified=False,
        )
        assert scene.placed_objects == []

    def test_asset_stands_at_identity_with_scene_yaw(self, lib):
        cfg = SceneConfig(name="yaw", objects=[
            ObjectPlacement(asset_query="die_01", count=1, yaw_deg=90.0),
        ])
        po = ProgrammaticSceneBackend(seed=0).resolve(cfg, lib).placed_objects[0]
        table_surface = cfg.table_height + cfg.table_size[2] / 2.0
        assert po.position[2] == pytest.approx(table_surface + po.asset.metadata.size_cm[2] / 200.0)
        assert po.quaternion == pytest.approx([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
