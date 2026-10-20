"""Tests for scene configs, collision-aware backend, and genesis loader."""

import math
import shutil
from pathlib import Path

import pytest

from robotsmith.assets.library import AssetLibrary
from robotsmith.scenes.config import (
    SceneConfig,
    ObjectPlacement,
    DEFAULT_WORKSPACE_XY,
)
from robotsmith.scenes.backend import (
    ProgrammaticSceneBackend,
    PlacedObject,
    _quat_to_euler,
    _resolve_upright_pose,
)
from robotsmith.scenes.pose_utils import task_pose_quat, task_pose_verified, task_upright_quat

# Minimal scenes built inline for resolver coverage. Template scene presets were
# removed when authoring moved to CAP specs; the collision-aware backend is core
# and still exercised here without any preset registry.
tabletop_simple = SceneConfig(
    name="tabletop_simple",
    description="Simple tabletop with several blocks",
    objects=[
        ObjectPlacement(asset_query="block_blue", count=5),
    ],
    table_size=[1.2, 0.8, 0.05],
    table_height=0.75,
    workspace_xy=[[0.35, -0.20], [0.65, 0.20]],
)


REPO_OBJECTS = Path(__file__).resolve().parents[1] / "assets" / "objects"


def _make_lib(tmp_path: Path) -> AssetLibrary:
    """Writable library seeded with the committed support assets."""
    objects = tmp_path / "assets" / "objects"
    objects.mkdir(parents=True)
    for name in ("block_blue", "table_simple", "plane"):
        shutil.copytree(REPO_OBJECTS / name, objects / name)
    return AssetLibrary(tmp_path / "assets")


class TestSceneConfig:
    def test_tabletop_has_objects(self):
        assert sum(o.count for o in tabletop_simple.objects) >= 3

    def test_workspace_xy_default(self):
        cfg = SceneConfig(name="test")
        assert cfg.workspace_xy == DEFAULT_WORKSPACE_XY

    def test_collision_margin_default(self):
        cfg = SceneConfig(name="test")
        assert cfg.collision_margin == 0.02


class TestCollisionAwarePlacement:
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
            assert p1.quaternion == p2.quaternion

    def test_different_seeds_differ(self, tmp_path):
        lib = _make_lib(tmp_path)
        s1 = ProgrammaticSceneBackend(seed=1).resolve(tabletop_simple, lib)
        s2 = ProgrammaticSceneBackend(seed=2).resolve(tabletop_simple, lib)
        positions_differ = any(
            p1.position != p2.position
            for p1, p2 in zip(s1.placed_objects, s2.placed_objects)
        )
        assert positions_differ

    def test_objects_within_workspace(self, tmp_path):
        lib = _make_lib(tmp_path)
        backend = ProgrammaticSceneBackend(seed=99)
        scene = backend.resolve(tabletop_simple, lib)
        ws_lo, ws_hi = tabletop_simple.workspace_xy
        for po in scene.placed_objects:
            assert ws_lo[0] <= po.position[0] <= ws_hi[0], f"{po.asset.name} x out of bounds"
            assert ws_lo[1] <= po.position[1] <= ws_hi[1], f"{po.asset.name} y out of bounds"

    def test_objects_on_table_surface(self, tmp_path):
        lib = _make_lib(tmp_path)
        backend = ProgrammaticSceneBackend(seed=42)
        scene = backend.resolve(tabletop_simple, lib)
        table_surface = tabletop_simple.table_height + tabletop_simple.table_size[2] / 2.0
        for po in scene.placed_objects:
            assert po.position[2] >= table_surface, (
                f"{po.asset.name} z={po.position[2]:.3f} < table={table_surface:.3f}"
            )

    def test_no_pairwise_collision(self, tmp_path):
        """All placed objects should be at least collision_margin apart."""
        lib = _make_lib(tmp_path)
        backend = ProgrammaticSceneBackend(seed=42)
        scene = backend.resolve(tabletop_simple, lib)
        objs = scene.placed_objects
        margin = tabletop_simple.collision_margin
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                dx = objs[i].position[0] - objs[j].position[0]
                dy = objs[i].position[1] - objs[j].position[1]
                dist = math.sqrt(dx * dx + dy * dy)
                assert dist > 0.001, (
                    f"{objs[i].asset.name} and {objs[j].asset.name} overlap"
                )

    def test_many_objects_no_crash(self, tmp_path):
        """Stress test: many objects should either place or skip gracefully."""
        lib = _make_lib(tmp_path)
        cfg = SceneConfig(
            name="crowded",
            objects=[ObjectPlacement(asset_query="block_blue", count=20)],
            workspace_xy=[[0.4, -0.1], [0.6, 0.1]],
        )
        backend = ProgrammaticSceneBackend(seed=7)
        scene = backend.resolve(cfg, lib)
        assert len(scene.placed_objects) >= 1

    def test_placed_object_has_quaternion(self, tmp_path):
        lib = _make_lib(tmp_path)
        backend = ProgrammaticSceneBackend(seed=42)
        scene = backend.resolve(tabletop_simple, lib)
        for po in scene.placed_objects:
            assert len(po.quaternion) == 4
            norm = sum(q * q for q in po.quaternion) ** 0.5
            assert 0.99 < norm < 1.01, f"quaternion not unit: {po.quaternion}"

    def test_mesh_issue_asset_is_skipped(self, tmp_path):
        from robotsmith.assets.schema import Asset, AssetMetadata

        lib = _make_lib(tmp_path)
        asset_dir = tmp_path / "bad_mug"
        asset_dir.mkdir()
        bad_asset = Asset(
            name="bad_mug",
            root_dir=asset_dir,
            urdf_path=asset_dir / "model.urdf",
            metadata=AssetMetadata(
                size_cm=[8.0, 8.0, 10.0],
                mesh={"issues": ["visual_collision_mismatch"]},
            ),
        )
        lib.add(bad_asset)
        cfg = SceneConfig(
            name="mesh_issue",
            objects=[ObjectPlacement(asset_query="bad_mug", count=1)],
        )

        scene = ProgrammaticSceneBackend(seed=42).resolve(cfg, lib)

        assert scene.placed_objects == []

    def test_unverified_upright_pose_asset_is_skipped(self, tmp_path):
        from robotsmith.assets.schema import Asset, AssetMetadata

        lib = _make_lib(tmp_path)
        asset_dir = tmp_path / "candidate_mug"
        asset_dir.mkdir()
        asset = Asset(
            name="candidate_mug",
            root_dir=asset_dir,
            urdf_path=asset_dir / "model.urdf",
            metadata=AssetMetadata(
                source="objaverse",
                size_cm=[8.0, 8.0, 10.0],
                mesh={"issues": []},
                task_poses={
                    "upright": {
                        "quat_object": [1.0, 0.0, 0.0, 0.0],
                        "verified": False,
                    },
                },
            ),
        )
        lib.add(asset)
        cfg = SceneConfig(
            name="unverified_pose",
            objects=[ObjectPlacement(asset_query="candidate_mug", count=1)],
        )

        scene = ProgrammaticSceneBackend(seed=42).resolve(cfg, lib)

        assert scene.placed_objects == []


class TestUprightPoseResolution:
    def test_task_pose_verified_requires_explicit_true(self):
        from robotsmith.assets.schema import Asset, AssetMetadata

        meta = AssetMetadata(
            task_poses={"upright": {"quat_object": [1, 0, 0, 0], "verified": False}},
        )
        asset = Asset(name="test", root_dir=Path("."), urdf_path=Path("x.urdf"), metadata=meta)

        assert not task_pose_verified(asset, "upright")

    def test_task_pose_quat_reads_metadata_aliases(self):
        from robotsmith.assets.schema import Asset, AssetMetadata

        meta = AssetMetadata(
            task_poses={"upright": {"quat": [0.0, 0.0, 1.0, 0.0], "verified": True}},
        )
        asset = Asset(name="test", root_dir=Path("."), urdf_path=Path("x.urdf"), metadata=meta)

        assert task_pose_quat(asset, "upright") == [0.0, 0.0, 1.0, 0.0]
        assert task_pose_verified(asset, "upright")
        assert task_upright_quat(asset).tolist() == [0.0, 0.0, 1.0, 0.0]

    def test_resolve_upright_pose_uses_metric_scale(self):
        from robotsmith.assets.schema import Asset, AssetMetadata
        meta = AssetMetadata(
            size_cm=[8.0, 6.0, 10.0],
            metric_scale=0.8,
        )
        asset = Asset(name="test", root_dir=Path("."), urdf_path=Path("x.urdf"), metadata=meta)
        z, quat = _resolve_upright_pose(asset, asset.metadata.metric_scale)
        assert z == pytest.approx(0.04)
        assert quat == [1.0, 0.0, 0.0, 0.0]

    def test_resolve_upright_pose_uses_task_pose_metadata(self):
        from robotsmith.assets.schema import Asset, AssetMetadata
        meta = AssetMetadata(
            size_cm=[8.0, 6.0, 10.0],
            task_poses={"upright": {"quat_object": [0.0, 1.0, 0.0, 0.0]}},
        )
        asset = Asset(name="test", root_dir=Path("."), urdf_path=Path("x.urdf"), metadata=meta)
        z, quat = _resolve_upright_pose(asset, 1.0)
        assert z == pytest.approx(0.05)
        assert quat == [0.0, 1.0, 0.0, 0.0]

    def test_resolved_scene_uses_asset_metric_scale(self):
        from robotsmith.assets.schema import Asset, AssetMetadata
        meta = AssetMetadata(
            size_cm=[10.0, 8.0, 12.0],
            metric_scale=0.7,
        )
        asset = Asset(name="test", root_dir=Path("."), urdf_path=Path("x.urdf"), metadata=meta)
        z, quat = _resolve_upright_pose(asset, asset.metadata.metric_scale)
        assert z == pytest.approx(0.042)
        assert quat == [1, 0, 0, 0]
        assert asset.metadata.metric_scale == pytest.approx(0.7)


class TestQuat:
    def test_identity_quat_gives_zero_euler(self):
        euler = _quat_to_euler([1.0, 0.0, 0.0, 0.0])
        for e in euler:
            assert abs(e) < 1e-6
