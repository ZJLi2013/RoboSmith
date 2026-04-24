"""Smoke tests for GraspGen integration modules.

Tests pointcloud_utils, transforms, and LearnedGraspPlanner
using real assets (bowls with meshes) and synthetic data.
No GPU / GraspGen model required — uses a mock model for planner tests.
"""

from pathlib import Path

import numpy as np
import pytest

from robotsmith.assets.schema import Asset, AssetMetadata
from robotsmith.grasp.transforms import (
    quat_wxyz_to_matrix,
    rotmat_to_quat_wxyz,
    pose_matrix,
    transform_points,
)
from robotsmith.grasp.pointcloud_utils import (
    mesh_to_pointcloud,
    asset_to_pointcloud,
)
from robotsmith.grasp.learned_planner import LearnedGraspPlanner
from robotsmith.grasp.plan import GraspPlan

ASSETS_ROOT = Path(__file__).resolve().parent.parent / "assets"


# ---- transforms.py ----

class TestTransforms:
    def test_identity_quat_to_matrix(self):
        R = quat_wxyz_to_matrix(np.array([1, 0, 0, 0]))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-7)

    def test_180_z_rotation(self):
        R = quat_wxyz_to_matrix(np.array([0, 0, 0, 1]))
        expected = np.diag([-1.0, -1.0, 1.0])
        np.testing.assert_allclose(R, expected, atol=1e-7)

    def test_quat_roundtrip(self):
        """quat → matrix → quat should be identity (up to sign)."""
        q_in = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        R = quat_wxyz_to_matrix(q_in)
        q_out = rotmat_to_quat_wxyz(R)
        # quaternions are equivalent up to sign
        if np.dot(q_in, q_out) < 0:
            q_out = -q_out
        np.testing.assert_allclose(q_out, q_in, atol=1e-5)

    def test_pose_matrix_identity(self):
        T = pose_matrix(np.array([0, 0, 0]))
        np.testing.assert_allclose(T, np.eye(4), atol=1e-7)

    def test_pose_matrix_translation(self):
        T = pose_matrix(np.array([1.0, 2.0, 3.0]))
        assert T[0, 3] == 1.0
        assert T[1, 3] == 2.0
        assert T[2, 3] == 3.0
        np.testing.assert_allclose(T[:3, :3], np.eye(3))

    def test_pose_matrix_with_rotation(self):
        q = np.array([0, 0, 0, 1], dtype=np.float32)  # 180 deg Z
        T = pose_matrix(np.array([1, 2, 3]), q)
        np.testing.assert_allclose(T[:3, 3], [1, 2, 3])
        np.testing.assert_allclose(T[:3, :3], np.diag([-1, -1, 1]), atol=1e-7)

    def test_transform_points_identity(self):
        pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        out = transform_points(pts, np.eye(4))
        np.testing.assert_allclose(out, pts, atol=1e-6)

    def test_transform_points_translation(self):
        pts = np.array([[0, 0, 0]], dtype=np.float32)
        T = pose_matrix(np.array([10, 20, 30]))
        out = transform_points(pts, T)
        np.testing.assert_allclose(out, [[10, 20, 30]], atol=1e-5)

    def test_rotmat_to_quat_all_branches(self):
        """Test rotmat_to_quat_wxyz with rotations that hit different code branches."""
        for axis_idx in range(3):
            q = np.array([1, 0, 0, 0], dtype=np.float32)
            q[0] = 0.0
            q[axis_idx + 1] = 1.0
            R = quat_wxyz_to_matrix(q)
            q_out = rotmat_to_quat_wxyz(R)
            if np.dot(q, q_out) < 0:
                q_out = -q_out
            np.testing.assert_allclose(q_out, q, atol=1e-5)


# ---- pointcloud_utils.py ----

class TestPointCloudUtils:
    def test_mesh_to_pointcloud_bowl(self):
        mesh_path = ASSETS_ROOT / "objects" / "bowl_01" / "visual.obj"
        if not mesh_path.exists():
            pytest.skip("bowl_01 visual.obj not found")
        pc = mesh_to_pointcloud(mesh_path, n_points=1024)
        assert pc.shape == (1024, 3)
        assert pc.dtype == np.float32

    def test_mesh_to_pointcloud_scaled(self):
        mesh_path = ASSETS_ROOT / "objects" / "bowl_02" / "visual.obj"
        if not mesh_path.exists():
            pytest.skip("bowl_02 visual.obj not found")
        pc_full = mesh_to_pointcloud(mesh_path, n_points=2048, scale=1.0)
        pc_half = mesh_to_pointcloud(mesh_path, n_points=2048, scale=0.5)
        # Scaled point cloud should have smaller extent
        extent_full = pc_full.max(axis=0) - pc_full.min(axis=0)
        extent_half = pc_half.max(axis=0) - pc_half.min(axis=0)
        np.testing.assert_allclose(extent_half, extent_full * 0.5, atol=0.01)

    def test_asset_to_pointcloud_with_mesh(self):
        """Real asset with visual.obj mesh."""
        mesh_path = ASSETS_ROOT / "objects" / "bowl_01" / "visual.obj"
        if not mesh_path.exists():
            pytest.skip("bowl_01 not found")
        asset = Asset(
            name="bowl_01",
            root_dir=ASSETS_ROOT / "objects" / "bowl_01",
            urdf_path=ASSETS_ROOT / "objects" / "bowl_01" / "model.urdf",
            metadata=AssetMetadata(tags=["bowl"]),
            visual_mesh=mesh_path,
        )
        pc = asset_to_pointcloud(asset, n_points=512)
        assert pc.shape == (512, 3)

    def test_asset_to_pointcloud_box_fallback(self):
        """Asset without mesh falls back to box from size_cm."""
        asset = Asset(
            name="block_red",
            root_dir=Path("/tmp"),
            urdf_path=Path("/tmp/model.urdf"),
            metadata=AssetMetadata(size_cm=[4.0, 4.0, 4.0]),
        )
        pc = asset_to_pointcloud(asset, n_points=256)
        assert pc.shape == (256, 3)
        extent = pc.max(axis=0) - pc.min(axis=0)
        np.testing.assert_allclose(extent, [0.04, 0.04, 0.04], atol=0.005)

    def test_asset_to_pointcloud_world_frame(self):
        """Points should shift when object_pos is provided."""
        asset = Asset(
            name="test",
            root_dir=Path("/tmp"),
            urdf_path=Path("/tmp/model.urdf"),
            metadata=AssetMetadata(size_cm=[2, 2, 2]),
        )
        pc_local = asset_to_pointcloud(asset, n_points=100)
        pc_world = asset_to_pointcloud(
            asset, n_points=100,
            object_pos=np.array([1.0, 2.0, 3.0]),
        )
        # Center should shift by approximately (1, 2, 3)
        shift = pc_world.mean(axis=0) - pc_local.mean(axis=0)
        np.testing.assert_allclose(shift, [1, 2, 3], atol=0.01)


# ---- learned_planner.py with mock model ----

class MockGraspGenModel:
    """Returns synthetic grasps for testing the planner pipeline."""

    def __init__(self, n_grasps: int = 10):
        self._n = n_grasps

    def predict(self, point_cloud: np.ndarray):
        poses = np.zeros((self._n, 4, 4), dtype=np.float32)
        scores = np.linspace(0.9, 0.5, self._n).astype(np.float32)
        for i in range(self._n):
            poses[i] = np.eye(4)
            # Grasps spread in a line above origin
            poses[i, 0, 3] = 0.0
            poses[i, 1, 3] = 0.0
            poses[i, 2, 3] = 0.02 * (i + 1)
        return poses, scores


class TestLearnedGraspPlanner:
    def _make_planner(self, **kwargs):
        model = MockGraspGenModel(n_grasps=10)
        kwargs.setdefault("z_offset", 0.0)
        return LearnedGraspPlanner(model, **kwargs)

    def _make_asset(self):
        return Asset(
            name="test_obj",
            root_dir=Path("/tmp"),
            urdf_path=Path("/tmp/model.urdf"),
            metadata=AssetMetadata(size_cm=[4, 4, 4], tags=["block"]),
        )

    def test_plan_returns_grasp_plans(self):
        planner = self._make_planner()
        plans = planner.plan(
            object_pos=np.array([0.5, 0.0, 0.02]),
            asset=self._make_asset(),
        )
        assert len(plans) >= 1
        plan = plans[0]
        assert isinstance(plan, GraspPlan)
        assert plan.grasp_pos.shape == (3,)
        assert plan.grasp_quat.shape == (4,)

    def test_plan_respects_top_k(self):
        planner = self._make_planner(top_k=3)
        plans = planner.plan(
            object_pos=np.array([0.5, 0.0, 0.02]),
            asset=self._make_asset(),
        )
        assert len(plans) <= 3

    def test_plan_height_filter(self):
        """Grasps below table should be filtered."""
        planner = self._make_planner(
            z_offset=0.5, min_grasp_z_margin=0.05, top_k=10,
        )
        plans = planner.plan(
            object_pos=np.array([0.5, 0.0, 0.5]),
            asset=self._make_asset(),
        )
        for plan in plans:
            assert plan.grasp_pos[2] >= 0.55

    def test_plan_without_asset_returns_empty(self):
        planner = self._make_planner()
        plans = planner.plan(
            object_pos=np.array([0.5, 0.0, 0.02]),
            asset=None,
        )
        assert plans == []

    def test_plan_quality_ordering(self):
        """Plans should be ordered by score (descending)."""
        planner = self._make_planner(top_k=5)
        plans = planner.plan(
            object_pos=np.array([0.5, 0.0, 0.02]),
            asset=self._make_asset(),
        )
        if len(plans) > 1:
            for i in range(len(plans) - 1):
                assert plans[i].quality >= plans[i + 1].quality

    def test_plan_metadata(self):
        planner = self._make_planner()
        plans = planner.plan(
            object_pos=np.array([0.5, 0.0, 0.02]),
            asset=self._make_asset(),
            category="bowl",
        )
        assert plans[0].metadata["source"] == "graspgen"
        assert plans[0].metadata["category"] == "bowl"

    def test_plan_place(self):
        planner = self._make_planner(z_offset=0.02)
        plan = planner.plan_place(
            np.array([0.5, 0.1, 0.0]),
            category="bowl",
        )
        assert isinstance(plan, GraspPlan)
        assert plan.metadata["source"] == "learned_place"
        assert plan.grasp_pos[0] == pytest.approx(0.5)
        assert plan.grasp_pos[1] == pytest.approx(0.1)

    def test_plan_hover_and_retreat_above_grasp(self):
        planner = self._make_planner(
            hover_clearance=0.15,
            retreat_clearance=0.20,
        )
        plans = planner.plan(
            object_pos=np.array([0.5, 0.0, 0.02]),
            asset=self._make_asset(),
        )
        plan = plans[0]
        assert plan.pre_grasp_pos[2] > plan.grasp_pos[2]
        assert plan.retreat_pos[2] > plan.grasp_pos[2]
        np.testing.assert_allclose(
            plan.pre_grasp_pos[2] - plan.grasp_pos[2], 0.15, atol=1e-6)
        np.testing.assert_allclose(
            plan.retreat_pos[2] - plan.grasp_pos[2], 0.20, atol=1e-6)
