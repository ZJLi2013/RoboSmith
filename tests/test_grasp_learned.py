"""Smoke + integration tests for GraspGen integration modules.

Tests transforms and LearnedGraspPlanner using real assets (bowls with
meshes) and synthetic data. Also covers bowl mesh → point cloud sanity,
stable-pose world-frame transforms, and grasp Z-range sanity.
No GPU / GraspGen model required — uses a mock model for planner tests.
"""

from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from robotsmith.assets.schema import Asset, AssetMetadata
from robotsmith.grasp.transforms import (
    quat_wxyz_to_matrix,
    rotmat_to_quat_wxyz,
    pose_matrix,
    transform_points,
)
from robotsmith.grasp.learned_planner import LearnedGraspPlanner
from robotsmith.grasp.planner import GraspPlan

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


# ---- learned_planner.py with mock model ----

BOWL_MESH = ASSETS_ROOT / "objects" / "bowl_02" / "visual.obj"
BOWL_02_DIR = BOWL_MESH.parent
TABLE_Z = 0.02  # typical table surface Z in sim


def _sample_mesh(mesh_path, n_points=4096, scale=1.0):
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if scale != 1.0:
        mesh.apply_scale(scale)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return np.asarray(pts, dtype=np.float32)


def _sample_asset_mesh(asset, n_points=4096, object_pos=None, object_quat=None):
    mesh_path = asset.visual_mesh or asset.collision_mesh
    pc = _sample_mesh(mesh_path, n_points)
    if object_pos is not None:
        T = pose_matrix(np.asarray(object_pos), object_quat)
        pc = transform_points(pc, T)
    return pc


def _load_bowl_asset(bowl_dir=BOWL_02_DIR):
    meta = AssetMetadata.load(bowl_dir / "metadata.json")
    return Asset(
        name=bowl_dir.name,
        root_dir=bowl_dir,
        urdf_path=bowl_dir / "model.urdf",
        metadata=meta,
        visual_mesh=bowl_dir / "visual.obj",
        collision_mesh=bowl_dir / "collision.obj",
    )


class RealisticBowlMockModel:
    """GraspGen-like grasps around a bowl rim (approach +Z, fingers +X)."""

    def __init__(self, n_grasps: int = 20):
        self._n = n_grasps

    def predict(self, point_cloud: np.ndarray):
        return self._generate_grasps(point_cloud)

    def predict_from_mesh(self, mesh_path, scale=1.0, num_sample_points=2000):
        mesh = trimesh.load(str(mesh_path), force="mesh")
        mesh.apply_scale(scale)
        xyz, _ = trimesh.sample.sample_surface(mesh, num_sample_points)
        return self._generate_grasps(np.asarray(xyz, dtype=np.float32))

    def _generate_grasps(self, point_cloud: np.ndarray):
        pc_center = point_cloud.mean(axis=0)
        pc_extent = point_cloud.max(axis=0) - point_cloud.min(axis=0)
        radius = max(pc_extent[0], pc_extent[1]) / 2.0
        height = pc_extent[2]

        poses = np.zeros((self._n, 4, 4), dtype=np.float32)
        scores = np.zeros(self._n, dtype=np.float32)

        for i in range(self._n):
            angle = 2 * np.pi * i / self._n
            x = pc_center[0] + radius * 0.7 * np.cos(angle)
            y = pc_center[1] + radius * 0.7 * np.sin(angle)
            z = pc_center[2] + height * 0.3

            R = Rotation.from_euler("x", 180, degrees=True).as_matrix()
            if i % 3 == 1:
                R = R @ Rotation.from_euler("y", 15, degrees=True).as_matrix()
                scores[i] = 0.85 - 0.01 * i
            elif i % 3 == 2:
                R = R @ Rotation.from_euler("y", -15, degrees=True).as_matrix()
                scores[i] = 0.80 - 0.01 * i
            else:
                scores[i] = 0.95 - 0.01 * i

            poses[i, :3, :3] = R.astype(np.float32)
            poses[i, :3, 3] = [x, y, z]
            poses[i, 3, :] = [0, 0, 0, 1]

        poses[-1, 2, 3] = pc_center[2] - height
        scores[-1] = 0.99
        poses[-2, 2, 3] = pc_center[2] + height * 2
        scores[-2] = 0.50

        sort_idx = np.argsort(-scores)
        return poses[sort_idx], scores[sort_idx]


class MockGraspGenModel:
    """Returns synthetic top-down grasps for testing the planner pipeline.

    Uses 180-deg X rotation so approach direction = [0,0,-1] (downward),
    matching GraspGen convention for top-down grasps.
    """

    def __init__(self, n_grasps: int = 10):
        self._n = n_grasps

    def predict(self, point_cloud: np.ndarray):
        poses = np.zeros((self._n, 4, 4), dtype=np.float32)
        scores = np.linspace(0.9, 0.5, self._n).astype(np.float32)
        R_top_down = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
        for i in range(self._n):
            poses[i, :3, :3] = R_top_down
            poses[i, 2, 3] = 0.02 * (i + 1)
            poses[i, 3, 3] = 1.0
        return poses, scores

    def predict_from_mesh(self, mesh_path, scale=1.0, num_sample_points=2000):
        dummy_pc = np.random.randn(num_sample_points, 3).astype(np.float32) * 0.05
        return self.predict(dummy_pc)

    def predict_from_points(self, point_cloud: np.ndarray):
        return self.predict(point_cloud)


def _need_bowl_mesh():
    if not BOWL_MESH.exists():
        pytest.skip("bowl_02/visual.obj not found")


class TestLearnedGraspPlanner:
    def _make_planner(self, **kwargs):
        model = MockGraspGenModel(n_grasps=10)
        kwargs.setdefault("z_offset", 0.0)
        return LearnedGraspPlanner(model, **kwargs)

    def _make_asset(self):
        _need_bowl_mesh()
        return Asset(
            name="test_obj",
            root_dir=BOWL_MESH.parent,
            urdf_path=BOWL_MESH.parent / "model.urdf",
            visual_mesh=BOWL_MESH,
            metadata=AssetMetadata(size_cm=[14, 14, 7]),
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
            place_point_world=np.array([0.5, 0.1, 0.85]),
        )
        assert isinstance(plan, GraspPlan)
        assert plan.metadata["source"] == "learned_place"
        assert plan.grasp_pos[0] == pytest.approx(0.5)
        assert plan.grasp_pos[1] == pytest.approx(0.1)
        # Flange goes to the resolved world drop point (no table-relative recompute).
        assert plan.grasp_pos[2] == pytest.approx(0.85)

    def test_plan_approach_and_retreat_clearance(self):
        planner = self._make_planner(
            approach_clearance=0.15,
            retreat_clearance=0.20,
        )
        plans = planner.plan(
            object_pos=np.array([0.5, 0.0, 0.02]),
            asset=self._make_asset(),
        )
        plan = plans[0]
        approach = np.array(plan.metadata["approach_dir"])
        expected_pre = plan.grasp_pos - approach * 0.15
        np.testing.assert_allclose(plan.pre_grasp_pos, expected_pre, atol=1e-6)
        assert plan.retreat_pos[2] > plan.grasp_pos[2]
        np.testing.assert_allclose(
            plan.retreat_pos[2] - plan.grasp_pos[2], 0.20, atol=1e-6)

    def test_plan_has_waypoints(self):
        planner = self._make_planner()
        plans = planner.plan(
            object_pos=np.array([0.5, 0.0, 0.02]),
            asset=self._make_asset(),
        )
        plan = plans[0]
        assert plan.waypoints is not None
        assert len(plan.waypoints) == 4
        assert plan.waypoints[0].finger_width == plan.finger_open
        assert plan.waypoints[-1].finger_width == plan.finger_closed


# ---- bowl mesh → point cloud sanity ----

class TestBowlPointCloud:
    def test_shape_and_dtype(self):
        _need_bowl_mesh()
        pc = _sample_mesh(BOWL_MESH, n_points=8192)
        assert pc.shape == (8192, 3)
        assert pc.dtype == np.float32

    def test_extent_matches_metadata(self):
        _need_bowl_mesh()
        meta = AssetMetadata.load(BOWL_02_DIR / "metadata.json")
        pc = _sample_mesh(BOWL_MESH, n_points=8192)
        extent = pc.max(axis=0) - pc.min(axis=0)
        expected_m = np.array(meta.size_cm) / 100.0
        np.testing.assert_allclose(extent, expected_m, rtol=0.35, atol=0.01)

    def test_centered_near_origin(self):
        _need_bowl_mesh()
        pc = _sample_mesh(BOWL_MESH, n_points=8192)
        assert np.linalg.norm(pc.mean(axis=0)) < 0.15

    def test_no_nan_or_inf(self):
        _need_bowl_mesh()
        pc = _sample_mesh(BOWL_MESH, n_points=4096)
        assert np.all(np.isfinite(pc))


# ---- point cloud under stable-pose rotation → world frame ----

class TestWorldFrameTransform:
    def test_stable_pose_rotation_valid(self):
        _need_bowl_mesh()
        asset = _load_bowl_asset()
        quat = np.array(asset.metadata.stable_poses[0]["quat"], dtype=np.float32)
        pc_local = _sample_asset_mesh(asset, n_points=4096)
        pc_world = _sample_asset_mesh(
            asset, n_points=4096,
            object_pos=np.array([0.5, 0.0, 0.1]), object_quat=quat,
        )
        assert pc_local.max(axis=0)[2] - pc_local.min(axis=0)[2] > 0
        assert pc_world.max(axis=0)[2] - pc_world.min(axis=0)[2] > 0

    def test_world_frame_center_at_object_pos(self):
        _need_bowl_mesh()
        asset = _load_bowl_asset()
        obj_pos = np.array([0.55, -0.1, 0.12])
        pc = _sample_asset_mesh(asset, n_points=4096, object_pos=obj_pos)
        np.testing.assert_allclose(pc.mean(axis=0), obj_pos, atol=0.08)

    def test_all_stable_poses_produce_valid_pc(self):
        _need_bowl_mesh()
        asset = _load_bowl_asset()
        for i, sp in enumerate(asset.metadata.stable_poses[:5]):
            quat = np.array(sp["quat"], dtype=np.float32)
            pc = _sample_asset_mesh(
                asset, n_points=1024,
                object_pos=np.array([0.5, 0.0, 0.1]), object_quat=quat,
            )
            assert np.all(np.isfinite(pc)), f"stable_pose[{i}] has NaN/Inf"
            extent = pc.max(axis=0) - pc.min(axis=0)
            assert extent.min() > 0.005, f"stable_pose[{i}] degenerate axis"

