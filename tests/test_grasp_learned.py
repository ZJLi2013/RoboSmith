"""Transforms and LearnedGraspPlanner with a mock GraspGen model (no GPU)."""

import numpy as np
import pytest
import trimesh

from robotsmith.assets.schema import Asset, AssetMetadata
from robotsmith.grasp.transforms import (
    quat_wxyz_to_matrix,
    rotmat_to_quat_wxyz,
    pose_matrix,
    transform_points,
)
from robotsmith.grasp.learned_planner import LearnedGraspPlanner
from robotsmith.grasp.planner import GraspPlan


@pytest.mark.parametrize("q", [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.5, 0.5, 0.5, 0.5],
])
def test_quat_matrix_roundtrip(q):
    q_in = np.array(q, dtype=np.float32)
    q_out = rotmat_to_quat_wxyz(quat_wxyz_to_matrix(q_in))
    if np.dot(q_in, q_out) < 0:
        q_out = -q_out
    np.testing.assert_allclose(q_out, q_in, atol=1e-5)


def test_180_z_rotation():
    R = quat_wxyz_to_matrix(np.array([0, 0, 0, 1]))
    np.testing.assert_allclose(R, np.diag([-1.0, -1.0, 1.0]), atol=1e-7)


def test_pose_matrix_with_rotation():
    T = pose_matrix(np.array([1, 2, 3]), np.array([0, 0, 0, 1], dtype=np.float32))
    np.testing.assert_allclose(T[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(T[:3, :3], np.diag([-1, -1, 1]), atol=1e-7)
    np.testing.assert_allclose(T[3], [0, 0, 0, 1])


def test_transform_points_translation():
    out = transform_points(np.zeros((1, 3), dtype=np.float32), pose_matrix(np.array([10, 20, 30])))
    np.testing.assert_allclose(out, [[10, 20, 30]], atol=1e-5)


class MockGraspGenModel:
    """Synthetic top-down grasps (approach -Z) with descending scores."""

    def __init__(self, n_grasps: int = 10):
        self._n = n_grasps

    def predict_from_mesh(self, mesh_path, scale=1.0, num_sample_points=2000, **kwargs):
        poses = np.zeros((self._n, 4, 4), dtype=np.float32)
        scores = np.linspace(0.9, 0.5, self._n).astype(np.float32)
        for i in range(self._n):
            poses[i, :3, :3] = np.diag([1.0, -1.0, -1.0])
            poses[i, 2, 3] = 0.02 * (i + 1)
            poses[i, 3, 3] = 1.0
        return poses, scores


OBJ_POS = np.array([0.5, 0.0, 0.02])


@pytest.fixture
def asset(tmp_path):
    mesh = tmp_path / "visual.obj"
    trimesh.creation.box(extents=[0.14, 0.14, 0.07]).export(str(mesh))
    return Asset(
        name="test_obj",
        root_dir=tmp_path,
        urdf_path=tmp_path / "model.urdf",
        visual_mesh=mesh,
        metadata=AssetMetadata(size_cm=[14, 14, 7]),
    )


def _planner(**kwargs):
    kwargs.setdefault("z_offset", 0.0)
    return LearnedGraspPlanner(MockGraspGenModel(), **kwargs)


def test_plan_returns_ordered_grasp_plans_with_waypoints(asset):
    plans = _planner(top_k=3).plan(object_pos=OBJ_POS, asset=asset, category="bowl")

    assert 1 <= len(plans) <= 3
    assert all(isinstance(p, GraspPlan) for p in plans)
    assert [p.quality for p in plans] == sorted((p.quality for p in plans), reverse=True)
    plan = plans[0]
    assert plan.grasp_pos.shape == (3,) and plan.grasp_quat.shape == (4,)
    assert plan.metadata["source"] == "graspgen"
    assert plan.metadata["category"] == "bowl"
    assert len(plan.waypoints) == 4
    assert plan.waypoints[0].finger_width == plan.finger_open
    assert plan.waypoints[-1].finger_width == plan.finger_closed


def test_plan_height_filter(asset):
    planner = _planner(z_offset=0.5, min_grasp_z_margin=0.05, top_k=10)
    plans = planner.plan(object_pos=np.array([0.5, 0.0, 0.5]), asset=asset)
    assert all(plan.grasp_pos[2] >= 0.55 for plan in plans)


def test_plan_without_asset_returns_empty():
    assert _planner().plan(object_pos=OBJ_POS, asset=None) == []


def test_plan_place():
    plan = _planner(z_offset=0.02).plan_place(
        np.array([0.5, 0.1, 0.0]),
        category="bowl",
        place_point_world=np.array([0.5, 0.1, 0.85]),
    )
    assert plan.metadata["source"] == "learned_place"
    # Flange goes to the resolved world drop point (no table-relative recompute).
    np.testing.assert_allclose(plan.grasp_pos, [0.5, 0.1, 0.85])


def test_plan_approach_and_retreat_clearance(asset):
    planner = _planner(approach_clearance=0.15, retreat_clearance=0.20)
    plan = planner.plan(object_pos=OBJ_POS, asset=asset)[0]
    approach = np.array(plan.metadata["approach_dir"])
    np.testing.assert_allclose(plan.pre_grasp_pos, plan.grasp_pos - approach * 0.15, atol=1e-6)
    np.testing.assert_allclose(plan.retreat_pos[2] - plan.grasp_pos[2], 0.20, atol=1e-6)
