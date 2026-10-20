import types

import numpy as np

from robotsmith.grasp.feasibility import (
    evaluate_grasp_candidates,
    select_best_grasp_plan,
    select_from_bucket_policy,
)
from robotsmith.grasp.learned_planner import LearnedGraspPlanner
from robotsmith.grasp.planner import GraspPlan


def _local_grasp_pose(approach_z: float, hand_offset_z: float) -> np.ndarray:
    """4x4 object-local grasp: Z axis = approach, origin offset along world z."""
    pose = np.eye(4)
    if approach_z < 0:  # top-down: 180deg about X so gripper Z points down
        pose[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    pose[:3, 3] = [0.0, 0.0, hand_offset_z]
    return pose


def test_learned_planner_floor_is_object_support_not_table(tmp_path):
    """Regression for s10: a grasp BELOW a shelved object must be rejected.

    The object floats at z=1.10 (on a shelf, well above the table at z=0).
    GraspGen offers a clean top-down grasp (hand 0.11 above center) and a
    from-below grasp (hand 0.11 below center). The from-below grasp sits at
    world z=0.99 — above the table+margin (old floor would keep it) but below
    the object's own support surface (1.075), so the support-aware floor must
    reject it.
    """
    mesh = tmp_path / "die.obj"
    mesh.write_text("o die\n")
    asset = types.SimpleNamespace(visual_mesh=str(mesh), collision_mesh=None, name="die")

    top_down = _local_grasp_pose(approach_z=-1.0, hand_offset_z=+0.11)
    from_below = _local_grasp_pose(approach_z=+1.0, hand_offset_z=-0.11)
    model = types.SimpleNamespace(
        predict_from_mesh=lambda *_a, **_k: (
            np.array([top_down, from_below]),
            np.array([0.9, 0.95], dtype=np.float32),
        )
    )
    # z_offset = global table (0.0); the fix must use the object support floor.
    planner = LearnedGraspPlanner(model, z_offset=0.0, min_grasp_z_margin=0.02)

    obj_pos = np.array([0.0, 0.0, 1.10])
    plans = planner.plan(obj_pos, object_height=0.05, asset=asset)

    assert len(plans) == 1, "from-below grasp should be floored out"
    assert plans[0].grasp_pos[2] > 1.08  # the top-down (hand above) survived

    # Control: without an object height the floor falls back to z_offset (table),
    # which keeps BOTH grasps — i.e. this is exactly the old, buggy behavior.
    plans_no_h = planner.plan(obj_pos, object_height=None, asset=asset)
    assert len(plans_no_h) == 2


def test_feasibility_selection_keeps_raw_orientation():
    plan = GraspPlan(
        grasp_pos=np.zeros(3),
        grasp_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        pre_grasp_pos=np.zeros(3),
        pre_grasp_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        retreat_pos=np.array([0.0, 0.0, 0.1]),
        retreat_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        finger_open=0.04,
        finger_closed=0.01,
        metadata={
            "source": "graspgen",
            "finger_dir": [1.0, 0.0, 0.0],
            "approach_dir": [0.0, 0.0, 1.0],
        },
    )

    def solve_ik(pos, quat, finger_width=None, *, init_qpos=None):
        if init_qpos is None:
            init_qpos = np.zeros(9)
        return np.asarray(init_qpos, dtype=np.float64).copy()

    selected = select_best_grasp_plan(
        [plan],
        solve_ik,
        np.zeros(9),
        support_z=-1.0,
        category="mug",
    )

    np.testing.assert_allclose(selected.grasp_quat, plan.grasp_quat)
    assert "yaw_variant_deg" not in selected.metadata


def test_support_band_rejects_bottom_crossing_grasp(monkeypatch):
    monkeypatch.setenv("GRASP_EVAL_SUPPORT_BAND_M", "0.01")

    def make_plan(z: float, quality: float) -> GraspPlan:
        return GraspPlan(
            grasp_pos=np.array([0.0, 0.0, z]),
            grasp_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            pre_grasp_pos=np.array([0.0, 0.0, z + 0.05]),
            pre_grasp_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            retreat_pos=np.array([0.0, 0.0, z + 0.1]),
            retreat_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            finger_open=0.04,
            finger_closed=0.01,
            quality=quality,
            metadata={"source": "graspgen"},
        )

    def solve_ik(pos, quat, finger_width=None, *, init_qpos=None):
        if init_qpos is None:
            init_qpos = np.zeros(9)
        return np.asarray(init_qpos, dtype=np.float64).copy()

    bottom_crossing = make_plan(0.005, 0.99)
    above_support = make_plan(0.03, 0.50)

    selected = select_best_grasp_plan(
        [bottom_crossing, above_support],
        solve_ik,
        np.zeros(9),
        support_z=0.0,
        category="mug",
    )

    assert selected is above_support
    assert selected.metadata["feasibility_support_ok_count"] == 1
    assert selected.metadata["feasibility_support_reject_count"] == 1


def test_bucket_policy_selection_is_explicit():
    def make_plan(
        *,
        quality: float,
        approach_dir: list[float],
        finger_dir: list[float],
    ) -> GraspPlan:
        return GraspPlan(
            grasp_pos=np.array([0.0, 0.0, 0.03]),
            grasp_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            pre_grasp_pos=np.array([0.0, 0.0, 0.08]),
            pre_grasp_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            retreat_pos=np.array([0.0, 0.0, 0.13]),
            retreat_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            finger_open=0.04,
            finger_closed=0.01,
            quality=quality,
            metadata={
                "source": "graspgen",
                "approach_dir": approach_dir,
                "finger_dir": finger_dir,
            },
        )

    top_down_horizontal = make_plan(
        quality=0.4,
        approach_dir=[0.0, 0.0, 1.0],
        finger_dir=[1.0, 0.0, 0.0],
    )
    side_vertical = make_plan(
        quality=0.9,
        approach_dir=[1.0, 0.0, 0.0],
        finger_dir=[0.0, 0.0, 1.0],
    )

    def solve_ik(pos, quat, finger_width=None, *, init_qpos=None):
        if init_qpos is None:
            init_qpos = np.zeros(9)
        return np.asarray(init_qpos, dtype=np.float64).copy()

    evaluation = evaluate_grasp_candidates(
        [top_down_horizontal, side_vertical],
        solve_ik,
        np.zeros(9),
        support_z=-1.0,
    )

    assert evaluation.bucket_hard_ok_counts == {
        "top_down_axis_horizontal": 1,
        "side_axis_vertical": 1,
    }

    selected = select_from_bucket_policy(
        evaluation,
        ["side_axis_vertical", "top_down_axis_horizontal"],
    )

    assert selected is not None
    selected_plan, _, _ = selected
    assert selected_plan is side_vertical
    assert selected_plan.metadata["selected_policy_bucket"] == "side_axis_vertical"


def test_select_best_grasp_plan_accepts_explicit_bucket_priority():
    def make_plan(
        *,
        quality: float,
        approach_dir: list[float],
        finger_dir: list[float],
    ) -> GraspPlan:
        return GraspPlan(
            grasp_pos=np.array([0.0, 0.0, 0.03]),
            grasp_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            pre_grasp_pos=np.array([0.0, 0.0, 0.08]),
            pre_grasp_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            retreat_pos=np.array([0.0, 0.0, 0.13]),
            retreat_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            finger_open=0.04,
            finger_closed=0.01,
            quality=quality,
            metadata={
                "source": "graspgen",
                "approach_dir": approach_dir,
                "finger_dir": finger_dir,
            },
        )

    lower_score_bucket = make_plan(
        quality=0.4,
        approach_dir=[0.0, 0.0, 1.0],
        finger_dir=[1.0, 0.0, 0.0],
    )
    higher_score_bucket = make_plan(
        quality=0.9,
        approach_dir=[1.0, 0.0, 0.0],
        finger_dir=[0.0, 0.0, 1.0],
    )

    def solve_ik(pos, quat, finger_width=None, *, init_qpos=None):
        if init_qpos is None:
            init_qpos = np.zeros(9)
        return np.asarray(init_qpos, dtype=np.float64).copy()

    selected = select_best_grasp_plan(
        [lower_score_bucket, higher_score_bucket],
        solve_ik,
        np.zeros(9),
        support_z=-1.0,
        bucket_priority=["top_down_axis_horizontal"],
    )

    assert selected is lower_score_bucket
    assert selected.metadata["requested_policy_bucket"] == "top_down_axis_horizontal"
    assert selected.metadata["requested_policy_bucket_missing"] is False

