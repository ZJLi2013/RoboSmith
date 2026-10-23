"""Geometric grasp candidates come from the object's oriented box and the gripper alone."""

import numpy as np
import pytest

from robotsmith.grasp.geometric import GeometricGraspPlanner, GripperSpec
from robotsmith.grasp.transforms import quat_wxyz_to_matrix


def _box(size, centre=(0.42, -0.27, 0.0), yaw=0.0):
    corners = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (0, 1)]) * size
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return corners @ rot.T + np.asarray(centre)


def test_geometric_die_sized_box_gives_four_top_down_and_four_side_grasps():
    plans = GeometricGraspPlanner().candidates(_box([0.046, 0.05, 0.05]))[:8]
    kinds = [p.metadata["kind"] for p in plans]
    assert kinds == ["top_down"] * 4 + ["side"] * 4
    assert [round(p.metadata["closing_width"], 3) for p in plans] == [0.046, 0.046, 0.05, 0.05] * 2
    tips = [p.metadata["tip_pos"][2] for p in plans[:2]]
    assert tips == pytest.approx([0.05 - 0.015, 0.05 - 0.025])
    td = plans[0]
    np.testing.assert_allclose(td.grasp_pos - np.asarray(td.metadata["tip_pos"]), [0, 0, GripperSpec().tcp_offset])
    np.testing.assert_allclose(quat_wxyz_to_matrix(td.grasp_quat)[:, 2], [0, 0, -1], atol=1e-9)
    for p in plans:
        assert p.retreat_pos[2] > p.grasp_pos[2] + 0.1
        assert p.waypoints[2].finger_width == p.finger_closed
    side = plans[4]
    assert side.metadata["tip_pos"][2] == pytest.approx(0.025)
    assert abs(side.metadata["approach_dir"][2]) < 1e-9


def test_geometric_skips_axes_wider_than_the_gripper_and_follows_yaw():
    assert GeometricGraspPlanner().candidates(_box([0.13, 0.19, 0.024])) == []
    plans = GeometricGraspPlanner(GripperSpec(usable_width=0.07)).candidates(_box([0.10, 0.03, 0.02], yaw=np.pi / 6))
    assert {round(p.metadata["closing_width"], 4) for p in plans} == {0.03}
    finger = np.asarray(plans[0].metadata["finger_dir"])
    assert abs(abs(finger @ [-np.sin(np.pi / 6), np.cos(np.pi / 6), 0]) - 1) < 1e-6


def test_geometric_plan_reads_the_urdf_and_the_object_pose(tmp_path):
    urdf = tmp_path / "model.urdf"
    urdf.write_text('<robot name="t"><link name="base"><visual><origin xyz="0 0 0.025"/>'
                    '<geometry><box size="0.06 0.03 0.05"/></geometry></visual></link></robot>', encoding="utf-8")
    asset = type("A", (), {"urdf_path": urdf})()
    yaw90 = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    plans = GeometricGraspPlanner().plan(np.array([0.5, 0.0, 0.8]), yaw90, asset=asset)
    widths = sorted({round(p.metadata["closing_width"], 3) for p in plans})
    assert widths == [0.03, 0.06]
    narrow = next(p for p in plans if p.metadata["closing_width"] == pytest.approx(0.03))
    assert abs(narrow.metadata["finger_dir"][0]) == pytest.approx(1.0, abs=1e-6)
    assert plans[0].metadata["tip_pos"][2] == pytest.approx(0.85 - 0.015)


def test_geometric_flipped_orientation_repeats_the_base_list_turned_about_the_approach():
    plans = GeometricGraspPlanner().candidates(_box([0.046, 0.05, 0.05]))
    base, flipped = plans[:8], plans[8:]
    assert [p.metadata["orientation"] for p in plans] == ["base"] * 8 + ["flipped"] * 8
    for b, f in zip(base, flipped):
        assert f.metadata["kind"] == b.metadata["kind"]
        np.testing.assert_allclose(f.metadata["tip_pos"], b.metadata["tip_pos"])
        np.testing.assert_allclose(f.metadata["approach_dir"], b.metadata["approach_dir"])
        np.testing.assert_allclose(f.metadata["finger_dir"], -np.asarray(b.metadata["finger_dir"]))
    assert [p.metadata["candidate_index"] for p in plans] == list(range(16))


def test_geometric_usable_width_bounds_which_axes_get_candidates():
    narrow = GeometricGraspPlanner(GripperSpec(usable_width=0.048)).candidates(_box([0.046, 0.05, 0.05]))
    assert {round(p.metadata["closing_width"], 3) for p in narrow} == {0.046}
    assert GeometricGraspPlanner(GripperSpec(usable_width=0.04)).candidates(_box([0.046, 0.05, 0.05])) == []
