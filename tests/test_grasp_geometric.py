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
    plans = GeometricGraspPlanner().candidates(_box([0.046, 0.05, 0.05]))
    kinds = [p.metadata["kind"] for p in plans]
    assert kinds == ["top_down"] * 4 + ["side"] * 4
    assert [round(p.metadata["closing_width"], 3) for p in plans] == [0.046, 0.046, 0.05, 0.05] * 2
    tips = [p.metadata["tip_pos"][2] for p in plans[:2]]
    assert tips == pytest.approx([0.05 - 0.015, 0.05 - 0.025])
    td = plans[0]
    np.testing.assert_allclose(td.grasp_pos - np.asarray(td.metadata["tip_pos"]), [0, 0, GripperSpec().tcp_offset])
    np.testing.assert_allclose(quat_wxyz_to_matrix(td.grasp_quat)[:, 2], [0, 0, -1], atol=1e-6)
    for p in plans:
        assert p.retreat_pos[2] > p.grasp_pos[2] + 0.1
        assert p.waypoints[2].finger_width == p.finger_closed
    side = plans[4]
    assert side.metadata["tip_pos"][2] == pytest.approx(0.025)
    assert abs(side.metadata["approach_dir"][2]) < 1e-9


def test_geometric_fingers_close_along_the_hand_y_axis():
    for p in GeometricGraspPlanner().candidates(_box([0.10, 0.03, 0.05], yaw=np.pi / 6)):
        rot = quat_wxyz_to_matrix(p.grasp_quat)
        np.testing.assert_allclose(rot[:, 1], p.metadata["finger_dir"], atol=1e-6)
        np.testing.assert_allclose(rot[:, 2], p.metadata["approach_dir"], atol=1e-6)


def test_geometric_top_down_keeps_the_finger_pads_clear_of_the_support():
    g = GripperSpec()
    plans = GeometricGraspPlanner().candidates(_box([0.06, 0.032, 0.024]))
    top = [p for p in plans if p.metadata["kind"] == "top_down"]
    assert top and all(p.metadata["tip_pos"][2] == pytest.approx(g.pad_below_tcp + g.support_clearance) for p in top)


def test_geometric_skips_axes_wider_than_the_gripper_and_follows_yaw():
    assert GeometricGraspPlanner().candidates(_box([0.13, 0.19, 0.024])) == []
    plans = GeometricGraspPlanner().candidates(_box([0.10, 0.03, 0.02], yaw=np.pi / 6))
    assert {round(p.metadata["closing_width"], 4) for p in plans} == {0.03}
    finger = np.asarray(plans[0].metadata["finger_dir"])
    assert abs(abs(finger @ [-np.sin(np.pi / 6), np.cos(np.pi / 6), 0]) - 1) < 1e-6


def test_geometric_plan_reads_the_urdf_and_the_object_pose(tmp_path):
    urdf = tmp_path / "model.urdf"
    urdf.write_text('<robot name="t"><link name="base"><visual><origin xyz="0 0 0.025"/>'
                    '<geometry><box size="0.06 0.03 0.05"/></geometry></visual></link></robot>', encoding="utf-8")
    asset = type("A", (), {"urdf_path": urdf, "root_dir": urdf.parent})()
    yaw90 = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    plans = GeometricGraspPlanner().plan(np.array([0.5, 0.0, 0.8]), yaw90, asset=asset)
    widths = sorted({round(p.metadata["closing_width"], 3) for p in plans})
    assert widths == [0.03, 0.06]
    narrow = next(p for p in plans if p.metadata["closing_width"] == pytest.approx(0.03))
    assert abs(narrow.metadata["finger_dir"][0]) == pytest.approx(1.0, abs=1e-6)
    assert plans[0].metadata["tip_pos"][2] == pytest.approx(0.85 - 0.015)


def _bottle_urdf(tmp_path, inertial_rpy="0 0 0"):
    urdf = tmp_path / "model.urdf"
    urdf.write_text(f'<robot name="t"><link name="base"><inertial><origin xyz="0 0 0.1" rpy="{inertial_rpy}"/>'
                    '<mass value="0.5"/></inertial><visual><origin xyz="0 0 0.1"/>'
                    '<geometry><box size="0.06 0.03 0.2"/></geometry></visual></link></robot>', encoding="utf-8")
    return type("A", (), {"urdf_path": urdf, "root_dir": urdf.parent})()


def test_geometric_plan_places_the_urdf_relative_to_the_inertial_origin_genesis_reports(tmp_path):
    yaw90 = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    plans = GeometricGraspPlanner().plan(np.array([0.5, 0.1, 0.875]), yaw90, asset=_bottle_urdf(tmp_path), scale=0.5)
    assert sorted({round(p.metadata["closing_width"], 4) for p in plans}) == [0.015, 0.03]
    top = plans[0].metadata["tip_pos"]
    assert top[:2] == pytest.approx([0.5, 0.1])
    assert top[2] == pytest.approx(0.875 + 0.05 - 0.015)
    side = next(p for p in plans if p.metadata["kind"] == "side")
    assert side.metadata["tip_pos"][2] == pytest.approx(0.875)
    narrow = next(p for p in plans if p.metadata["closing_width"] == pytest.approx(0.015))
    assert abs(narrow.metadata["finger_dir"][0]) == pytest.approx(1.0, abs=1e-6)


def test_geometric_plan_refuses_a_rotated_inertial_frame(tmp_path):
    with pytest.raises(ValueError, match="rotated inertial frame"):
        GeometricGraspPlanner().plan(np.array([0.5, 0.0, 0.875]), asset=_bottle_urdf(tmp_path, "0 0 0.3"))
