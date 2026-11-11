"""The diagnosis request says what stopped each candidate, gate rejections included."""

import trimesh

from robotsmith.grasp.diagnostics import request
from robotsmith.grasp.geometric import GeometricGraspPlanner, GripperSpec


def _asset(tmp_path):
    trimesh.creation.box([0.064, 0.032, 0.024]).export(tmp_path / "box.obj")
    urdf = tmp_path / "model.urdf"
    urdf.write_text('<robot name="t"><link name="base"><inertial><mass value="0.03"/></inertial>'
                    '<visual><geometry><mesh filename="box.obj"/></geometry></visual></link></robot>', encoding="utf-8")
    return urdf


def test_evidence_carries_each_candidates_outcome_and_counts_them(tmp_path):
    plans = GeometricGraspPlanner().candidates(trimesh.creation.box([0.064, 0.032, 0.024]).vertices)
    gate = {"segment": "close", "index": 12, "reason": "collision",
            "pair": [["robot", "left_finger"], ["env", "table"]]}
    label = {"trials": [
        {"candidate_index": 0, "kind": "top_down", "lifted": False, "lift_m": None, "gate": gate},
        {"candidate_index": 1, "kind": "top_down", "lifted": False, "lift_m": 0.004, "gate": "passed",
         "approach": {"ok": True}, "place": {"ok": True}},
        {"candidate_index": 2, "kind": "top_down", "lifted": False, "lift_m": None,
         "approach": {"ok": True}, "place": {"ok": False, "label": "search_exhausted"}, "gate": "passed"},
    ]}
    packet = request.evidence(urdf_path=_asset(tmp_path), category="block", pos=[0, 0, 0], quat=[1, 0, 0, 0],
                              aabb=[[-0.032, -0.016, -0.012], [0.032, 0.016, 0.012]], table_z=-0.012,
                              gripper=GripperSpec(), plans=plans, label=label, object_xy=[0, 0])
    rows = packet["trials"]
    assert [r["outcome"] for r in rows] == ["gate_rejected", "tried", "place_refused"]
    assert rows[0]["gate"] == gate and "lift_m" not in rows[0]
    assert rows[1]["lift_m"] == 0.004 and rows[2]["place"] == "search_exhausted"
    assert rows[0]["tip_xyz"] == [round(v, 4) for v in plans[0].metadata["tip_pos"]]
    assert packet["question"] == (
        f"The planner produced {len(plans)} candidates; 1 was rejected by the execution gate before any physics step, "
        "1 was skipped because the place from where their pick ends was refused, "
        "1 was tried and none lifted the object (max lift 0.004 m); diagnose.")


def test_support_clearance_is_no_longer_a_repair_direction():
    assert "support_clearance" not in request.DIRECTIONS
