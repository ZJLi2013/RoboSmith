"""The execution gate on a real Genesis scene (needs a GPU; run with ROBOSMITH_GPU_TESTS=1 and -s for the readings)."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("ROBOSMITH_GPU_TESTS"), reason="needs a GPU and Genesis")

SCENARIO = Path(__file__).resolve().parents[1] / "scenarios" / "acceptance" / "f1_wall_gap.yaml"
TOP_DOWN = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
TCP = 0.1034
# wall_01 on the table at (0.55, 0.12): x 0.54-0.56, y -0.08-0.32, z 0.775-0.835.
CLEAR_TCP = np.array([0.45, -0.20, 0.975])
WALL_TCP = np.array([0.55, 0.12, 0.805])
INTO_WALL = np.array([0.55, 0.12, 0.81])
OVER_WALL = np.array([0.55, 0.12, 0.95])
# Clear air beside the wall: the die is carried here and released.
RELEASE = np.array([0.45, -0.05, 0.90])


def test_gate_on_the_wall_scene():
    from robotsmith.execution.gate import ExecutionGate, Holding, MotionRejected, Span, StaleMotion
    from robotsmith.grasp.transforms import pose_matrix
    from robotsmith.scenario.load import load_scenario
    from robotsmith.sim.sim_env import SimEnv

    env = SimEnv.build(load_scenario(SCENARIO, seed=100).scene, seed=100)
    gate = ExecutionGate(env)
    ik = lambda tcp: np.asarray(env.solve_ik(tcp + [0, 0, TCP], quat=TOP_DOWN, finger_pos=0.04),  # noqa: E731
                                dtype=np.float64)
    q_clear, q_wall = ik(CLEAR_TCP), ik(WALL_TCP)
    die = env.object_pose("die")[0]
    q_grasp, q_into, q_over, q_release = ik(die), ik(INTO_WALL), ik(OVER_WALL), ik(RELEASE)
    carry = [Span("close", 0, 1, allow="grip"), Span("carry", 1, 2, holding=True)]
    release = carry + [Span("retreat", 2, 3, allow="grip")]
    t_hand_die = np.linalg.inv(pose_matrix(*env.hand_pose(q_grasp))) @ pose_matrix(*env.object_pose("die"))
    held_at = lambda q: (pose_matrix(*env.hand_pose(q)) @ t_hand_die)[:3, 3]  # noqa: E731
    holding = Holding("die", t_hand_die)

    readings = {"die_pos": np.round(die, 4).tolist(),
                "held_offset_m": {k: round(float(np.linalg.norm(held_at(q) - p)), 4)
                                  for k, q, p in [("into", q_into, INTO_WALL), ("over", q_over, OVER_WALL),
                                                  ("release", q_release, RELEASE)]}}
    with pytest.raises(MotionRejected) as midway:
        gate.verify([q_clear, q_wall, q_clear], [Span("move", 0, 3)], target="die", q_start=q_clear)
    readings["midway_wall"] = midway.value.to_dict()
    clear = gate.verify([q_clear, q_clear], [Span("move", 0, 2)], target="die", q_start=q_clear)
    readings["clear_checked"] = clear.checked
    with pytest.raises(MotionRejected) as into:
        gate.verify([q_grasp, q_into], carry, target="die", q_start=q_grasp, holding=holding)
    readings["held_into_wall"] = into.value.to_dict()
    over = gate.verify([q_grasp, q_over], carry, target="die", q_start=q_grasp, holding=holding)
    readings["held_over_wall_checked"] = over.checked
    # The palm goes down to where the die was let go.
    with pytest.raises(MotionRejected) as through:
        gate.verify([q_grasp, q_release, ik(RELEASE - [0, 0, TCP])], release, target="die", q_start=q_grasp,
                    holding=holding)
    readings["retreat_through_released"] = through.value.to_dict()
    up = gate.verify([q_grasp, q_release, ik(RELEASE + [0, 0, 0.1])], release, target="die", q_start=q_grasp,
                     holding=holding)
    readings["retreat_up_checked"] = up.checked
    gate.check_current(over)
    env.step()
    with pytest.raises(StaleMotion):
        gate.check_current(over)
    readings["stale_after_step"] = True
    print("GATE " + json.dumps(readings))

    assert readings["midway_wall"]["index"] == 1
    assert {tuple(p)[0] for p in readings["midway_wall"]["pair"]} == {"robot", "wall"}
    assert readings["held_into_wall"]["segment"] == "carry"
    assert {tuple(p)[0] for p in readings["held_into_wall"]["pair"]} == {"die", "wall"}
    assert clear.checked == 2 and over.checked == 2 and up.checked == 3
    assert all(d < 0.01 for d in readings["held_offset_m"].values())
    assert readings["retreat_through_released"]["segment"] == "retreat"
    assert {tuple(p)[0] for p in readings["retreat_through_released"]["pair"]} == {"robot", "die"}
