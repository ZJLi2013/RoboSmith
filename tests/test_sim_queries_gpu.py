"""SimEnv's static queries on a real Genesis scene (needs a GPU; run with ROBOSMITH_GPU_TESTS=1 and -s for the readings)."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("ROBOSMITH_GPU_TESTS"), reason="needs a GPU and Genesis")

SCENARIO = Path(__file__).resolve().parents[1] / "scenarios" / "acceptance" / "f1_wall_gap.yaml"
TOP_DOWN = np.array([0.0, 1.0, 0.0, 0.0])
TCP = 0.1034
# wall_01 is a 0.02 x 0.4 x 0.06 box standing on the table (z 0.775) at (0.55, 0.12):
# x 0.54-0.56, y -0.08-0.32, z 0.775-0.835.
CLEAR_TCP = np.array([0.45, -0.20, 0.975])
NEAR_TCP = np.array([0.45, 0.12, 0.90])  # 9 cm short of the wall, fingertips above its top
WALL_TCP = np.array([0.55, 0.12, 0.805])  # fingers inside the wall
YAW30 = np.array([np.cos(np.pi / 12), 0.0, 0.0, np.sin(np.pi / 12)])
# Hand frame top-down is world (x, -y, -z): hand · T puts the die (0.10, 0, -dz) from the hand origin.
T_INTO_WALL = (np.array([0.10, 0.0, NEAR_TCP[2] + TCP - 0.81]), YAW30)  # die centre (0.55, 0.12, 0.81)
T_OVER_WALL = (np.array([0.10, 0.0, 0.10]), YAW30)  # die centre (0.55, 0.12, 0.9034), bottom above 0.835


def _hits(contacts, who):
    return sorted({(b[0] if a[0] == who else a[0]) for a, b in contacts if (a[0] == who) != (b[0] == who)})


def _ms(fn, n):
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e3)
    return {"n": n, "min": round(min(out), 2), "median": round(float(np.median(out)), 2),
            "p90": round(float(np.percentile(out, 90)), 2), "max": round(max(out), 2)}


def test_contacts_at_restores_the_scene():
    from robotsmith.grasp.transforms import pose_matrix
    from robotsmith.scenario.load import load_scenario
    from robotsmith.sim.sim_env import JOINT_NAMES, SimEnv

    env = SimEnv.build(load_scenario(SCENARIO, seed=100).scene, seed=100)
    ik = lambda tcp: np.asarray(env.solve_ik(tcp + [0, 0, TCP], quat=TOP_DOWN.astype(np.float32),  # noqa: E731
                                             finger_pos=0.04), dtype=np.float64)
    q = {"clear": ik(CLEAR_TCP), "near": ik(NEAR_TCP), "wall": ik(WALL_TCP)}
    fp0 = env.state_fingerprint()

    order = ["clear", "wall", "clear", "wall", "near"]
    robot = [(name, _hits(env.contacts_at(q[name]), "robot")) for name in order]
    wall_pairs = sorted({f"{a[0]}/{a[1]}-{b[0]}/{b[1]}" for a, b in env.contacts_at(q["wall"]) if "wall" in (a[0], b[0])})
    hand_wall = env.hand_pose(q["wall"])
    held = {}
    for name, t in [("into_wall", T_INTO_WALL), ("over_wall", T_OVER_WALL)]:
        c = env.contacts_at(q["near"], held=("die", pose_matrix(*t)))
        held[name] = {"die": _hits(c, "die"), "robot": _hits(c, "robot")}
    fp_queries = env.state_fingerprint()
    with pytest.raises(KeyError):
        env.contacts_at(q["wall"], held=("missing", pose_matrix(*T_INTO_WALL)))
    fp_error = env.state_fingerprint()

    _ms(lambda: env.contacts_at(q["wall"]), 5)
    timing = {"contacts_at_clear": _ms(lambda: env.contacts_at(q["clear"]), 50),
              "contacts_at_wall": _ms(lambda: env.contacts_at(q["wall"]), 50),
              "contacts_at_held": _ms(lambda: env.contacts_at(q["near"], held=("die", pose_matrix(*T_INTO_WALL))), 50),
              "hand_pose": _ms(lambda: env.hand_pose(q["wall"]), 50)}

    env.trial_from_here([q["clear"]] * 10, "die")
    fp_trial = env.state_fingerprint()
    start = {n: tuple(po.position) for n, po in env.placed_map.items()}
    env.reset(start)
    fp_reset1, q_reset1 = env.state_fingerprint(), np.asarray(env.franka.get_qpos().cpu()).reshape(-1)
    env.reset(start)
    fp_reset2, q_reset2 = env.state_fingerprint(), np.asarray(env.franka.get_qpos().cpu()).reshape(-1)
    for _ in range(5):
        env.step()
    fp_steps = env.state_fingerprint()

    readings = {
        "scenario": SCENARIO.name, "seed": 100, "joint_order": list(JOINT_NAMES),
        "qpos": {k: np.round(v, 4).tolist() for k, v in q.items()},
        "tcp_targets": {"clear": CLEAR_TCP.tolist(), "near": NEAR_TCP.tolist(), "wall": WALL_TCP.tolist()},
        "held_T": {"into_wall": [T_INTO_WALL[0].round(4).tolist(), YAW30.round(4).tolist()],
                   "over_wall": [T_OVER_WALL[0].round(4).tolist(), YAW30.round(4).tolist()]},
        "robot_contacts_in_order": robot, "wall_pairs_at_wall": wall_pairs, "held_at_near": held,
        "hand_wall_pos": np.round(hand_wall[0], 4).tolist(), "hand_wall_target": (WALL_TCP + [0, 0, TCP]).tolist(),
        "fingerprint_same": {"after_queries": fp_queries == fp0, "after_failed_query": fp_error == fp0},
        "fingerprint_changed": {"after_trial": fp_trial != fp0, "after_second_same_reset": fp_reset2 != fp_reset1,
                                "after_5_steps": fp_steps != fp_reset2},
        "same_reset_arm_qpos_max_diff": float(np.abs(q_reset1 - q_reset2).max()),
        "timing_ms_after_5_warmup_calls": timing,
    }
    print("PROBE " + json.dumps(readings))

    assert [("wall" in hits, hits == []) for _, hits in robot] == [(False, True), (True, False)] * 2 + [(False, True)]
    assert any("finger" in p for p in wall_pairs)
    assert held == {"into_wall": {"die": ["wall"], "robot": []}, "over_wall": {"die": [], "robot": []}}
    assert np.linalg.norm(hand_wall[0] - (WALL_TCP + [0, 0, TCP])) < 0.01
    assert all(readings["fingerprint_same"].values()) and all(readings["fingerprint_changed"].values())
