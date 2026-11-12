"""The acceptance contact monitor on a real Genesis scene (needs a GPU; ROBOSMITH_GPU_TESTS=1, -s for the readings)."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("ROBOSMITH_GPU_TESTS"), reason="needs a GPU and Genesis")

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "acceptance" / "f1_wall_gap.yaml"
TOP_DOWN = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
TCP = 0.1034
CLEAR_TCP = np.array([0.45, -0.20, 0.975])
WALL_TCP = np.array([0.55, 0.12, 0.805])


def test_monitor_reads_a_known_legal_and_a_known_illegal_motion():
    from robotsmith.execution.gate import Span
    from robotsmith.scenario.load import load_scenario
    from robotsmith.sim.sim_env import SimEnv

    spec = importlib.util.spec_from_file_location("certify", ROOT / "scripts" / "acceptance" / "certify.py")
    certify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(certify)

    env = SimEnv.build(load_scenario(SCENARIO, seed=100).scene, seed=100)
    mon = certify.Monitor(env)
    ctx = ([Span("move", 0, 10_000)], "die", {"env"}, "probe")
    readings = {}
    for name, tcp in [("clear", CLEAR_TCP), ("wall", WALL_TCP)]:
        env.reset({}, settle_steps=10)
        q = np.asarray(env.solve_ik(tcp + [0, 0, TCP], quat=TOP_DOWN, finger_pos=0.04), dtype=np.float32)
        mon.start()
        mon.enter("replay", ctx)
        for _ in range(90):
            env.franka.control_dofs_position(q, env.motors_dof)
            env.step()
        readings[name] = {"steps": mon.steps["replay"], "violating": mon.violating["replay"], "first": mon.first}
    print("MONITOR " + json.dumps(readings))

    assert readings["clear"] == {"steps": 90, "violating": 0, "first": None}
    assert readings["wall"]["violating"] > 0
    assert "robot" in {p[0] for p in readings["wall"]["first"]["pair"]}
