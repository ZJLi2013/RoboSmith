"""VAMP backend: world conversion, timing, goal validity and backend selection, without the vamp extension."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from robotsmith.motion import vamp_backend as vb
from robotsmith.motion.planner import Waypoint
from robotsmith.scenario_runtime import runner


def test_box_moves_into_the_base_frame_with_half_extents_and_its_rotation():
    q = [np.cos(0.3), 0.0, 0.0, np.sin(0.3)]  # 0.6 rad about z
    (p,) = vb.to_vamp_primitives([{"type": "box", "center": [0.5, 0.1, 0.9], "extent": [0.2, 0.4, 0.06], "quat": q}],
                                 base_z=0.775)
    assert np.allclose(p["center"], [0.5, 0.1, 0.125])
    assert np.allclose(p["half_extents"], [0.1, 0.2, 0.03])
    assert np.allclose(p["euler_xyz"], [0.0, 0.0, 0.6])


def test_upward_halfspace_becomes_a_slab_whose_top_is_the_table():
    (p,) = vb.to_vamp_primitives([{"type": "halfspace", "point": [0, 0, 0.775], "normal": [0.0, 0.0, 1.0]}], base_z=0.775)
    assert p["center"][2] + p["half_extents"][2] == pytest.approx(0.0)


def test_quaternion_round_trip():
    q = np.array([0.8, 0.2, -0.3, 0.4]) / np.linalg.norm([0.8, 0.2, -0.3, 0.4])
    x, y, z, w = vb.matrix_to_quat_xyzw(vb.quat_wxyz_to_matrix(q))
    assert np.allclose(np.sign(w) * np.array([w, x, y, z]), np.sign(q[0]) * q, atol=1e-9)


def test_timing_bounds_every_step_and_keeps_the_endpoints():
    path = np.array([[0.0] * 7, [0.01] * 7, [1.5, -1.0, 0, 0, 0, 0, 0.2], [1.6, -1.0, 0, 0, 0, 0, 0.2]])
    traj = vb.time_parameterize(path, max_joint_vel=1.0, control_dt=0.02)
    assert np.allclose(traj[0], path[0]) and np.allclose(traj[-1], path[-1])
    assert np.max(np.abs(np.diff(traj, axis=0))) <= 0.02 + 1e-9


class _FakeRobot:
    def __init__(self, goal_ok):
        self.goal_ok = goal_ok

    def halton(self):
        return None

    def validate(self, q, env):
        return self.goal_ok or not np.allclose(q, 1.0)

    def rrtc(self, start, goal, env, settings, rng):
        return SimpleNamespace(solved=True, path=None, start=start, goal=goal)

    def simplify(self, path, env, settings, rng):
        return SimpleNamespace(path=SimpleNamespace(numpy=lambda: np.stack([np.zeros(7), np.ones(7)])))


def _backend(monkeypatch, goal_ok=True):
    fake = SimpleNamespace(panda=_FakeRobot(goal_ok), Environment=lambda: SimpleNamespace(add_cuboid=lambda c: None),
                           Cuboid=lambda *a: a, configure_robot_and_planner_with_kwargs=lambda r, p: (None, None, None, None))
    monkeypatch.setitem(sys.modules, "vamp", fake)
    return vb.VampBackend(lambda pos, quat, finger, init_qpos=None: np.ones(9), base_z=0.775)


WP = [Waypoint(np.array([0.5, 0.0, 1.0]), np.array([0.0, 1.0, 0.0, 0.0]), 0.04)]
WORLD = [{"type": "halfspace", "point": [0, 0, 0.775], "normal": [0.0, 0.0, 1.0]}]


def test_plan_returns_nine_dim_timed_steps(monkeypatch):
    res = _backend(monkeypatch).plan_motion(np.zeros(9), WP, world=WORLD, control_dt=0.02)
    assert res.success and all(q.shape == (9,) for q in res.trajectory)
    assert np.allclose(res.trajectory[-1][7:], 0.04)


def test_invalid_goal_is_a_failed_plan(monkeypatch):
    res = _backend(monkeypatch, goal_ok=False).plan_motion(np.zeros(9), WP, world=WORLD, control_dt=0.02)
    assert not res.success and res.reason == "goal-invalid"


def test_default_environment_does_not_select_vamp(monkeypatch):
    monkeypatch.delenv("ROBOSMITH_MOTION_BACKEND", raising=False)
    monkeypatch.delenv("ROCROBO_BACKEND", raising=False)
    assert runner._maybe_motion_planner(SimpleNamespace(solve_ik=None, table_surface_z=0.775)) is None
