"""Genesis plan_path backend: timing, reach gate, post-hoc contact gate, held object and rollback, without Genesis."""

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from robotsmith.motion import genesis_plan_backend as gb
from robotsmith.motion.planner import Waypoint
from robotsmith.scenario_runtime import runner
from robotsmith.sim.sim_env import SimEnv

QUAT = np.array([0.0, 1.0, 0.0, 0.0])
WP = [Waypoint(np.array([0.5, 0.0, 1.0]), QUAT, 0.04)]
WORLD = [{"type": "halfspace", "point": [0, 0, 0.775], "normal": [0.0, 0.0, 1.0]}]
NO_CONTACT = {"geom_a": [], "geom_b": [], "link_a": [], "link_b": [], "penetration": []}


class _Franka:
    geom_start, geom_end = 0, 10
    links = [SimpleNamespace(idx=i, name=n) for i, n in enumerate(["link0", "link7", "left_finger", "right_finger"])]

    def __init__(self, solved=True, contact_at=None):
        self.solved, self.contact_at, self.q, self.plan_kwargs = solved, contact_at, None, None
        self._solver = SimpleNamespace(_kernel_detect_collision=lambda: None)

    def set_qpos(self, q, zero_velocity=False):
        self.q = np.asarray(q, dtype=np.float64)

    def plan_path(self, qpos_goal, qpos_start, return_valid_mask, **kw):
        self.plan_kwargs = kw
        path = np.linspace(qpos_start, qpos_goal, 300)
        return path, np.array([self.solved])

    def get_contacts(self):
        if self.contact_at is not None and np.allclose(self.q[0], self.contact_at, atol=0.02):
            return {"geom_a": [1], "geom_b": [20], "link_a": [1], "link_b": [99], "penetration": [0.01]}
        return NO_CONTACT


class _Env:
    def __init__(self, franka, ik_pos_err=0.0):
        self.franka, self.ik_pos_err, self.rolled_back = franka, ik_pos_err, 0
        self.end_effector = SimpleNamespace(get_pos=lambda: WP[0].pos + [self.ik_pos_err, 0, 0], get_quat=lambda: QUAT)
        self.entity_map = {"cube": SimpleNamespace(geom_start=10, geom_end=11, set_pos=lambda p: None,
                                                   set_quat=lambda q: None, get_contacts=lambda: NO_CONTACT)}

    def solve_ik(self, pos, quat, finger, init_qpos=None):
        return np.r_[np.ones(7), finger, finger]

    @contextmanager
    def preserved_state(self):
        try:
            yield
        finally:
            self.rolled_back += 1


def test_timing_bounds_every_step_drops_crowded_points_and_keeps_the_endpoints():
    path = np.vstack([np.linspace(0, 0.1, 300)[:, None] * np.ones(7), [[1.6, -1.0, 0, 0, 0, 0, 0.2]]])
    traj = gb.time_parameterize(path, max_joint_vel=1.0, control_dt=0.02)
    assert np.allclose(traj[0], path[0]) and np.allclose(traj[-1], path[-1])
    assert np.max(np.abs(np.diff(traj, axis=0))) <= 0.02 + 1e-9
    assert len(traj) < 150


def test_plan_returns_nine_dim_timed_steps_and_rolls_the_scene_back():
    env = _Env(_Franka())
    res = gb.GenesisPlanBackend(env).plan_motion(np.zeros(9), WP, world=WORLD, control_dt=1 / 30)
    assert res.success and all(q.shape == (9,) for q in res.trajectory)
    assert np.allclose(res.trajectory[-1][7:], 0.04) and np.allclose(res.trajectory[-1][:7], 1.0)
    assert env.rolled_back == 1


@pytest.mark.parametrize("env,reason", [
    (_Env(_Franka(), ik_pos_err=0.02), "ik-miss"),
    (_Env(_Franka(solved=False)), "plan-path-failed"),
    (_Env(_Franka(contact_at=0.5)), "posthoc-contact"),
])
def test_each_gate_returns_a_failed_plan(env, reason):
    res = gb.GenesisPlanBackend(env).plan_motion(np.zeros(9), WP, world=WORLD, control_dt=1 / 30)
    assert not res.success and res.reason == reason and env.rolled_back == 1


def test_held_object_rides_into_plan_path():
    env = _Env(_Franka())
    attach = {"target": "cube", "T_ee_obj": np.eye(4).tolist(), "spheres": []}
    gb.GenesisPlanBackend(env).plan_motion(np.zeros(9), WP, world=WORLD, control_dt=1 / 30, attach=attach)
    assert env.franka.plan_kwargs == {"ee_link_name": "hand", "with_entity": env.entity_map["cube"]}


def test_preserved_state_rolls_back_on_error():
    calls = []
    env = SimpleNamespace(scene=SimpleNamespace(get_state=lambda: "snap", reset=calls.append))
    with pytest.raises(RuntimeError):
        with SimEnv.preserved_state(env):
            raise RuntimeError("boom")
    assert calls == ["snap"]


def test_default_environment_does_not_select_genesis_planning(monkeypatch):
    monkeypatch.delenv("ROBOSMITH_MOTION_BACKEND", raising=False)
    monkeypatch.delenv("ROCROBO_BACKEND", raising=False)
    assert runner._maybe_motion_planner(SimpleNamespace(solve_ik=None, table_surface_z=0.775)) is None
