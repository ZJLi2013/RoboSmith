import numpy as np
import pytest

from robotsmith.sim.world_state import collect_world_state, settle_window


class _Ent:
    links = ()

    def __init__(self, pos):
        self.pos = np.asarray(pos, dtype=float)

    def get_pos(self):
        return self.pos

    def get_quat(self):
        return np.array([1.0, 0.0, 0.0, 0.0])

    def get_AABB(self):
        return np.stack([self.pos - 0.01, self.pos + 0.01])

    def get_contacts(self, with_entity=None):
        """Every two scene entities touch; nothing touches the arm."""
        return {"link_a": np.zeros(1 if type(with_entity) is _Ent and with_entity is not self else 0)}


class _Scene:
    dt = 0.01

    def __init__(self):
        self.t = 0

    def step(self):
        self.t += 1


class _Franka(_Ent):
    def __init__(self):
        super().__init__([0.0, 0.0, 0.0])
        self.commands = []

    def get_contacts(self, with_entity=None):
        return {"link_a": np.zeros(0), "link_b": np.zeros(0)}

    def control_dofs_position(self, q, dofs):
        self.commands.append(np.asarray(q))


class _Env:
    motors_dof = list(range(9))

    def __init__(self):
        self.scene = _Scene()
        self.franka = _Franka()
        self.entity_map = {"die": _Ent([0.5, 0.0, 0.84]), "supporter": _Ent([0.56, 0.0, 0.775])}

    def get_joint_positions(self):
        return {}


def test_zero_settle_steps_nothing():
    env = _Env()
    assert settle_window(env, np.zeros(9), settle_s=0.0) == {"steps": [], "poses": {"die": [], "supporter": []}}
    assert env.scene.t == 0 and env.franka.commands == []


def test_settle_holds_the_command_and_samples_evenly():
    env, hold = _Env(), np.arange(9.0)
    window = settle_window(env, hold, settle_s=0.5, samples=10)
    assert env.scene.t == 50
    assert window["steps"] == list(range(5, 51, 5))
    assert len(window["poses"]["die"]) == 10
    assert all(np.array_equal(q, hold) for q in env.franka.commands)


def test_world_state_reads_one_step_without_advancing():
    env = _Env()
    env.scene.t = 7
    state = collect_world_state(env)
    assert env.scene.t == 7 and state["step"] == 7
    assert state["contacts"]["die"] == ["supporter"]
    assert state["aabbs"]["die"][0] == pytest.approx([0.49, -0.01, 0.83])
