from types import SimpleNamespace

import pytest

from robotsmith.sim.sim_env import SimEnv


class _FakeFranka:
    def set_dofs_position(self, *args, **kwargs):
        pass

    def control_dofs_position(self, *args, **kwargs):
        pass

    def zero_all_dofs_velocity(self):
        pass


class _FakeEntity:
    def __init__(self):
        self.quat = None

    def set_pos(self, *args, **kwargs):
        pass

    def set_quat(self, quat, *args, **kwargs):
        self.quat = quat

    def get_pos(self):
        return [0.5, 0.0, 0.797]

    def get_quat(self):
        return [0.0, 0.707107, 0.707107, 0.0]


class _FakeScene:
    def step(self):
        pass


def test_reset_converts_metadata_wxyz_quat_to_genesis_xyzw():
    env = object.__new__(SimEnv)
    ent = _FakeEntity()
    env.franka = _FakeFranka()
    env.motors_dof = []
    env.entity_map = {"object": ent}
    env.placed_map = {
        "object": SimpleNamespace(quaternion=[0.0, 0.707107, 0.707107, 0.0])
    }
    env.object_heights = {"object": 0.04}
    env.object_quats = {"object": [1.0, 0.0, 0.0, 0.0]}
    env.table_surface_z = 0.775
    env.scene = _FakeScene()

    positions = env.reset({"object": (0.5, 0.0)}, settle_steps=0)

    assert ent.quat is not None
    assert ent.quat.squeeze(0).tolist() == pytest.approx([0.707107, 0.707107, 0.0, 0.0])
    assert positions["object"].tolist() == pytest.approx([0.5, 0.0, 0.797])
    assert env.object_quats["object"].tolist() == pytest.approx([0.0, 0.707107, 0.707107, 0.0])
