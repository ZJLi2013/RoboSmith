from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
from robotsmith.sim.sim_env import SceneInvalid, SimEnv, settle_violations

UP = [1.0, 0.0, 0.0, 0.0]
YAW_90 = [0.707107, 0.0, 0.0, 0.707107]
ON_SIDE = [0.707107, 0.0, 0.707107, 0.0]


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


@pytest.mark.parametrize("settled, bad", [
    (((0.501, 0.001, 0.796), UP), False),
    (((0.70, 0.0, 0.797), UP), True),
    (((0.5, 0.0, 0.70), UP), True),
    (((0.5, 0.0, 0.797), ON_SIDE), True),
])
def test_settle_violations_flags_roll_drop_and_tip(settled, bad):
    assert bool(settle_violations({"object": ((0.5, 0.0, 0.797), UP)}, {"object": settled})) is bad


def test_settle_violations_ignores_yaw_of_the_placement():
    spawned = {"object": ((0.5, 0.0, 0.797), YAW_90)}
    assert settle_violations(spawned, {"object": ((0.5, 0.0, 0.797), YAW_90)}) == []


def test_reset_raises_when_the_object_rolls_away():
    env = object.__new__(SimEnv)
    ent = _FakeEntity()
    ent.get_pos = lambda: [-0.08, -0.46, 0.92]
    env.franka = _FakeFranka()
    env.motors_dof = []
    env.entity_map = {"object": ent}
    env.placed_map = {"object": SimpleNamespace(quaternion=[0.0, 0.707107, 0.707107, 0.0])}
    env.object_heights = {"object": 0.04}
    env.object_quats = {}
    env.articulated_joint_dofs = {}
    env.table_surface_z = 0.775
    env.scene = _FakeScene()

    with pytest.raises(SceneInvalid, match="object: moved 0.6"):
        env.reset({"object": (0.42, -0.27)}, settle_steps=0)
