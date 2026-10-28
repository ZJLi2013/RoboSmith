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

    with pytest.raises(SceneInvalid, match="object: moved 0.535 m"):
        env.reset({"object": (0.42, -0.27)}, settle_steps=0)


class _SnapshotScene:
    def __init__(self, ent):
        self.ent, self.calls = ent, []

    def get_state(self):
        self.calls.append("get_state")
        return ("snapshot", tuple(self.ent.pos))

    def step(self):
        self.calls.append("step")
        self.ent.pos[2] += 0.05

    def reset(self, state):
        self.calls.append("reset")
        self.ent.pos = list(state[1])


class _MovingEntity:
    def __init__(self):
        self.pos = [0.42, -0.27, 0.80]

    def get_pos(self):
        return list(self.pos)


class _RecordingFranka(_FakeFranka):
    def __init__(self):
        self.commands = []

    def get_dofs_position(self, *args, **kwargs):
        return [0.1] * 9

    def control_dofs_position(self, q, *args, **kwargs):
        self.commands.append(list(q))


def test_trial_from_here_restores_the_scene_and_the_arm_target():
    env = object.__new__(SimEnv)
    ent = _MovingEntity()
    env.franka = _RecordingFranka()
    env.motors_dof = list(range(9))
    env.entity_map = {"object": ent}
    env.scene = _SnapshotScene(ent)

    result = env.trial_from_here([[0.0] * 9] * 3, "object")

    assert result["lifted"] and result["lift_m"] == pytest.approx(0.15)
    assert env.scene.calls == ["get_state", "step", "step", "step", "reset"]
    assert ent.pos == pytest.approx([0.42, -0.27, 0.80])
    assert env.franka.commands[-1] == pytest.approx([0.1] * 9)
