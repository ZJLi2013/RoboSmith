from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
from robotsmith.sim.sim_env import SceneInvalid, SimEnv, settle_violations

UP = [1.0, 0.0, 0.0, 0.0]
YAW_90 = [0.707107, 0.0, 0.0, 0.707107]
ON_SIDE = [0.707107, 0.0, 0.707107, 0.0]
SPAWN = (0.5, 0.0, 0.797)


class _FakeFranka:
    def set_dofs_position(self, *args, **kwargs):
        pass

    def control_dofs_position(self, *args, **kwargs):
        pass

    def zero_all_dofs_velocity(self):
        pass


class _FakeEntity:
    def __init__(self, pos):
        self.pos = pos

    def set_pos(self, *args, **kwargs):
        pass

    def set_quat(self, *args, **kwargs):
        pass

    def get_pos(self):
        return list(self.pos)

    def get_quat(self):
        return UP


class _FakeScene:
    def step(self):
        pass


@pytest.mark.parametrize("spawn_quat, settled, bad", [
    (UP, ((0.501, 0.001, 0.796), UP), False),
    (YAW_90, (SPAWN, YAW_90), False),
    (UP, ((0.70, 0.0, 0.797), UP), True),
    (UP, ((0.5, 0.0, 0.70), UP), True),
    (UP, (SPAWN, ON_SIDE), True),
])
def test_settle_violations_flag_roll_drop_and_tip_but_not_yaw(spawn_quat, settled, bad):
    assert bool(settle_violations({"object": (SPAWN, spawn_quat)}, {"object": settled})) is bad


def test_reset_raises_when_the_object_rolls_away():
    env = object.__new__(SimEnv)
    env.franka = _FakeFranka()
    env.motors_dof = []
    env.entity_map = {"object": _FakeEntity([-0.08, -0.46, 0.92])}
    env.placed_map = {"object": SimpleNamespace(quaternion=UP)}
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
        return tuple(self.ent.pos)

    def step(self):
        self.calls.append("step")
        self.ent.pos[2] += 0.05

    def reset(self, state):
        self.calls.append("reset")
        self.ent.pos = list(state)


class _RecordingFranka(_FakeFranka):
    def __init__(self):
        self.commands = []

    def get_dofs_position(self, *args, **kwargs):
        return [0.1] * 9

    def control_dofs_position(self, q, *args, **kwargs):
        self.commands.append(list(q))


def test_trial_from_here_restores_the_scene_and_the_arm_target():
    env = object.__new__(SimEnv)
    ent = _FakeEntity([0.42, -0.27, 0.80])
    env.franka = _RecordingFranka()
    env.motors_dof = list(range(9))
    env.entity_map = {"object": ent}
    env.scene = _SnapshotScene(ent)

    result = env.trial_from_here([[0.0] * 9] * 3, "object")

    assert result["lifted"] and result["lift_m"] == pytest.approx(0.15)
    assert env.scene.calls == ["get_state", "step", "step", "step", "reset"]
    assert ent.pos == pytest.approx([0.42, -0.27, 0.80])
    assert env.franka.commands[-1] == pytest.approx([0.1] * 9)


def _geoms(*items):
    return [SimpleNamespace(idx=i, link=SimpleNamespace(name=n)) for i, n in items]


class _QueryEntity(_FakeEntity):
    def __init__(self, pos, geoms, qpos=()):
        super().__init__(pos)
        self.geoms, self.qpos = geoms, list(qpos)
        self.n_dofs = len(self.qpos)

    def set_dofs_position(self, qpos, *args, **kwargs):
        self.qpos[: len(qpos)] = list(qpos)

    def get_qpos(self):
        return list(self.qpos)

    def get_dofs_velocity(self):
        return [0.0] * self.n_dofs


class _QueryScene(_SnapshotScene):
    def __init__(self, ent, pairs):
        super().__init__(ent)
        self.sim = SimpleNamespace(rigid_solver=SimpleNamespace(detect_collision=lambda: pairs))


def _query_env(pairs):
    env = object.__new__(SimEnv)
    die = _QueryEntity([0.5, 0.0, 0.8], _geoms((3, "die_link")))
    env.franka = _QueryEntity([0.0, 0.0, 0.0], _geoms((0, "hand"), (1, "left_finger")), qpos=[0.0] * 9)
    env.end_effector = _FakeEntity([0.5, 0.0, 1.0])
    env.motors_dof = list(range(9))
    env.entity_map = {"die": die}
    env.scene = _QueryScene(die, pairs)
    return env


def test_contacts_at_names_the_links_and_restores_the_scene():
    env = _query_env([[1, 3], [3, 7]])

    assert env.contacts_at([0.0] * 9) == [(("robot", "left_finger"), ("die", "die_link")),
                                          (("die", "die_link"), ("env", "?"))]
    assert env.scene.calls == ["get_state", "reset"]


def test_contacts_at_restores_the_scene_when_the_query_fails():
    env = _query_env([])

    with pytest.raises(KeyError):
        env.contacts_at([0.0] * 9, held=("missing", np.eye(4)))
    assert env.scene.calls == ["get_state", "reset"]


def test_state_fingerprint_follows_the_joint_state():
    env = _query_env([])
    before = env.state_fingerprint()
    env.franka.qpos[0] = 0.1

    assert env.state_fingerprint() != before


def test_a_real_step_changes_the_fingerprint_and_a_query_does_not():
    env = _query_env([[1, 3]])
    before = env.state_fingerprint()
    env.contacts_at([0.0] * 9)
    env.hand_pose([0.0] * 9)
    assert env.state_fingerprint() == before

    env.step()
    assert env.state_fingerprint() != before


def test_posed_restores_the_fingerprint_and_voids_motions_verified_inside():
    from robotsmith.execution.gate import ExecutionGate, StaleMotion

    env = _query_env([])
    arm, die = env.franka, env.entity_map["die"]
    stepped = (list(arm.qpos), list(die.pos))
    # As in Genesis, a restore returns to the last stepped state and drops the writes since.
    env.scene.get_state = lambda: stepped
    env.scene.reset = lambda state: (setattr(arm, "qpos", list(state[0])), setattr(die, "pos", list(state[1])))
    gate = ExecutionGate(env)
    before = env.state_fingerprint()
    with env.posed([0.0] * 9, {"die": np.eye(4)}):
        inside = gate.ungated([[0.0] * 9])
        gate.check_current(inside)
    assert env.state_fingerprint() == before
    with pytest.raises(StaleMotion):
        gate.check_current(inside)
    with env.posed([0.3] * 9, {}):
        env.contacts_at([0.0] * 9)
        env.hand_pose([0.0] * 9)
        assert arm.qpos == pytest.approx([0.3] * 9)
        with pytest.raises(StaleMotion):
            gate.check_current(inside)
    assert arm.qpos == pytest.approx([0.0] * 9) and env.state_fingerprint() == before


class _ResettableFranka(_QueryEntity):
    def control_dofs_position(self, *args, **kwargs):
        pass

    def zero_all_dofs_velocity(self):
        pass


def test_a_reset_changes_the_fingerprint():
    from robotsmith.execution.gate import ExecutionGate, StaleMotion
    from robotsmith.sim.franka import HOME_QPOS

    env = _query_env([])
    env.franka = _ResettableFranka([0.0, 0.0, 0.0], [], qpos=list(HOME_QPOS))
    env.entity_map, env.object_quats, env.articulated_joint_dofs, env.articulated_joint_init = {}, {}, {}, {}
    gate = ExecutionGate(env)
    motion = gate.ungated([HOME_QPOS])
    gate.check_current(motion)

    env.reset({}, settle_steps=0)
    assert env.franka.get_qpos() == pytest.approx(list(HOME_QPOS))
    with pytest.raises(StaleMotion):
        gate.check_current(motion)


def _link(idx, name, parent):
    return SimpleNamespace(idx=idx, name=name, parent_idx=parent)


def test_joint_limits_and_adjacent_links_come_from_the_robot():
    env = _query_env([])
    names = ["link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand"]
    env.franka.links = ([_link(i, n, i - 1) for i, n in enumerate(names)]
                        + [_link(9, "left_finger", 8), _link(10, "right_finger", 8)])
    env.franka.get_dofs_limit = lambda dofs: ([-2.0] * 7 + [0.0, 0.0], [2.0] * 7 + [0.04, 0.04])

    lo, hi = env.joint_limits
    assert lo.tolist() == [-2.0] * 7 + [0.0, 0.0] and hi.tolist() == [2.0] * 7 + [0.04, 0.04]
    joined = env.joined_links
    for pair in [("link0", "link1"), ("link7", "hand"), ("hand", "left_finger"), ("left_finger", "right_finger")]:
        assert frozenset(pair) in joined
    for pair in [("link3", "link5"), ("link7", "left_finger"), ("link0", "hand")]:
        assert frozenset(pair) not in joined


class _Body:
    def __init__(self, idx, name, parent, pos, half):
        self.idx, self.name, self.parent_idx, self.pos, self.half = idx, name, parent, np.asarray(pos, float), half
        self.n_geoms = 1 if half else 0

    def get_pos(self):
        return self.pos

    def get_quat(self):
        return np.array(UP)

    def get_AABB(self):
        return np.stack([self.pos - self.half, self.pos + self.half])


def test_lever_arms_bound_every_link_and_the_held_object():
    from robotsmith.sim.franka import JOINT_NAMES

    # link0 at the base, link1..link7 0.1 m apart up z, the hand 0.1 m above link7, fingers 2 cm to either side.
    links = [_Body(0, "link0", -1, (0, 0, 0), 0.0)]
    links += [_Body(i, f"link{i}", i - 1, (0, 0, 0.1 * i), 0.05) for i in range(1, 8)]
    links += [_Body(8, "hand", 7, (0, 0, 0.8), 0.05),
              _Body(9, "left_finger", 8, (0, 0.02, 0.85), 0.01), _Body(10, "right_finger", 8, (0, -0.02, 0.85), 0.01)]
    joint_link = dict(zip(JOINT_NAMES, [links[i] for i in (1, 2, 3, 4, 5, 6, 7, 9, 10)]))
    env = object.__new__(SimEnv)
    env.franka = SimpleNamespace(
        links=links, get_dofs_limit=lambda dofs: ([-2.0] * 7 + [0.0, 0.0], [2.0] * 7 + [0.04, 0.04]),
        get_joint=lambda n: SimpleNamespace(link=joint_link[n], get_anchor_pos=lambda: joint_link[n].pos))
    env.motors_dof, env.end_effector = list(range(9)), links[8]
    env.entity_map = {"die": _Body(20, "die", -1, (0, 0, 0.9), 0.02)}

    box = np.sqrt(3) * 0.05
    finger = np.hypot(0.02, 0.05) + 0.04 + np.sqrt(3) * 0.01
    assert finger > box
    r = env.lever_arms()
    assert r[6] == pytest.approx(0.1 + finger) and r[0] == pytest.approx(0.7 + finger)
    assert r[7:].tolist() == [1.0, 1.0]
    held = env.lever_arms(("die", np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.1], [0, 0, 0, 1.0]])))
    assert held[6] == pytest.approx(0.1 + max(finger, 0.1 + np.sqrt(3) * 0.02))
    assert np.all(held[:7] >= r[:7]) and held[6] > r[6]


class _Kind:
    def __init__(self, name):
        self.name = name


def _clearance_env(verts, faces, *, box=True):
    """One robot link carrying ``verts``/``faces``, a wall that is a box unless ``box`` is false, and a mesh die."""
    env = object.__new__(SimEnv)
    env._posed = None
    env.motors_dof = list(range(9))
    env.asset_map = {"wall": SimpleNamespace(is_fixture=True), "die": SimpleNamespace(is_fixture=False)}
    robot = SimpleNamespace(
        geoms=[SimpleNamespace(link=SimpleNamespace(name="link5"), init_verts=verts, init_faces=faces,
                               get_pos=lambda: np.zeros(3), get_quat=lambda: np.array([1.0, 0.0, 0.0, 0.0]))],
        set_dofs_position=lambda *a, **k: None)
    wall_geom = SimpleNamespace(
        link=SimpleNamespace(name="base"), type=_Kind("BOX" if box else "MESH"), data=np.array([0.02, 0.02, 0.02]),
        get_pos=lambda: np.zeros(3), get_quat=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    die_geom = SimpleNamespace(
        link=SimpleNamespace(name="base"), type=_Kind("MESH"), data=None,
        init_verts=np.zeros((1, 3)), init_faces=np.zeros((0, 3), int))
    wall = SimpleNamespace(geoms=[wall_geom])
    die = SimpleNamespace(geoms=[die_geom])
    env.franka = robot
    env.entity_map = {"wall": wall, "die": die}
    env.scene = SimpleNamespace(entities=[robot, wall, die], get_state=lambda: "snap", reset=lambda state: None)
    return env


def test_the_clearance_lower_bound_sees_a_face_whose_vertices_are_far():
    from robotsmith.execution.gate import CLEARANCE_M

    d = CLEARANCE_M
    z = 0.01 + d / 2
    radius = 0.05
    verts = np.array([[radius, 0.0, z],
                      [-radius / 2, radius * np.sqrt(3) / 2, z],
                      [-radius / 2, -radius * np.sqrt(3) / 2, z]])
    half = np.array([0.01, 0.01, 0.01])
    vertex_distance = np.linalg.norm(np.maximum(np.abs(verts) - half, 0.0), axis=1)
    assert np.all(vertex_distance > d)
    env = _clearance_env(verts, np.array([[0, 1, 2]]))
    bounds = {(link, entity): dist for link, entity, dist in env.clearance_at(np.zeros(9))}
    assert bounds[("link5", "wall")] <= d / 2
    assert bounds[("link5", "wall")] < vertex_distance.min()
    assert "die" not in {entity for _, entity in bounds}


def test_a_fixture_without_box_collision_geometry_raises():
    env = _clearance_env(np.zeros((1, 3)), np.zeros((0, 3), int), box=False)
    with pytest.raises(ValueError, match="wall"):
        env.clearance_at(np.zeros(9))
