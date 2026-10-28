import numpy as np
import pytest

from robotsmith.scenario.specs import FrameRef
from robotsmith.scenario_runtime.runner import run_scenario_episode
from robotsmith.tasks.task_spec import SuccessNode, TaskSpec as LegacyTaskSpec
from robotsmith.grasp.planner import GraspPlan
from robotsmith.motion.params import MotionParams
from robotsmith.skills import Skill, run_skills


def test_run_scenario_episode_reanchors_to_live_resolved_positions(monkeypatch):
    class FakeEntity:
        def __init__(self, pos):
            self._pos = np.asarray(pos, dtype=np.float32)

        def get_pos(self):
            return self._pos

    class FakePlaced:
        def __init__(self, position):
            self.position = position

    class FakeEnv:
        def __init__(self):
            self.placed_map = {
                "die": FakePlaced([0.42, -0.10, 0.82]),
                "mug": FakePlaced([0.62, 0.10, 0.81]),
            }
            self.entity_map = {
                "die": FakeEntity([0.42, -0.10, 0.82]),
                "mug": FakeEntity([0.62, 0.10, 0.81]),
            }
            self.object_heights = {"die": 0.02, "mug": 0.08}
            self.asset_map = {}
            self.object_quats = {}
            self.table_surface_z = 0.80
            self.articulated_joint_dofs = {}
            self.planner = object()
            self.did_reset = False

        def reset(self, obj_xyz_map, *, settle_steps):
            # Reset receives the full resolved xyz so shelf/in-drawer starts keep their height.
            assert obj_xyz_map == {
                "die": (pytest.approx(0.42), pytest.approx(-0.10), pytest.approx(0.82)),
                "mug": (pytest.approx(0.62), pytest.approx(0.10), pytest.approx(0.81)),
            }
            self.did_reset = True

        def get_joint_positions(self):
            return {}

        def solve_ik(self, *args, **kwargs):
            raise AssertionError("not used by stubbed plan_segment")

    seen: dict = {}

    def fake_plan_segment(
        skill, index, skills, planner, executor, solve_ik, scene_state, params,
        qpos, held_category=None, held_place_z=None, phase="all", **_,
    ):
        seen["positions"] = scene_state["positions"]
        assert "joint_positions" in scene_state
        scene_state["skill_traces"].append({"skill": skill.name, "target": skill.target})
        return [np.zeros(9, dtype=np.float32)], held_category, held_place_z

    recorded: list[int] = []

    def fake_record_episode(env, dataset, seg, task_spec, **kwargs):
        recorded.append(len(seg))

    monkeypatch.setattr(
        "robotsmith.scenario_runtime.runner.plan_segment", fake_plan_segment
    )
    task = LegacyTaskSpec(
        name="cap_move_die_to_goal",
        scene="cap_move_die_to_goal",
        instruction="Move the die to the goal.",
        success=SuccessNode.leaf("object_above", {"object": "die", "reference": "table"}),
        skills=[Skill("pick", "die", "die"), Skill("place", "goal", "die")],
    )

    env = FakeEnv()
    result = run_scenario_episode(
        env,
        task,
        executor=object(),
        motion_params=object(),
        dataset=object(),
        record_fn=fake_record_episode,
        target_frames={"goal": FrameRef(kind="world", xyz=(0.62, -0.10, 0.809))},
    )

    assert env.did_reset
    assert result.initial_positions["die"] == pytest.approx([0.42, -0.10, 0.82])
    assert seen["positions"]["die"] == pytest.approx([0.42, -0.10, 0.82])
    assert seen["positions"]["goal"] == pytest.approx([0.62, -0.10, 0.809])
    assert result.skill_traces == [
        {"skill": "pick", "target": "die"},
        {"skill": "place", "target": "goal"},
    ]
    assert result.frame_count == 2
    assert recorded == [1, 1]


def _fake_plan(x, y, z):
    q = np.array([1.0, 0.0, 0.0, 0.0])
    return GraspPlan(
        grasp_pos=np.array([x, y, z]),
        grasp_quat=q,
        pre_grasp_pos=np.array([x, y, z + 0.1]),
        pre_grasp_quat=q,
        retreat_pos=np.array([x, y, z + 0.15]),
        retreat_quat=q,
        finger_open=0.04,
        finger_closed=0.01,
        metadata={"source": "fake"},
    )


class _FakePlanner:
    resolved_place_z = None

    def plan(self, *args, **kwargs):
        return [_fake_plan(0.5, -0.1, 0.92)]

    def plan_place(self, place_pos, *, category="block", place_point_world=None, approach=None):
        self.place_point_world = np.asarray(place_point_world, dtype=np.float64)
        return _fake_plan(*self.place_point_world)


class _FakeExecutor:
    last_motion_trace = None

    def pick(self, plan, motion_planner, qpos, params, **kwargs):
        return [np.ones(9)]

    def place(self, plan, motion_planner, qpos, params, *, phase="all", **kwargs):
        self.last_motion_trace = {"action": "place", "segments": []}
        return [np.full(9, 2.0)]


# Drop point = goal marker z 0.809 + place_z. Unset place_z falls back to the
# held grasp's offset above the support: 0.92 - 0.80 = 0.12.
@pytest.mark.parametrize("place_params, drop_z, resolved, source", [
    ({"place_z": 0.82}, 1.629, 0.82, "requested"),
    ({}, 0.929, 0.12, "selected_grasp"),
])
def test_place_z_resolution(place_params, drop_z, resolved, source):
    planner = _FakePlanner()
    scene_state = {
        "home_qpos": np.zeros(9),
        "positions": {
            "die": np.array([0.45, -0.1, 0.80]),
            "goal": np.array([0.52, -0.11, 0.809]),
        },
        "table_surface_z": 0.80,
        "skill_traces": [],
    }

    run_skills(
        [Skill("pick", "die", "die"), Skill("place", "goal", "die", place_params)],
        planner,
        _FakeExecutor(),
        solve_ik=lambda *args, **kwargs: np.zeros(9),
        scene_state=scene_state,
        params=MotionParams(),
    )

    assert planner.place_point_world[2] == pytest.approx(drop_z)
    place_trace = scene_state["skill_traces"][-1]
    assert place_trace["requested_place_z"] == place_params.get("place_z")
    assert place_trace["resolved_place_z"] == pytest.approx(resolved)
    assert place_trace["place_z_source"] == source
