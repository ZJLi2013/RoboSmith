from pathlib import Path

import numpy as np
import pytest

from robotsmith.cap.specs import FrameRef
from robotsmith.scenario_runtime import load_candidate, materialize_candidate
from robotsmith.scenario_runtime.runner import run_scenario_episode
from scripts.datagen.run_generated_scenario import episode_diagnostics
from robotsmith.tasks.task_spec import SuccessNode, TaskSpec as LegacyTaskSpec
from robotsmith.grasp.planner import GraspPlan
from robotsmith.motion.params import MotionParams
from robotsmith.skills import Skill, run_skills


def _write_candidate(path: Path) -> Path:
    path.write_text(
        """
from robotsmith.cap.intents import intent_sequence, pick, place
from robotsmith.cap.predicates import object_in_container
from robotsmith.cap.scene_api import layout, obj, region, scene, target_position
from robotsmith.cap.task_api import task

table_layout = layout(regions=[
    region("left_reachable", ((0.42, -0.14), (0.50, -0.06))),
    region("goal_area", ((0.58, -0.14), (0.66, -0.06)), z=0.809),
])

scenario_scene = scene(
    "cap_move_die_to_goal",
    layout=table_layout,
    objects=[
        obj("die", asset="die_01", region="left_reachable"),
        obj("mug", asset="mug_02", fixed_position=(0.58, 0.10, 0.809)),
    ],
    target_positions=[target_position("goal", region="goal_area")],
)

scenario_task = task(
    "cap_move_die_to_goal",
    scene=scenario_scene,
    instruction="Move the die to the goal.",
    success=object_in_container("die", "goal", xy_threshold=0.04),
)

scenario_intents = intent_sequence([
    pick("die", category="die"),
    place("goal", category="die"),
])


def build():
    return scenario_scene, scenario_task, scenario_intents
""",
        encoding="utf-8",
    )
    return path


def test_load_candidate_uses_build_entrypoint(tmp_path):
    candidate_path = _write_candidate(tmp_path / "generated_scenario.py")

    candidate = load_candidate(candidate_path)

    assert candidate.source_path == candidate_path
    assert candidate.scene.name == "cap_move_die_to_goal"
    assert candidate.task.name == "cap_move_die_to_goal"
    assert [intent.name for intent in candidate.intents.intents] == ["pick", "place"]


def test_materialize_candidate_reconstructs_legacy_runtime_objects(tmp_path):
    candidate_path = _write_candidate(tmp_path / "generated_scenario.py")

    package = materialize_candidate(candidate_path)

    assert package.legacy_scene.name == "cap_move_die_to_goal"
    assert [obj.name_override for obj in package.legacy_scene.objects] == ["die", "mug"]
    assert package.legacy_task.scene == "cap_move_die_to_goal"
    assert package.legacy_task.success.op == "leaf"
    assert package.legacy_task.success.predicate == "object_in_container"
    assert package.legacy_task.success.params["object"] == "die"
    assert package.legacy_task.success.params["container"] == "goal"
    # Full skill-dict lowering shape is covered in test_cap_authoring; here we
    # only confirm materialize reconstructs the skill sequence + routing.
    assert [(s.name, s.target) for s in package.legacy_task.skills] == [
        ("pick", "die"),
        ("place", "goal"),
    ]


def test_load_candidate_rejects_missing_build(tmp_path):
    candidate_path = tmp_path / "generated_scenario.py"
    candidate_path.write_text("x = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="build"):
        load_candidate(candidate_path)


def test_smoke_diagnostics_records_object_target_error():
    task = LegacyTaskSpec(
        name="cap_move_die_to_goal",
        scene="cap_move_die_to_goal",
        instruction="Move the die to the goal.",
        success=SuccessNode.leaf("object_in_container", {"object": "die", "container": "goal"}),
        skills=[],
    )

    diagnostics = episode_diagnostics(
        task,
        initial_positions={"die": np.array([0.42, -0.1, 0.82], dtype=np.float32)},
        final_positions={"die": np.array([0.60, -0.12, 0.82], dtype=np.float32)},
        target_positions={"goal": np.array([0.62, -0.12, 0.809], dtype=np.float32)},
        skill_traces=[{"skill": "pick", "target": "die", "score_exec": np.float32(0.5)}],
    )

    assert diagnostics["final_positions"]["die"] == pytest.approx([0.60, -0.12, 0.82])
    assert diagnostics["target_positions"]["goal"] == pytest.approx([0.62, -0.12, 0.809])
    assert diagnostics["success_xy_error"] == pytest.approx(0.02)
    assert diagnostics["skill_traces"][0]["score_exec"] == pytest.approx(0.5)


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
        qpos, held_category=None, held_place_z=None, phase="all",
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
        success=SuccessNode.leaf("object_in_container", {"object": "die", "container": "goal"}),
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
