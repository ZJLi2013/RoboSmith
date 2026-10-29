"""open / close bind to the joint the executor drags and are judged on the state at the end of the drag."""

from types import SimpleNamespace

import numpy as np
import pytest

from robotsmith.assets.schema import UrdfJoint
from robotsmith.scenario.adapters import to_step_bindings
from robotsmith.scenario.intents import SkillIntentSequenceSpec, SkillIntentSpec
from robotsmith.scenario.specs import TaskSuccessCfg
from robotsmith.scenario_runtime import runner
from robotsmith.scenario_runtime.runner import evaluate_scenario_task, run_scenario_episode
from robotsmith.skills import Skill
from robotsmith.skills.frames import _resolve_handle_drag
from robotsmith.tasks import evaluate_predicate, failed_leaves
from robotsmith.tasks.predicates import JudgeInputError
from robotsmith.tasks.task_spec import SuccessNode, TaskSpec


def _drawer(lower=0.0, upper=0.35, extra=()):
    joints = [UrdfJoint("slide", "prismatic", "base", "drawer", (1.0, 0.0, 0.0), lower, upper), *extra]
    return SimpleNamespace(
        name="drawer",
        metadata=SimpleNamespace(handles=[{"local_pos": [0.27, 0.0, 0.33]}]),
        movable_joints=joints,
        get_joint=lambda n: next((j for j in joints if j.name == n), None),
    )


def _bind(asset, name, params=None):
    steps = SkillIntentSequenceSpec(intents=(SkillIntentSpec(name=name, target="drawer", params=params or {}),))
    return to_step_bindings(steps, TaskSuccessCfg(predicate="object_above", params={}), {"drawer": asset})[0]


def _drag(asset, name, slide, params=None):
    state = {"positions": {"drawer": np.zeros(3)}, "assets": {"drawer": asset},
             "object_quats": {"drawer": [1.0, 0.0, 0.0, 0.0]}, "joint_positions": {"drawer": {"slide": slide}}}
    _, move, _ = _resolve_handle_drag(SimpleNamespace(skill=Skill(name, "drawer", params=params or {}),
                                                      scene_state=state), opening=name == "open")
    return move


def test_default_open_and_close_bind_the_endpoints_the_drag_reaches():
    asset = _drawer()
    assert _bind(asset, "open") == {"status": "bound", "predicate": "joint_opened",
                                    "params": {"object": "drawer", "joint": "slide", "threshold": pytest.approx(0.33)}}
    assert _bind(asset, "close")["params"]["threshold"] == pytest.approx(0.02)
    assert np.allclose(_drag(asset, "open", 0.0), [0.35, 0, 0]) and np.allclose(_drag(asset, "close", 0.35), [-0.35, 0, 0])


def test_nonzero_lower_closes_to_the_lower_limit():
    asset = _drawer(lower=0.05)
    assert _bind(asset, "close")["params"]["threshold"] == pytest.approx(0.07)
    assert np.allclose(_drag(asset, "close", 0.30), [-0.25, 0, 0])


def test_open_from_a_partial_opening_drags_the_remainder_to_the_same_endpoint():
    asset = _drawer()
    assert np.allclose(_drag(asset, "open", 0.10), [0.25, 0, 0])
    assert _bind(asset, "open")["params"]["threshold"] == pytest.approx(0.33)


def test_explicit_joint_and_distance_bind_what_the_drag_uses():
    other = UrdfJoint("lid", "prismatic", "base", "lid", (0.0, 1.0, 0.0), 0.0, 0.5)
    asset = _drawer(extra=(other,))
    params = {"joint": "lid", "distance": 0.2}
    assert _bind(asset, "open", params)["params"] == {"object": "drawer", "joint": "lid", "threshold": pytest.approx(0.18)}
    assert np.allclose(_drag(asset, "open", 0.0, params), [0, 0.2, 0])


def test_a_joint_the_asset_lacks_fails_at_load():
    with pytest.raises(ValueError, match="no task joint"):
        _bind(_drawer(), "open", {"joint": "ghost"})


class _Ent:
    def __init__(self, pos):
        self.pos = np.asarray(pos, dtype=float)

    def get_pos(self):
        return self.pos

    def get_quat(self):
        return np.array([1.0, 0.0, 0.0, 0.0])

    def get_AABB(self):
        return np.stack([self.pos - 0.01, self.pos + 0.01])

    def get_contacts(self, with_entity=None):
        return {"link_a": np.zeros(1 if isinstance(with_entity, _Ent) and with_entity is not self else 0)}


class _Env:
    """A die inside a drawer whose slide reads ``q``; each unrecorded sim step sets it to ``rebound``."""

    motors_dof = list(range(9))

    def __init__(self, q, rebound=None):
        self.q, self.rebound = q, rebound
        self.scene = SimpleNamespace(dt=1.0 / 30.0, t=0, step=self._step)
        self.franka = SimpleNamespace(control_dofs_position=lambda q, dofs: None)
        self.entity_map = {"die": _Ent([0.6, 0.0, 0.835]), "drawer": _Ent([0.6, 0.0, 0.775])}
        self.placed_map, self.object_heights, self.asset_map, self.object_quats = {}, {}, {}, {}
        self.table_surface_z, self.articulated_joint_dofs, self.planner = 0.775, {}, object()

    def _step(self):
        self.scene.t += 1
        if self.rebound is not None:
            self.q = self.rebound

    def reset(self, obj_xyz_map, *, settle_steps):
        pass

    def get_joint_positions(self):
        return {} if self.q is None else {"drawer": {"slide": self.q}}

    def solve_ik(self, *args, **kwargs):
        raise AssertionError("not used by the stubbed plan_segment")


CLOSED = {"status": "bound", "predicate": "joint_closed", "params": {"object": "drawer", "joint": "slide", "threshold": 0.02}}
INSIDE = SuccessNode.leaf("object_in_box", {"object": "die", "parent": "drawer", "mode": "inside", "yaw_deg": 0.0,
                                            "box": [[-0.1, -0.1, 0.0], [0.1, 0.1, 0.05]]})


def _close(monkeypatch, env, success, on_record=None):
    monkeypatch.setattr(runner, "plan_segment", lambda skill, *a, **kw: ([np.zeros(9)] * 2, None, None))
    recorded = []

    def record(env, dataset, seg, task_spec, **kw):
        env.scene.t += len(seg)
        recorded.append(len(seg))
        if on_record:
            on_record(env)
        return []

    task = TaskSpec(name="t", scene="t", instruction="", success=success,
                    skills=[Skill("close", "drawer")], step_bindings=[CLOSED])
    result = run_scenario_episode(env, task, executor=object(), motion_params=object(), dataset=object(),
                                  record_fn=record)
    return task, result, recorded


def test_close_is_judged_at_the_end_of_the_drag_without_a_hold(monkeypatch):
    env = _Env(q=0.01)
    _, result, recorded = _close(monkeypatch, env, SuccessNode.leaf("joint_closed", CLOSED["params"]))
    step = result.handoffs[0]
    assert step["done"] is True and step["completion"]["settle_frames"] == 0
    assert evaluate_predicate("joint_closed", step["completion"]["judge_state"], CLOSED["params"]) is True
    assert recorded == [2] and result.frame_count == 2 and env.scene.t == 2
    assert result.final_window is None


def test_a_missing_joint_observation_is_an_error(monkeypatch):
    with pytest.raises(JudgeInputError, match="slide"):
        _close(monkeypatch, _Env(q=None), SuccessNode.leaf("joint_closed", CLOSED["params"]))


def test_a_close_that_rebounds_keeps_its_step_result_while_the_task_fails(monkeypatch):
    env = _Env(q=0.0, rebound=0.1)
    task, result, _ = _close(monkeypatch, env, SuccessNode.leaf("joint_closed", CLOSED["params"]))
    ok, state = evaluate_scenario_task(env, task, initial_positions={}, hold=result.final_qpos,
                                       window=result.final_window)
    assert result.handoffs[0]["done"] is True and ok is False
    assert len(state["window"]["steps"]) == 10


def test_inside_and_closed_is_rejudged_after_close_and_fails_when_the_die_is_pushed_out(monkeypatch):
    env = _Env(q=0.0)
    success = SuccessNode(op="all", terms=[INSIDE, SuccessNode.leaf("joint_closed", CLOSED["params"])])

    def push_out(env):
        env.entity_map["die"].pos = np.array([0.9, 0.0, 0.835])

    task, result, _ = _close(monkeypatch, env, success, on_record=push_out)
    ok, state = evaluate_scenario_task(env, task, initial_positions={}, hold=result.final_qpos,
                                       window=result.final_window)
    assert result.handoffs[0]["done"] is True and ok is False
    assert failed_leaves(success, state) == [INSIDE.to_dict()]
