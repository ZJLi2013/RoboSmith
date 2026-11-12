"""Joint-state success predicates.

Runtime ``joint_opened`` / ``joint_closed`` read ``env_state["joint_positions"]``;
authoring helpers build leaves of the composable success tree (no core-loop change). Validated locally against a synthetic env_state.
"""

import pytest

from robotsmith.scenario.predicates import all_of, joint_closed, joint_opened
from robotsmith.scenario.validators import ValidationError, check_task
from robotsmith.scenes.config import ObjectPlacement, SceneConfig
from robotsmith.tasks.predicates import evaluate_predicate, evaluate_success
from robotsmith.tasks.task_spec import TaskSpec


def _env(qpos: float) -> dict:
    return {"joint_positions": {"drawer": {"drawer_slide": qpos}}}


class TestRuntimePredicates:
    def test_joint_opened_threshold(self):
        params = {"object": "drawer", "joint": "drawer_slide", "threshold": 0.25}
        assert evaluate_predicate("joint_opened", _env(0.30), params) is True
        assert evaluate_predicate("joint_opened", _env(0.25), params) is True
        assert evaluate_predicate("joint_opened", _env(0.10), params) is False

    def test_joint_closed_threshold(self):
        params = {"object": "drawer", "joint": "drawer_slide", "threshold": 0.02}
        assert evaluate_predicate("joint_closed", _env(0.00), params) is True
        assert evaluate_predicate("joint_closed", _env(0.05), params) is False


class TestAuthoringHelpers:
    def test_joint_opened_leaf(self):
        node = joint_opened("drawer", "drawer_slide", open_position=0.25)
        assert (node.op, node.predicate) == ("leaf", "joint_opened")
        assert node.params == {
            "object": "drawer",
            "joint": "drawer_slide",
            "threshold": 0.25,
        }

    def test_joint_closed_leaf_default(self):
        node = joint_closed("drawer", "drawer_slide")
        assert node.predicate == "joint_closed"
        assert node.params["threshold"] == 0.02


def _drawer_scene():
    return SceneConfig(name="open_drawer", objects=[
        ObjectPlacement(asset_query="drawer_cabinet", fixed_position=[0.55, 0.0, 0.0], name_override="drawer")])


def _task(name, instruction, success):
    return TaskSpec(name=name, scene="open_drawer", instruction=instruction, success=success)


class TestRoundTrip:
    def test_check_and_evaluate(self):
        task = _task("open_drawer", "Open the drawer",
                     joint_opened("drawer", "drawer_slide", open_position=0.25))
        check_task(task, _drawer_scene(), ())  # "drawer" resolves to the scene object

        assert evaluate_success(task.success, _env(0.30)) is True
        assert evaluate_success(task.success, _env(0.10)) is False

    def test_composite_open_and_closed_other(self):
        # all_of works the same with joint leaves (no core-loop change).
        task = _task("open_drawer_partial", "Open the drawer partway", all_of(
            joint_opened("drawer", "drawer_slide", open_position=0.25),
            joint_closed("drawer", "drawer_slide", closed_position=0.40),
        ))
        check_task(task, _drawer_scene(), ())
        assert evaluate_success(task.success, _env(0.30)) is True
        assert evaluate_success(task.success, _env(0.45)) is False

    def test_unknown_object_ref_rejected(self):
        task = _task("bad_ref", "Open the missing drawer",
                     joint_opened("ghost", "drawer_slide", open_position=0.25))
        with pytest.raises(ValidationError, match="ghost"):
            check_task(task, _drawer_scene(), ())
