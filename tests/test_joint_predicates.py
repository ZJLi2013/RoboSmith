"""Feature C: joint-state success predicates.

Runtime ``joint_opened`` / ``joint_closed`` read ``env_state["joint_positions"]``;
authoring helpers lower into the composable success tree with self-describing refs
(no core-loop change). Validated locally against a synthetic env_state.
"""

import pytest

from robotsmith.cap.predicates import all_of, joint_closed, joint_opened
from robotsmith.cap.scene_api import layout, obj, region, scene
from robotsmith.cap.task_api import task
from robotsmith.cap.adapters import to_legacy_task_spec
from robotsmith.cap.validators import ValidationError, validate_task
from robotsmith.tasks.predicates import evaluate_predicate, evaluate_success


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
    def test_joint_opened_cfg(self):
        cfg = joint_opened("drawer", "drawer_slide", open_position=0.25)
        assert cfg.predicate == "joint_opened"
        assert cfg.params == {
            "object": "drawer",
            "joint": "drawer_slide",
            "threshold": 0.25,
        }
        assert cfg.refs == ("drawer",)

    def test_joint_closed_cfg_default(self):
        cfg = joint_closed("drawer", "drawer_slide")
        assert cfg.predicate == "joint_closed"
        assert cfg.params["threshold"] == 0.02
        assert cfg.refs == ("drawer",)


def _drawer_scene():
    return scene(
        "open_drawer",
        layout=layout(regions=[region("front", ((0.40, -0.10), (0.55, 0.10)))]),
        objects=[obj("drawer", asset="drawer_cabinet", fixed_position=(0.55, 0.0, 0.0))],
    )


class TestRoundTrip:
    def test_validate_and_lower_and_evaluate(self):
        cap_task = task(
            "open_drawer",
            scene=_drawer_scene(),
            instruction="Open the drawer",
            success=joint_opened("drawer", "drawer_slide", open_position=0.25),
        )
        validate_task(cap_task)  # refs=("drawer",) resolves to the scene object

        node = to_legacy_task_spec(cap_task).success
        assert evaluate_success(node, _env(0.30)) is True
        assert evaluate_success(node, _env(0.10)) is False

    def test_composite_open_and_closed_other(self):
        # all_of works the same with joint leaves (no core-loop change).
        spec = all_of(
            joint_opened("drawer", "drawer_slide", open_position=0.25),
            joint_closed("drawer", "drawer_slide", closed_position=0.40),
        )
        cap_task = task(
            "open_drawer_partial",
            scene=_drawer_scene(),
            instruction="Open the drawer partway",
            success=spec,
        )
        validate_task(cap_task)
        node = to_legacy_task_spec(cap_task).success
        assert evaluate_success(node, _env(0.30)) is True
        assert evaluate_success(node, _env(0.45)) is False

    def test_unknown_object_ref_rejected(self):
        cap_task = task(
            "bad_ref",
            scene=_drawer_scene(),
            instruction="Open the missing drawer",
            success=joint_opened("ghost", "drawer_slide", open_position=0.25),
        )
        with pytest.raises(ValidationError):
            validate_task(cap_task)
