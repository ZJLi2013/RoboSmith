"""Calibration set for object_in_box: what the simulator last saw decides ``on`` / ``inside``."""

import json

import pytest

from robotsmith.tasks.predicates import JudgeInputError, evaluate_success, object_in_box
from robotsmith.tasks.task_spec import SuccessNode

SHELF = [[-0.07, -0.085, 0.046], [0.07, 0.085, 0.054]]   # top at z 0.054 in the parent frame
FLOOR = [[-0.10, -0.10, -0.01], [0.10, 0.10, 0.00]]      # container floor, top at z 0
STILL = [[0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]] * 10


def _state(aabb, *, touching=("parent",), window=STILL, obj_pos=(0.0, 0.0, 0.1)):
    return {
        "object_positions": {"obj": list(obj_pos), "parent": [0.0, 0.0, 0.0]},
        "aabbs": {"obj": aabb},
        "contacts": {"obj": list(touching)},
        "window": {"steps": list(range(len(window))), "poses": {"obj": window}},
    }


def _judge(state, box=SHELF, mode="on"):
    return object_in_box(state, object="obj", parent="parent", box=box, yaw_deg=0.0, mode=mode)


@pytest.mark.parametrize("case, state, box, mode, expected", [
    ("block standing on its end", _state([[-0.008, -0.016, 0.054], [0.011, 0.016, 0.118]]), SHELF, "on", True),
    ("still held by the gripper", _state([[-0.01, -0.01, 0.054], [0.01, 0.01, 0.078]],
                                         touching=("parent", "robot")), SHELF, "on", False),
    ("fell to the table", _state([[0.2, 0.0, -0.10], [0.25, 0.05, -0.05]]), SHELF, "on", False),
    ("rocking with first and last frames equal",
     _state([[-0.01, -0.01, 0.054], [0.01, 0.01, 0.078]],
            window=[[x, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0] for x in (0, .01, .02, .01, 0, .01, .02, .01, .005, 0)]),
     SHELF, "on", False),
    ("not touching the parent", _state([[-0.01, -0.01, 0.054], [0.01, 0.01, 0.078]], touching=()), SHELF, "on", False),
    ("position off-centre from the AABB (inertial origin)",
     _state([[-0.03, -0.03, 0.054], [0.03, 0.03, 0.35]], obj_pos=(0.0, 0.0, 0.2)), SHELF, "on", True),
    ("in the drawer", _state([[-0.02, -0.02, 0.0], [0.02, 0.02, 0.024]]), FLOOR, "inside", True),
    ("bridging the drawer rim", _state([[-0.15, -0.03, 0.10], [0.15, 0.03, 0.30]]), FLOOR, "inside", False),
], ids=lambda v: v if isinstance(v, str) else "")
def test_object_in_box(case, state, box, mode, expected):
    assert _judge(state, box, mode) is expected


@pytest.mark.parametrize("broken", [
    lambda s: s["window"]["poses"].pop("obj"),
    lambda s: s["contacts"].pop("obj"),
    lambda s: s["aabbs"].update(obj=[[float("nan")] * 3, [0.0] * 3]),
], ids=["no window", "no contacts", "nan aabb"])
def test_missing_observation_raises(broken):
    state = _state([[-0.01, -0.01, 0.054], [0.01, 0.01, 0.078]])
    broken(state)
    with pytest.raises(JudgeInputError):
        _judge(state)


def test_judge_state_rejudges_from_json():
    state = _state([[-0.008, -0.016, 0.054], [0.011, 0.016, 0.118]])
    tree = SuccessNode.leaf("object_in_box", {"object": "obj", "parent": "parent", "box": SHELF,
                                              "yaw_deg": 0.0, "mode": "on"})
    recorded = evaluate_success(tree, state)
    replayed = json.loads(json.dumps({"tree": tree.to_dict(), "state": state}))
    assert evaluate_success(SuccessNode.from_dict(replayed["tree"]), replayed["state"]) is recorded is True
