"""Rollout wiring for asset-pick validation.

The CAP -> legacy adapter shapes (position_range, success leaf, skill dict)
are covered in test_cap_authoring. Here we only assert the rollout-specific
wiring in _make_scene / _make_task: the fixed "object" name and the
name-derived grasp category.
"""

from robotsmith.rollout.asset_pick_validation import _make_scene, _make_task


def test_make_scene_wires_single_named_object():
    scene = _make_scene("die_01")

    assert scene.name == "pick_die_01"
    assert len(scene.objects) == 1
    obj = scene.objects[0]
    assert obj.asset_query == "die_01"
    assert obj.name_override == "object"


def test_make_task_derives_category_from_asset_name():
    task = _make_task("apple_01")

    assert task.name == "pick_apple_01"
    assert task.success.predicate == "object_above"
    assert [skill.to_dict() for skill in task.skills] == [
        {"name": "pick", "target": "object", "params": {}, "category": "apple"},
    ]
