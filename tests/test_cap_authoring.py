import numpy as np
import pytest

from robotsmith.cap.adapters import (
    resolved_target_positions,
    to_legacy_scene_config,
    to_legacy_skills,
    to_legacy_task_spec,
)
from robotsmith.cap.intents import (
    SkillIntentSpec,
    intent_sequence,
    pick,
    place,
)
from robotsmith.cap.predicates import (
    all_of,
    negate,
    object_in_container,
    object_lifted,
    objects_aligned,
    stacked,
)
from robotsmith.cap.scene_api import layout, obj, region, scene, target_position
from robotsmith.cap.task_api import task
from robotsmith.cap.validators import (
    ValidationError,
    validate_scene,
    validate_skill_intent_sequence,
    validate_task,
)
from robotsmith.tasks.predicates import evaluate_predicate, evaluate_success
from robotsmith.tasks.task_spec import SuccessNode


def test_cap_scene_lowers_part6_regions_to_legacy_scene_config():
    cap_scene = scene(
        "cap_align_die_mug_apple",
        layout=layout(
            regions=[
                region(
                    "left_reachable",
                    ((0.42, -0.14), (0.50, -0.06)),
                    keep_out_radius=0.04,
                ),
                region("reachable_center", ((0.50, -0.05), (0.60, 0.05)), z=0.800),
            ],
        ),
        objects=[
            obj("die", asset="die_01", region="left_reachable"),
            obj("mug", asset="mug_02", fixed_position=(0.58, 0.10, 0.809)),
            obj("apple", asset="apple_01", region="reachable_center"),
        ],
    )

    validate_scene(cap_scene)
    legacy = to_legacy_scene_config(cap_scene)

    assert legacy.name == "cap_align_die_mug_apple"
    assert [o.name_override for o in legacy.objects] == ["die", "mug", "apple"]
    assert legacy.objects[0].asset_query == "die_01"
    assert legacy.objects[0].position_range == [[0.42, -0.14, 0.0], [0.50, -0.06, 0.0]]
    assert legacy.objects[1].fixed_position == [0.58, 0.10, 0.809]


def test_cap_task_lowers_success_and_contact_objects_to_legacy_task_spec():
    cap_scene = scene(
        "cap_pick_die",
        layout=layout(
            regions=[
                region("left_reachable", ((0.42, -0.14), (0.50, -0.06))),
            ],
        ),
        objects=[obj("die", asset="die_01", region="left_reachable")],
    )
    cap_task = task(
        "cap_pick_die",
        scene=cap_scene,
        instruction="Pick up the die",
        success=object_lifted("die", z_margin=0.05),
    )

    validate_task(cap_task)
    legacy = to_legacy_task_spec(cap_task)

    assert legacy.name == "cap_pick_die"
    assert legacy.scene == "cap_pick_die"
    assert legacy.skills == []
    assert legacy.success.op == "leaf"
    assert legacy.success.predicate == "object_above"
    assert legacy.success.params == {
        "object": "die",
        "reference": "table",
        "z_margin": 0.05,
    }
    assert legacy.contact_objects == ["die", "table"]


def _place_die_at_goal_task():
    cap_scene = scene(
        "cap_place_die_at_goal",
        layout=layout(regions=[region("left_reachable", ((0.42, -0.14), (0.50, -0.06)))]),
        objects=[obj("die", asset="die_01", region="left_reachable")],
        target_positions=[target_position("goal", fixed_position=(0.60, 0.10, 0.809))],
    )
    return task(
        "cap_place_die_at_goal",
        scene=cap_scene,
        instruction="Place the die at the goal",
        success=object_in_container("die", "goal", xy_threshold=0.02),
    )


def test_resolved_target_positions_resolves_fixed_targets():
    cap_scene = _place_die_at_goal_task().scene

    validate_scene(cap_scene)
    assert resolved_target_positions(cap_scene) == {"goal": (0.60, 0.10, 0.809)}


def test_object_in_container_keeps_z_check_for_physical_containers():
    env_state = {
        "object_positions": {
            "die": np.array([0.60, 0.10, 0.798]),
            "bin": np.array([0.60, 0.10, 0.809]),
        },
        "target_positions": {},
    }

    assert not evaluate_predicate(
        "object_in_container",
        env_state,
        {"object": "die", "container": "bin", "xy_threshold": 0.02},
    )


def test_validator_rejects_unknown_region_and_unknown_success_ref():
    base_layout = layout(
        regions=[
            region(
                "left_reachable",
                ((0.42, -0.14), (0.50, -0.06)),
                keep_out_radius=0.04,
            ),
            region("right_reachable", ((0.56, 0.06), (0.64, 0.14)), z=0.809),
            region("reachable_center", ((0.50, -0.05), (0.60, 0.05)), z=0.800),
        ],
    )
    bad_scene = scene(
        "bad_region",
        layout=base_layout,
        objects=[obj("die", asset="die_01", region="missing_region")],
    )
    with pytest.raises(ValidationError, match="unknown region"):
        validate_scene(bad_scene)

    cap_scene = scene(
        "bad_ref",
        layout=base_layout,
        objects=[obj("die", asset="die_01", region="left_reachable")],
    )
    bad_task = task(
        "bad_ref",
        scene=cap_scene,
        instruction="Pick up the mug",
        success=object_lifted("mug"),
    )
    with pytest.raises(ValidationError, match="unknown names"):
        validate_task(bad_task)


def test_skill_intents_lower_to_legacy_runtime_skills():
    cap_task = _place_die_at_goal_task()
    intents = intent_sequence([
        pick("die", category="die"),
        place("goal", category="die"),
    ])

    validate_skill_intent_sequence(intents, cap_task)
    legacy_task = to_legacy_task_spec(cap_task)
    legacy_skills = to_legacy_skills(intents, cap_task)

    assert legacy_task.skills == []
    assert [skill.to_dict() for skill in legacy_skills] == [
        {"name": "pick", "target": "die", "params": {}, "category": "die"},
        {
            "name": "place",
            "target": "goal",
            "params": {},
            "category": "die",
        },
    ]


def test_skill_intent_validator_rejects_invalid_refs_and_names():
    cap_task = _place_die_at_goal_task()

    with pytest.raises(ValidationError, match="unknown object 'goal'"):
        validate_skill_intent_sequence(intent_sequence([pick("goal")]), cap_task)

    with pytest.raises(ValidationError, match="unknown object or target 'missing'"):
        validate_skill_intent_sequence(intent_sequence([place("missing")]), cap_task)

    with pytest.raises(ValidationError, match="unknown skill intent"):
        validate_skill_intent_sequence(
            intent_sequence([SkillIntentSpec(name="move_to", target="die")]),
            cap_task,
        )


def _two_fruit_in_bowl_task():
    cap_scene = scene(
        "cap_two_fruit_in_bowl",
        layout=layout(
            regions=[region("reachable", ((0.42, -0.14), (0.60, 0.06)))],
        ),
        objects=[
            obj("apple", asset="apple_01", region="reachable"),
            obj("banana", asset="banana_01", region="reachable"),
        ],
        target_positions=[target_position("bowl", fixed_position=(0.55, 0.0, 0.809))],
    )
    return task(
        "cap_two_fruit_in_bowl",
        scene=cap_scene,
        instruction="Put both fruit in the bowl",
        success=all_of(
            object_in_container("apple", "bowl"),
            object_in_container("banana", "bowl"),
        ),
    )


def test_composite_success_lowers_to_runtime_tree_and_unions_refs():
    cap_task = _two_fruit_in_bowl_task()

    validate_task(cap_task)
    legacy = to_legacy_task_spec(cap_task)

    assert legacy.success.op == "all"
    assert [t.predicate for t in legacy.success.terms] == [
        "object_in_container",
        "object_in_container",
    ]
    # contact_objects unions refs across all leaves (+ table), de-duplicated.
    assert legacy.contact_objects == ["apple", "bowl", "banana", "table"]


def test_predicate_helpers_self_describe_refs():
    # Each leaf carries its own refs (Feature 5), replacing the old
    # name-based predicate_refs switch. Values must match the legacy mapping.
    assert object_lifted("die").refs == ("die", "table")
    assert object_in_container("die", "goal").refs == ("die", "goal")
    assert objects_aligned(["die", "mug", "apple"]).refs == ("die", "mug", "apple")
    assert stacked(["block_a", "block_b"]).refs == ("block_a", "block_b")


def test_evaluate_success_walks_all_any_not():
    inside = {
        "object_positions": {"apple": np.array([0.55, 0.0, 0.81])},
        "target_positions": {"bowl": np.array([0.55, 0.0, 0.809])},
    }
    outside = {
        "object_positions": {"apple": np.array([0.20, 0.0, 0.81])},
        "target_positions": {"bowl": np.array([0.55, 0.0, 0.809])},
    }
    leaf = SuccessNode.leaf(
        "object_in_container",
        {"object": "apple", "container": "bowl", "xy_threshold": 0.05},
    )

    assert evaluate_success(leaf, inside) is True
    assert evaluate_success(leaf, outside) is False
    assert evaluate_success(SuccessNode(op="not", terms=[leaf]), outside) is True
    assert evaluate_success(SuccessNode(op="all", terms=[leaf, leaf]), outside) is False
    assert evaluate_success(SuccessNode(op="any", terms=[leaf]), inside) is True


def test_success_node_to_dict_from_dict_roundtrip():
    node = SuccessNode(
        op="all",
        terms=[
            SuccessNode.leaf("object_in_container", {"object": "apple", "container": "bowl"}),
            SuccessNode(
                op="not",
                terms=[SuccessNode.leaf("object_above", {"object": "banana", "reference": "table"})],
            ),
        ],
    )

    restored = SuccessNode.from_dict(node.to_dict())

    assert restored == node


def test_validator_rejects_empty_composite_terms():
    cap_task = _two_fruit_in_bowl_task()
    bad_task = task(
        cap_task.name,
        scene=cap_task.scene,
        instruction=cap_task.instruction,
        success=all_of(),
    )
    with pytest.raises(ValidationError, match="requires at least one term"):
        validate_task(bad_task)


def test_negate_composes_with_leaf():
    cap_task = _two_fruit_in_bowl_task()
    negated = task(
        cap_task.name,
        scene=cap_task.scene,
        instruction=cap_task.instruction,
        success=negate(object_in_container("apple", "bowl")),
    )
    legacy = to_legacy_task_spec(negated)
    assert legacy.success.op == "not"
    assert legacy.success.terms[0].predicate == "object_in_container"
