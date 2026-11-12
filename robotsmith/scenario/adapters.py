"""Adapters from scenario plan steps to the current RoboSmith runtime: Skills and per-step bindings.

The scene, its target frames, the task and its SuccessNode tree are built once, by scenario.load.
"""

from __future__ import annotations

from robotsmith.scenario.intents import SkillIntentSequenceSpec
from robotsmith.scenario.load import CONTACT_TOL_M, relation_params
from robotsmith.scenario.relations import SolveError, place_target, split_ref
from robotsmith.scenario.validators import validate_skill_intent_sequence
from robotsmith.skills import Skill
from robotsmith.skills.frames import resolve_drag_joint
from robotsmith.scenes.config import SceneConfig
from robotsmith.tasks.task_spec import SuccessNode

PICK_LIFT_M = 0.02


def to_legacy_skills(
    intents: SkillIntentSequenceSpec,
    scene: SceneConfig,
    target_names,
) -> list[Skill]:
    """Lower scenario task-level skill intents to current runtime Skills.

    This is the Feature 2 boundary: scenario intent selects an existing scripted
    expert primitive, while planner, IK, MotionExecutor, and recorder stay in
    the runtime.
    """

    validate_skill_intent_sequence(intents, scene, target_names)
    return [
        Skill(
            name=intent.name,
            target=intent.target,
            category=intent.category,
            params=dict(intent.params),
        )
        for intent in intents.intents
    ]


def _placement_relations(node: SuccessNode) -> tuple[list[dict], bool]:
    """Positive on / inside relations of a success tree, and whether the tree is a pure conjunction."""
    if node.op == "leaf":
        return ([dict(node.params)] if node.predicate == "object_in_box" else []), True
    if node.op == "all":
        found, conj = [], True
        for term in node.terms:
            rels, term_conj = _placement_relations(term)
            found += rels
            conj &= term_conj
        return found, conj
    return [], False


def _unchecked(reason: str) -> dict:
    return {"status": "unchecked", "reason": reason}


def _joint_binding(intent, assets: dict) -> dict:
    """``open`` / ``close`` bind to the joint the executor drags and the endpoint it drags to, within the
    task loader's tolerance: opened past ``end - tol``, closed below ``lower + tol``."""
    asset = assets.get(intent.target)
    if asset is None:
        raise SolveError(f"plan step {intent.name} {intent.target!r} has no asset")
    drag = resolve_drag_joint(asset, intent.name, dict(intent.params))
    if intent.name == "open":
        predicate, threshold = "joint_opened", drag.end - CONTACT_TOL_M
    else:
        predicate, threshold = "joint_closed", drag.lower + CONTACT_TOL_M
    return {"status": "bound", "predicate": predicate,
            "params": {"object": intent.target, "joint": drag.joint.name, "threshold": threshold}}


def to_step_bindings(intents: SkillIntentSequenceSpec, success: SuccessNode,
                     assets: dict | None = None) -> list[dict]:
    """Bind each plan step to the completion condition it can be judged by, one entry per step.

    A ``pick`` binds to ``grasped``: both fingers on its target and a lift of ``PICK_LIFT_M`` from the step's
    start. ``open`` / ``close`` bind to their joint through ``_joint_binding`` (``assets`` maps scene object
    names to assets). A ``place`` binds when the object placed is the one the preceding ``pick`` took and
    the success tree has a single positive on / inside relation for that object. A relation whose parent
    or target disagrees with the place raises ``SolveError``; trees with ``any_of`` / ``not``, an object
    placed twice, or several candidate relations stay ``unchecked`` rather than guessed.
    """
    assets = assets or {}
    relations, conjunction = _placement_relations(success)
    bindings, held, place_count = [], None, {}
    for intent in intents.intents:
        if intent.name == "pick":
            held = intent.target
        elif intent.name == "place" and held is not None:
            place_count[held] = place_count.get(held, 0) + 1
            held = None
    held = None
    for i, intent in enumerate(intents.intents):
        if intent.name == "pick":
            held = intent.target
            bindings.append({"status": "bound", "predicate": "grasped",
                             "params": {"object": intent.target, "lift": PICK_LIFT_M}})
            continue
        if intent.name in ("open", "close"):
            bindings.append(_joint_binding(intent, assets))
            continue
        if intent.name != "place":
            bindings.append(_unchecked("not migrated"))
            continue
        obj, held = held, None
        parent, target = split_ref(intent.target)
        candidates = [r for r in relations if r["object"] == obj]
        if obj is None:
            bindings.append(_unchecked("no preceding pick"))
        elif not conjunction:
            bindings.append(_unchecked("success tree is not a conjunction"))
        elif place_count.get(obj, 0) > 1:
            bindings.append(_unchecked("object placed more than once"))
        elif len(candidates) != 1:
            bindings.append(_unchecked(f"{len(candidates)} success relations for {obj}"))
        else:
            rel = candidates[0]
            if (rel["parent"], rel.get("target")) != (parent, target):
                raise SolveError(f"plan step {i} places {obj} at {intent.target}, but success wants "
                                 f"{rel['mode']} {rel['parent']}.{rel.get('target')}")
            bindings.append({"status": "bound", "predicate": "object_in_box", "params": rel})
    return bindings


def bindings_from_steps(intents: SkillIntentSequenceSpec, assets: dict, yaw_deg: dict[str, float],
                        held: str | None = None) -> list[dict]:
    """Bind each step to its own completion condition, without reading the success tree: a ``place`` binds to
    the object held at that point being on its target (inside when the target opens on a joint). ``held`` is
    the object already in hand when the steps start."""
    bindings = []
    for intent in intents.intents:
        if intent.name == "pick":
            held = intent.target
            bindings.append({"status": "bound", "predicate": "grasped",
                             "params": {"object": intent.target, "lift": PICK_LIFT_M}})
        elif intent.name in ("open", "close"):
            bindings.append(_joint_binding(intent, assets))
        elif intent.name == "place" and held is not None:
            parent, target = split_ref(intent.target)
            mode = "inside" if "joint" in place_target(assets[parent], target) else "on"
            bindings.append({"status": "bound", "predicate": "object_in_box",
                             "params": relation_params(mode, held, intent.target, assets, yaw_deg.get(parent, 0.0))})
            held = None
        else:
            bindings.append(_unchecked("no preceding pick"))
    return bindings
