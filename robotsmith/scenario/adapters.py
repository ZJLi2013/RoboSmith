"""Per-step completion bindings for a list of plan Skills.

The scene, its target frames, the task, its SuccessNode tree and the plan's Skills are built once, by scenario.load.
"""

from __future__ import annotations

from robotsmith.scenario import relations
from robotsmith.scenario.relations import SolveError, place_target, split_ref
from robotsmith.skills import Skill
from robotsmith.skills.frames import resolve_drag_joint
from robotsmith.tasks.task_spec import SuccessNode

PICK_LIFT_M = 0.02
CONTACT_TOL_M = 0.02


def relation_params(mode: str, obj: str, ref: str, assets: dict, yaw_deg: float) -> dict:
    """``object_in_box`` params for ``obj`` on / inside the place target ``ref``."""
    parent, target = relations.split_ref(ref)
    if target is None:
        raise relations.SolveError(f"{mode}: '{parent}' must name a place target as <id>.<target>")
    spec = relations.place_target(assets[parent], target)
    params = {"object": obj, "parent": parent, "box": spec["box"], "target": target,
              "yaw_deg": yaw_deg, "mode": mode, "tol": CONTACT_TOL_M}
    if "joint" in spec:
        params["joint"] = spec["joint"]
        params["axis"] = list(assets[parent].get_joint(spec["joint"]).axis)
    return params


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


def _joint_binding(skill, assets: dict) -> dict:
    """``open`` / ``close`` bind to the joint the executor drags and the endpoint it drags to, within the
    task loader's tolerance: opened past ``end - tol``, closed below ``lower + tol``."""
    asset = assets.get(skill.target)
    if asset is None:
        raise SolveError(f"plan step {skill.name} {skill.target!r} has no asset")
    drag = resolve_drag_joint(asset, skill.name, dict(skill.params))
    if skill.name == "open":
        predicate, threshold = "joint_opened", drag.end - CONTACT_TOL_M
    else:
        predicate, threshold = "joint_closed", drag.lower + CONTACT_TOL_M
    return {"status": "bound", "predicate": predicate,
            "params": {"object": skill.target, "joint": drag.joint.name, "threshold": threshold}}


def to_step_bindings(skills: list[Skill], success: SuccessNode,
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
    for skill in skills:
        if skill.name == "pick":
            held = skill.target
        elif skill.name == "place" and held is not None:
            place_count[held] = place_count.get(held, 0) + 1
            held = None
    held = None
    for i, skill in enumerate(skills):
        if skill.name == "pick":
            held = skill.target
            bindings.append({"status": "bound", "predicate": "grasped",
                             "params": {"object": skill.target, "lift": PICK_LIFT_M}})
            continue
        if skill.name in ("open", "close"):
            bindings.append(_joint_binding(skill, assets))
            continue
        if skill.name != "place":
            bindings.append(_unchecked("not migrated"))
            continue
        obj, held = held, None
        parent, target = split_ref(skill.target)
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
                raise SolveError(f"plan step {i} places {obj} at {skill.target}, but success wants "
                                 f"{rel['mode']} {rel['parent']}.{rel.get('target')}")
            bindings.append({"status": "bound", "predicate": "object_in_box", "params": rel})
    return bindings


def bindings_from_steps(skills: list[Skill], assets: dict, yaw_deg: dict[str, float],
                        held: str | None = None) -> list[dict]:
    """Bind each step to its own completion condition, without reading the success tree: a ``place`` binds to
    the object held at that point being on its target (inside when the target opens on a joint). ``held`` is
    the object already in hand when the steps start."""
    bindings = []
    for skill in skills:
        if skill.name == "pick":
            held = skill.target
            bindings.append({"status": "bound", "predicate": "grasped",
                             "params": {"object": skill.target, "lift": PICK_LIFT_M}})
        elif skill.name in ("open", "close"):
            bindings.append(_joint_binding(skill, assets))
        elif skill.name == "place" and held is not None:
            parent, target = split_ref(skill.target)
            mode = "inside" if "joint" in place_target(assets[parent], target) else "on"
            bindings.append({"status": "bound", "predicate": "object_in_box",
                             "params": relation_params(mode, held, skill.target, assets, yaw_deg.get(parent, 0.0))})
            held = None
        else:
            bindings.append(_unchecked("no preceding pick"))
    return bindings
