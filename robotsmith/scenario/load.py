"""Load a YAML scenario into one ``Scenario``: validate, apply variations, solve relations, build the runtime
scene, task, plan skills and target frames.

A scenario file holds the scene (objects and their relations) and the task
(instruction + success tree, hidden from the executor). The scene is written
inline or names a layout ``scenes/<name>.yaml`` beside the scenarios directory,
so several tasks can share one layout. The optional plan file lives next to it in
``plans/<same name>.yaml``: the skill steps ``--planner file`` runs, a pick optionally with its lift.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import jsonschema
import yaml

from robotsmith.assets import store
from robotsmith.assets.library import AssetLibrary
from robotsmith.scenario import relations
from robotsmith.scenario.adapters import CONTACT_TOL_M, relation_params, to_step_bindings
from robotsmith.scenario.predicates import (
    joint_closed,
    joint_opened,
    object_lifted,
    objects_aligned,
    stacked,
)
from robotsmith.scenario.specs import FrameRef
from robotsmith.scenes.config import ObjectPlacement, SceneConfig
from robotsmith.skills import Skill
from robotsmith.tasks.predicates import PREDICATE_REGISTRY
from robotsmith.tasks.task_spec import SuccessNode, TaskSpec

DEFAULT_ASSETS = Path(__file__).resolve().parents[2] / "assets"


class _Loader(yaml.SafeLoader):
    """YAML 1.2 booleans: only true / false, so the ``on`` key stays a string."""


_Loader.yaml_implicit_resolvers = {
    ch: [(tag, rx) for tag, rx in resolvers if tag != "tag:yaml.org,2002:bool"]
    for ch, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def _read(path: Path):
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_Loader)

_ID = {"type": "string", "pattern": r"^[A-Za-z_]\w*$"}
_REF = {"type": "string", "pattern": r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)?$"}
_SEEDS = {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}
_RANGE = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}


def _leaf(name: str, props: dict, required: list[str]) -> dict:
    return {
        "type": "object", "additionalProperties": False, "required": [name],
        "properties": {name: {"type": "object", "additionalProperties": False,
                              "required": required, "properties": props}},
    }


_SCENE_PROPS = {
    "support": {"enum": ["table_simple"]},
    "objects": {"type": "array", "minItems": 1, "items": {
        "type": "object", "additionalProperties": False, "required": ["id", "asset", "on"],
        "properties": {
            "id": _ID,
            "asset": {"type": "string"},
            "on": _REF,
            "anchor_xy": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            "near": {"enum": ["robot"]},
            "offset_from": {
                "type": "object", "additionalProperties": False, "required": ["of", "x", "y"],
                "properties": {"of": _ID, "x": _RANGE, "y": _RANGE},
            },
            "clear_of": {"type": "array", "items": {"type": "string"}},
            "facing": {"enum": ["robot"]},
            "yaw_deg": {"type": "number"},
            "joint_init": {"type": "object", "additionalProperties": {"type": "number"}},
        },
    }},
}

LAYOUT_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["scene", "support", "objects"],
    "properties": {"scene": _ID, **_SCENE_PROPS},
}

SCHEMA = {
    "$defs": {
        "success": {"oneOf": [
            {"type": "object", "additionalProperties": False, "required": ["all_of"],
             "properties": {"all_of": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/success"}}}},
            {"type": "object", "additionalProperties": False, "required": ["any_of"],
             "properties": {"any_of": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/success"}}}},
            {"type": "object", "additionalProperties": False, "required": ["not"],
             "properties": {"not": {"$ref": "#/$defs/success"}}},
            _leaf("inside", {"object": _ID, "container": _REF}, ["object", "container"]),
            _leaf("on", {"object": _ID, "surface": _REF}, ["object", "surface"]),
            _leaf("lifted", {"object": _ID, "height": {"type": "number"}}, ["object", "height"]),
            _leaf("joint", {"object": _ID, "joint": {"type": "string"},
                            "at": {"enum": ["open", "closed"]}, "tol": {"type": "number"}},
                  ["object", "joint", "at"]),
            _leaf("stacked", {"objects": {"type": "array", "items": _ID},
                              "z_tolerance": {"type": "number"}}, ["objects"]),
            _leaf("aligned", {"objects": {"type": "array", "items": _ID}, "axis": {"enum": ["x", "y"]},
                              "xy_threshold": {"type": "number"}}, ["objects"]),
        ]},
    },
    "type": "object",
    "additionalProperties": False,
    "required": ["scenario", "robot", "scene", "task"],
    "properties": {
        "scenario": _ID,
        "robot": {"enum": ["franka"]},
        "scene": {"oneOf": [
            _ID,
            {"type": "object", "additionalProperties": False, "required": ["support", "objects"],
             "properties": _SCENE_PROPS},
        ]},
        "task": {
            "type": "object", "additionalProperties": False, "required": ["instruction", "success"],
            "properties": {"instruction": {"type": "string"}, "success": {"$ref": "#/$defs/success"}},
        },
        "variations": {"type": "object", "additionalProperties": {"type": "array", "minItems": 1}},
        "split": {"type": "object", "additionalProperties": False,
                  "properties": {"playground": _SEEDS, "test": _SEEDS}},
    },
}

_LINE = {"type": "object", "additionalProperties": False, "required": ["direction", "dist"],
         "properties": {"direction": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                        "dist": {"type": "number", "exclusiveMinimum": 0}}}

PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "steps"],
    "properties": {
        "name": _ID,
        "steps": {"type": "array", "minItems": 1, "items": {"oneOf": [
            {"type": "object", "additionalProperties": False, "required": ["pick"],
             "properties": {"pick": _ID, "lift": _LINE}},
        ] + [
            {"type": "object", "additionalProperties": False, "required": [k], "properties": {k: _ID}}
            for k in ("open", "close")
        ] + [_leaf("place", {"target": _REF}, ["target"])]}},
    },
}


class PlanInvalid(ValueError):
    """Steps that cannot run from the hand they start with: a pick while holding, or a place with nothing held."""


class ValidationError(ValueError):
    """Raised when a scenario authoring spec is invalid."""


def check_object_ids(ids: list[str]) -> None:
    _require_unique(ids, "object names")


def check_task(task: TaskSpec, scene: SceneConfig, target_names) -> None:
    """A task's name and instruction are non-empty, and its success tree names registered predicates and only
    objects of ``scene``, ``target_names`` or the table."""

    _require(task.name.strip(), "TaskSpec.name must be non-empty")
    _require(task.instruction.strip(), "TaskSpec.instruction must be non-empty")
    _validate_success(task.success)

    allowed_refs = {o.name_override for o in scene.objects} | set(target_names) | {"table"}
    missing = [ref for ref in _refs(task.success) if ref not in allowed_refs]
    _require(not missing, f"success predicate references unknown names: {missing}")


def _validate_success(node: SuccessNode) -> None:
    if node.op == "leaf":
        _require(node.predicate in PREDICATE_REGISTRY, f"unknown success predicate '{node.predicate}'")
        return
    _require(node.op in ("all", "any", "not"), f"unsupported success op: {node.op!r}")
    _require(len(node.terms) == 1 if node.op == "not" else node.terms,
             f"{node.op} requires {'one term' if node.op == 'not' else 'at least one term'}")
    for term in node.terms:
        _validate_success(term)


def _refs(node: SuccessNode) -> list[str]:
    """Scene names a success tree reads: each leaf's ``object`` / ``objects`` / ``parent`` / ``reference`` params."""

    if node.op != "leaf":
        return [ref for term in node.terms for ref in _refs(term)]
    p = node.params
    return ([p["object"]] if "object" in p else []) + list(p.get("objects", ())) + [
        p[k] for k in ("parent", "reference") if k in p]


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _require_unique(values: list[str], label: str) -> None:
    duplicates = sorted({v for v in values if values.count(v) > 1})
    _require(not duplicates, f"duplicate {label}: {duplicates}")


@dataclass(frozen=True)
class Scenario:
    """One scenario at one seed, built once before ``SimEnv.build`` and not written afterwards.

    Plan mode: ``task`` carries the plan file's skills and their step bindings, ``target_frames`` the place targets
    those steps use. Agent mode: ``task`` has no skills or bindings and ``target_frames`` holds every place target."""

    source_path: Path
    scene: SceneConfig
    task: TaskSpec
    target_frames: dict[str, FrameRef]


def load_scenario(
    path: str | Path,
    *,
    seed: int,
    agent_plan: bool = False,
    library: AssetLibrary | None = None,
) -> Scenario:
    """The scenario at ``seed``; the plan file is read and checked in both modes, and used only in plan mode."""
    path = Path(path)
    data = _read(path)
    jsonschema.validate(data, SCHEMA)
    library = library or AssetLibrary(DEFAULT_ASSETS)
    rng = random.Random(seed)

    scene_data = data["scene"]
    if isinstance(scene_data, str):
        scene_data = _read(path.parent.parent / "scenes" / f"{scene_data}.yaml")
        jsonschema.validate(scene_data, LAYOUT_SCHEMA)
    objects = [dict(o) for o in scene_data["objects"]]
    check_object_ids([o["id"] for o in objects])
    by_id = {o["id"]: o for o in objects}
    for key, choices in data.get("variations", {}).items():
        obj_id, _, field = key.partition(".")
        if obj_id not in by_id or not field:
            raise relations.SolveError(f"variation '{key}' must be <object id>.<field>")
        by_id[obj_id][field] = rng.choice(choices)
    for name in sorted({o["asset"] for o in objects}):
        asset = library.get(name)
        reasons = store.draft_reasons(asset.root_dir) if asset else []
        if reasons:
            raise store.AssetNotAdmitted(f"{name}: {'; '.join(reasons)}")

    solved = relations.solve(objects, library, rng)
    assets = {o["id"]: library.get(o["asset"]) for o in objects}

    plan_file = path.parent / "plans" / path.name
    skills, targets = None, {}
    if plan_file.exists():
        skills, targets = _plan_file(plan_file, assets)

    scene = SceneConfig(
        name=data["scenario"],
        objects=[
            ObjectPlacement(
                asset_query=o["asset"],
                fixed_position=list(solved.placements[o["id"]].position),
                name_override=o["id"],
                joint_init=dict(o["joint_init"]) if o.get("joint_init") else None,
                yaw_deg=solved.placements[o["id"]].yaw_deg,
            )
            for o in objects
        ],
        camera_position=list(solved.camera_position),
        camera_target=list(solved.camera_target),
        table_size=list(solved.table_size),
    )
    task = TaskSpec(
        name=data["scenario"],
        scene=scene.name,
        instruction=data["task"]["instruction"],
        success=_success(data["task"]["success"], assets, solved.placements),
    )
    check_task(task, scene, targets)
    if agent_plan:
        targets = all_place_target_frames(assets)
    elif skills is not None:
        task.skills = skills
        task.step_bindings = to_step_bindings(skills, task.success, assets)
    return Scenario(source_path=path, scene=scene, task=task, target_frames=targets)


def read_plan_file(path: Path) -> dict:
    """The plan file; the only reader of plan files. Reads only."""
    return _read(Path(path))


def _plan_file(path: Path, assets: dict) -> tuple[list[Skill], dict[str, FrameRef]]:
    data = read_plan_file(path)
    jsonschema.validate(data, PLAN_SCHEMA)
    return steps_to_skills(data["steps"], assets)


def steps_to_skills(steps: list[dict], assets: dict, *, held: str | None = None
                    ) -> tuple[list[Skill], dict[str, FrameRef]]:
    """Steps -> skills, plus a frame for every place target they use; the one check of a step list (plan file and
    agent reply), and it reads nothing but its arguments.

    ``held`` is the object in hand when the steps start. A pick needs an empty hand and a place a held object; the
    steps may end holding. A pick's ``lift`` goes into its params."""
    jsonschema.validate(steps, PLAN_SCHEMA["properties"]["steps"])
    skills, targets = [], {}
    for i, step in enumerate(steps):
        name, arg = next((k, v) for k, v in step.items() if k != "lift")
        if name == "place":
            ref = arg["target"]
            try:
                _check_place_ref(ref, assets, targets)
            except relations.SolveError as e:
                raise PlanInvalid(f"step {i}: {e}") from e
            if held is None:
                raise PlanInvalid(f"step {i}: place {ref} with nothing held")
            skills.append(Skill(name="place", target=ref))
            held = None
            continue
        if arg not in assets:
            raise PlanInvalid(f"step {i}: {name}: unknown object '{arg}'")
        if name == "pick":
            if held is not None:
                raise PlanInvalid(f"step {i}: pick {arg} while holding {held}")
            held = arg
        skills.append(Skill(name=name, target=arg, params={"lift": step["lift"]} if "lift" in step else {}))
    return skills, targets


def _check_place_ref(ref: str, assets: dict, targets: dict) -> None:
    parent, target = relations.split_ref(ref)
    if parent not in assets:
        raise relations.SolveError(f"place: unknown object '{parent}'")
    if target is not None:
        targets[ref] = place_target_frame(ref, assets)


def all_place_target_frames(assets: dict) -> dict[str, FrameRef]:
    """A frame for every place target of every scene object, for steps not known at load."""
    return {f"{obj}.{t['name']}": place_target_frame(f"{obj}.{t['name']}", assets)
            for obj, asset in assets.items() for t in asset.metadata.place_targets}


def place_target_frame(ref: str, assets: dict) -> FrameRef:
    parent, target = relations.split_ref(ref)
    spec = relations.place_target(assets[parent], target)
    kind = "articulated_opening" if "joint" in spec else "placement"
    return FrameRef(kind=kind, parent=parent, opening=target)


def _success(node: dict, assets: dict, placements: dict) -> SuccessNode:
    (key, val), = node.items()
    if key in ("all_of", "any_of"):
        return SuccessNode(op=key[:3], terms=[_success(t, assets, placements) for t in val])
    if key == "not":
        return SuccessNode(op="not", terms=[_success(val, assets, placements)])
    if key in ("inside", "on"):
        obj = val["object"]
        ref = val["container" if key == "inside" else "surface"]
        parent, _ = relations.split_ref(ref)
        params = relation_params(key, obj, ref, assets, placements[parent].yaw_deg)
        return SuccessNode.leaf("object_in_box", params)
    if key == "lifted":
        return object_lifted(val["object"], z_margin=val["height"])
    if key == "joint":
        joint = assets[val["object"]].get_joint(val["joint"])
        tol = val.get("tol", CONTACT_TOL_M)
        if val["at"] == "closed":
            return joint_closed(val["object"], val["joint"], closed_position=(joint.lower or 0.0) + tol)
        return joint_opened(val["object"], val["joint"], open_position=joint.upper - tol)
    if key == "stacked":
        return stacked(val["objects"], **{k: v for k, v in val.items() if k != "objects"})
    return objects_aligned(val["objects"], **{k: v for k, v in val.items() if k != "objects"})
