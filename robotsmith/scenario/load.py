"""Load a YAML scenario: validate, apply variations, solve relations, lower to specs.

A scenario file holds the scene (objects and their relations) and the task
(instruction + success tree, hidden from the executor). The scene is written
inline or names a layout ``scenes/<name>.yaml`` beside the scenarios directory,
so several tasks can share one layout. The optional teacher plan for scripted
rollouts lives next to it in ``teacher/<same name>.yaml``.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import jsonschema
import yaml

from robotsmith.assets import derive, store
from robotsmith.assets.library import AssetLibrary
from robotsmith.scenario import relations
from robotsmith.scenario.intents import SkillIntentSequenceSpec, SkillIntentSpec
from robotsmith.scenario.predicates import (
    joint_closed,
    joint_opened,
    object_lifted,
    objects_aligned,
    stacked,
)
from robotsmith.scenario.specs import (
    AllOf,
    AnyOf,
    FrameRef,
    LayoutSpec,
    Not,
    ObjectSpec,
    SceneSpec,
    SuccessSpec,
    TargetPositionSpec,
    TaskSpec,
    TaskSuccessCfg,
)

DEFAULT_ASSETS = Path(__file__).resolve().parents[2] / "assets"
CONTACT_TOL_M = 0.02


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

TEACHER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["teacher", "steps"],
    "properties": {
        "teacher": _ID,
        "steps": {"type": "array", "minItems": 1, "items": {"oneOf": [
            {"type": "object", "additionalProperties": False, "required": [k], "properties": {k: _ID}}
            for k in ("pick", "open", "close")
        ] + [_leaf("place", {"target": _REF}, ["target"])]}},
    },
}


def load_scenario(
    path: str | Path,
    *,
    seed: int,
    library: AssetLibrary | None = None,
) -> tuple[SceneSpec, TaskSpec, SkillIntentSequenceSpec | None]:
    """Scene, task and teacher intents (None without a teacher file) for one seed."""
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

    teacher = path.parent / "teacher" / path.name
    intents, targets = None, {}
    if teacher.exists():
        intents, targets = _teacher(teacher, assets)

    scene = SceneSpec(
        name=data["scenario"],
        layout=LayoutSpec(),
        objects=tuple(
            ObjectSpec(
                name=o["id"],
                asset=o["asset"],
                fixed_position=solved.placements[o["id"]].position,
                yaw_deg=solved.placements[o["id"]].yaw_deg,
                joint_init=dict(o.get("joint_init", {})),
            )
            for o in objects
        ),
        target_positions=tuple(targets.values()),
        camera_position=solved.camera_position,
        camera_target=solved.camera_target,
        table_size=solved.table_size,
    )
    task = TaskSpec(
        name=data["scenario"],
        scene=scene,
        instruction=data["task"]["instruction"],
        success=_success(data["task"]["success"], assets, solved.placements),
    )
    return scene, task, intents


def _teacher(path: Path, assets: dict) -> tuple[SkillIntentSequenceSpec, dict[str, TargetPositionSpec]]:
    data = _read(path)
    jsonschema.validate(data, TEACHER_SCHEMA)
    intents, targets = [], {}
    for step in data["steps"]:
        (name, arg), = step.items()
        if name != "place":
            intents.append(SkillIntentSpec(name=name, target=arg))
            continue
        ref = arg["target"]
        parent, target = relations.split_ref(ref)
        if target is not None:
            spec = relations.place_target(assets[parent], target)
            kind = "articulated_opening" if "joint" in spec else "placement"
            targets[ref] = TargetPositionSpec(name=ref, anchor=FrameRef(kind=kind, parent=parent, opening=target))
        intents.append(SkillIntentSpec(name="place", target=ref))
    return SkillIntentSequenceSpec(intents=tuple(intents)), targets


def _success(node: dict, assets: dict, placements: dict) -> SuccessSpec:
    (key, val), = node.items()
    if key == "all_of":
        return AllOf(terms=tuple(_success(t, assets, placements) for t in val))
    if key == "any_of":
        return AnyOf(terms=tuple(_success(t, assets, placements) for t in val))
    if key == "not":
        return Not(term=_success(val, assets, placements))
    if key in ("inside", "on"):
        obj = val["object"]
        parent, target = relations.split_ref(val["container" if key == "inside" else "surface"])
        if target is None:
            raise relations.SolveError(f"{key}: '{parent}' must name a place target as <id>.<target>")
        spec = relations.place_target(assets[parent], target)
        lo, hi = derive.visual_bounds(assets[obj].urdf_path)
        params = {
            "object": obj,
            "parent": parent,
            "box": spec["box"],
            "yaw_deg": placements[parent].yaw_deg,
            "object_bottom": float(-lo[2]),
            "object_height": float(hi[2] - lo[2]),
            "mode": key,
            "tol": CONTACT_TOL_M,
        }
        if "joint" in spec:
            params["joint"] = spec["joint"]
            params["axis"] = list(assets[parent].get_joint(spec["joint"]).axis)
        return TaskSuccessCfg(predicate="object_in_box", params=params, refs=(obj, parent))
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
