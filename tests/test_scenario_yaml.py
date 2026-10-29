"""YAML scenarios load, and their success trees give the approved truth values
(the task-semantics tables of the scenario contract)."""

import shutil
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from robotsmith.assets import derive, store
from robotsmith.assets.library import AssetLibrary
from robotsmith.scenario.load import SCHEMA, _read, load_scenario
from robotsmith.scenario_runtime import materialize_candidate
from robotsmith.tasks import evaluate_success

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"
ASSETS = SCENARIOS.parent / "assets"
TABLE_Z = 0.775
DIE_HALF = 0.025


def _package(name: str):
    return materialize_candidate(SCENARIOS / f"{name}.yaml", seed=0)


def _placed(pkg) -> dict:
    return {o.name_override: np.asarray(o.fixed_position) for o in pkg.legacy_scene.objects}


def _bounds(pkg) -> dict:
    library = AssetLibrary(ASSETS)
    return {o.name_override: derive.visual_bounds(library.get(o.asset_query).urdf_path)
            for o in pkg.legacy_scene.objects}


def _box_leaf(pkg) -> dict:
    node = pkg.legacy_task.success
    while node.op != "leaf":
        node = node.terms[0]
    return node.params


def _on_box(pkg, slide: float = 0.0, dy: float = 0.0) -> list[float]:
    """Object resting on the success box, its centre offset ``dy`` along local y."""
    leaf = _box_leaf(pkg)
    base = _placed(pkg)[leaf["parent"]]
    lo, hi = np.asarray(leaf["box"][0]), np.asarray(leaf["box"][1])
    local = (lo + hi) / 2.0 + np.asarray(leaf.get("axis") or [0, 0, 0]) * slide + [0.0, dy, 0.0]
    c, s = np.cos(np.radians(leaf["yaw_deg"])), np.sin(np.radians(leaf["yaw_deg"]))
    xy = base[:2] + [c * local[0] - s * local[1], s * local[0] + c * local[1]]
    return [xy[0], xy[1], base[2] + hi[2] - _bounds(pkg)[leaf["object"]][0][2]]


def _success(pkg, moved: dict, joints: dict | None = None) -> bool:
    """Judge an upright, settled layout where every object touches every other one but not the robot."""
    placed = _placed(pkg)
    positions = {**placed, **{k: np.asarray(v) for k, v in moved.items()}}
    bounds = _bounds(pkg)
    state = {
        "object_positions": positions,
        "initial_positions": placed,
        "joint_positions": joints or {},
        "target_positions": {},
        "aabbs": {n: [p + bounds[n][0], p + bounds[n][1]] for n, p in positions.items()},
        "contacts": {n: [m for m in positions if m != n] for n in positions},
        "window": {n: [[*p, 1.0, 0.0, 0.0, 0.0]] * 2 for n, p in positions.items()},
    }
    return evaluate_success(pkg.legacy_task.success, state)


@pytest.mark.parametrize("name", sorted(p.stem for p in SCENARIOS.glob("*.yaml")))
def test_scenario_loads_with_teacher(name):
    pkg = _package(name)
    assert pkg.legacy_task.skills
    assert all(o.fixed_position is not None for o in pkg.legacy_scene.objects)


CLOSED, OPEN = {"drawer": {"drawer_slide": 0.0}}, {"drawer": {"drawer_slide": 0.2}}

CASES = [
    ("pick_place_into_drawer", "in tray, closed", lambda p, o: {"die": _on_box(p)}, CLOSED, True),
    ("pick_place_into_drawer", "in tray, open", lambda p, o: {"die": _on_box(p, slide=0.2)}, OPEN, False),
    ("pick_place_into_drawer", "on table in front",
     lambda p, o: {"die": [o["drawer"][0] - 0.23, 0.0, TABLE_Z + DIE_HALF]}, CLOSED, False),
    ("pick_place_into_drawer", "on cabinet top",
     lambda p, o: {"die": [o["drawer"][0], 0.0, TABLE_Z + 0.256 + DIE_HALF]}, CLOSED, False),
    ("pick_place_into_drawer", "never moved", lambda p, o: {}, CLOSED, False),
    ("pick_one_of_three", "apple lifted 0.15", lambda p, o: {"apple": o["apple"] + [0, 0, 0.15]}, None, True),
    ("pick_one_of_three", "apple pushed", lambda p, o: {"apple": o["apple"] + [0.05, 0, 0]}, None, False),
    ("pick_one_of_three", "mug lifted", lambda p, o: {"mug": o["mug"] + [0, 0, 0.15]}, None, False),
    ("pick_one_of_three", "apple on mug", lambda p, o: {"apple": o["mug"] + [0, 0, 0.064]}, None, False),
    ("pick_place_onto_supporter", "on lower shelf", lambda p, o: {"die": _on_box(p)}, None, True),
    ("pick_place_onto_supporter", "still on upper shelf", lambda p, o: {}, None, False),
    ("pick_place_onto_supporter", "fell to table",
     lambda p, o: {"die": [o["supporter"][0] - 0.12, 0.0, TABLE_Z + DIE_HALF]}, None, False),
    ("pick_place_onto_supporter", "lower shelf, 0.06 m off centre",
     lambda p, o: {"die": _on_box(p, dy=0.06)}, None, True),
    ("pick_place_table_to_lower_shelf", "on lower shelf", lambda p, o: {"die": _on_box(p)}, None, True),
    ("pick_place_table_to_lower_shelf", "never moved", lambda p, o: {}, None, False),
    ("pick_place_table_to_lower_shelf", "pushed next to the base",
     lambda p, o: {"die": [o["supporter"][0] - 0.12, 0.0, TABLE_Z + DIE_HALF]}, None, False),
    ("pick_place_table_to_lower_shelf", "on upper shelf",
     lambda p, o: {"die": [o["supporter"][0], 0.0, TABLE_Z + 0.304 + DIE_HALF]}, None, False),
]


@pytest.mark.parametrize("name, case, moved, joints, expected", CASES, ids=[f"{c[0]}: {c[1]}" for c in CASES])
def test_task_semantics(name, case, moved, joints, expected):
    pkg = _package(name)
    assert _success(pkg, moved(pkg, _placed(pkg)), joints) is expected


def test_on_key_is_not_a_boolean_and_schema_rejects_unknown_relations(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("scenario: s\nrobot: franka\nscene:\n  support: table_simple\n  objects:\n"
                    "    - {id: die, asset: die_01, on: support, beside: robot}\n"
                    "task: {instruction: x, success: {lifted: {object: die, height: 0.1}}}\n")
    data = _read(path)
    assert data["scene"]["objects"][0]["on"] == "support"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, SCHEMA)


def test_scenario_refuses_a_draft_asset(tmp_path):
    shutil.copytree(ASSETS / "die_01", tmp_path / "assets" / "die_01")
    (tmp_path / "assets" / "die_01" / "checks.jsonl").unlink()
    path = tmp_path / "s.yaml"
    path.write_text("scenario: s\nrobot: franka\nscene:\n  support: table_simple\n  objects:\n"
                    "    - {id: die, asset: die_01, on: support, near: robot}\n"
                    "task: {instruction: x, success: {lifted: {object: die, height: 0.1}}}\n")
    with pytest.raises(store.AssetNotAdmitted, match="die_01: no llm_review on the current build"):
        load_scenario(path, seed=0, library=AssetLibrary(tmp_path / "assets"))
