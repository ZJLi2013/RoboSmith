"""Grasps written back to an asset after a repair lifted it in simulation.

``asset.json`` stores them as the ``grasp_candidates`` annotation, in the object frame the simulator reports
(the base link's inertial origin). They are used only while a passing ``trial_grasp`` check is current for
the asset's revision, build files and embodiment; a rebuild or another robot drops them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robotsmith.assets import store
from robotsmith.grasp.transforms import quat_wxyz_to_matrix

ANNOTATION = "grasp_candidates"
CHECK = "trial_grasp"
EMBODIMENT = "panda"


def _rot(object_quat) -> np.ndarray:
    return np.eye(3) if object_quat is None else quat_wxyz_to_matrix(np.asarray(object_quat, dtype=np.float64))


def current_grasps(asset_dir: Path, embodiment: str = EMBODIMENT) -> list[dict]:
    if not (Path(asset_dir) / store.ASSET_FILE).exists():
        return []
    data = store.read_asset(asset_dir)
    value = data.get("annotations", {}).get(ANNOTATION, {}).get("value")
    if not value:
        return []
    check = store.current_checks(asset_dir, data).get(CHECK)
    if not check or check["result"] != "pass" or check["embodiment"] != embodiment:
        return []
    if (value.get("embodiment"), value.get("asset_revision"), value.get("build_hash")) != (
            embodiment, data["revision"], data["build"]["hash"]):
        return []
    return value["candidates"]


def to_world(candidate: dict, object_pos, object_quat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = _rot(object_quat)
    return (np.asarray(object_pos, dtype=np.float64) + R @ np.asarray(candidate["tip_local"]),
            R @ np.asarray(candidate["approach_local"]), R @ np.asarray(candidate["finger_axis_local"]))


def to_local(tip, approach, finger_axis, object_pos, object_quat) -> dict:
    R = _rot(object_quat)
    return {"tip_local": (R.T @ (np.asarray(tip, dtype=np.float64) - np.asarray(object_pos))).round(5).tolist(),
            "approach_local": (R.T @ np.asarray(approach, dtype=np.float64)).round(5).tolist(),
            "finger_axis_local": (R.T @ np.asarray(finger_axis, dtype=np.float64)).round(5).tolist()}


def write_grasp(asset_dir: Path, candidate: dict, object_pos, object_quat, *, lift_m: float, by: str,
                run: dict, embodiment: str = EMBODIMENT) -> None:
    """Record ``candidate`` (world frame, as passed to the trial) and the trial that lifted it."""
    data = store.read_asset(asset_dir)
    local = to_local(candidate["tip_xyz"], candidate["approach"], candidate["finger_axis"], object_pos, object_quat)
    value = {"frame": "object pose reported by the simulator (base link inertial origin), metres",
             "embodiment": embodiment, "asset_revision": data["revision"], "build_hash": data["build"]["hash"],
             "candidates": [{**local, "ik_seed": candidate.get("ik_seed")}]}
    store.set_annotation(asset_dir, ANNOTATION, value, by=by, note=f"repair verified by trial: {run}")
    store.append_check(asset_dir, CHECK, "pass", {"lift_m": lift_m, **run}, embodiment=embodiment)
