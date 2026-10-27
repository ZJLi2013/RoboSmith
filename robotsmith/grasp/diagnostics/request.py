"""Evidence package and prompt for a diagnosis agent, built from the failed run itself."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import numpy as np

from robotsmith.assets import derive

PLAYBOOK = Path(__file__).with_name("playbook.md")
DIRECTIONS = ("coordinate_frame, closing_axis, support_clearance, grip_control, scene_placement, add_candidate, "
              "ik_reachability, asset_mass_or_inertia, asset_friction, asset_geometry_or_collision, solver_settings, "
              "unknown")
NEXT_STEPS = ("propose_candidate_pose, propose_ik_seed, request_asset_fix, flag_scene_placement, "
              "report_primitive_defect, report_unknown, change_control_parameters_for_this_asset")

ROLE = """You are the grasp-diagnosis agent of a simulation benchmark. A geometric grasp primitive plans candidates from
the object's geometry and the gripper spec; each candidate is tried in simulation and judged by lift > 0.02 m.
Read the evidence package and decide the diagnosis direction and who should fix it.
Permissions: you may propose candidate poses or IK seeds, request an asset fix, or flag scene placement. You may not
change the gripper spec, controller parameters or candidate rules, and never set controller parameters per asset.
A problem shared by all assets is reported as a primitive defect, not patched. If the evidence does not support a
direction, report unknown. Fields marked "unknown" were not observed."""

SCHEMA = f"""Answer with exactly one JSON object and nothing else:
{{"evidence": ["<field path>: <reading>", ...], "direction": "<one of DIRECTIONS>", "owner": "<one of OWNERS>",
 "next_step": "<one of NEXT_STEPS>", "rationale": "<at most three sentences>"}}
DIRECTIONS: {DIRECTIONS}
OWNERS: asset (this asset's own data or grasp annotation), scene (placement / layout), primitive (shared grasp
primitive or controller, affects every asset), unknown
NEXT_STEPS: {NEXT_STEPS}
If next_step is propose_candidate_pose or propose_ik_seed, add "repair": the one candidate (same fields as
candidates.json) that lifted in your tool calls; otherwise omit it."""


def _tool(tool_cmd: str, table_z: float, gripper) -> str:
    return f"""Tool: {tool_cmd} <candidates.json> tries up to 8 grasps in simulation from the same start state and prints,
per candidate, lifted, lift_m, motion_error (IK / planning rejection) and the hand position used. You have 4 calls.
candidates.json is a JSON list of {{"tip_xyz": [x, y, z], "approach": [x, y, z], "finger_axis": [x, y, z],
"ik_seed": [7 joint angles in rad] or null}}. World frame, metres, z up, table surface at z = {table_z:.3f}.
tip_xyz is the tool centre point midway between the finger pads; the pads extend {gripper.pad_below_tcp} m beyond it
along approach. approach is the unit direction from the hand toward the object ([0, 0, -1] is top-down).
finger_axis is the closing direction; it is made orthogonal to approach.
The object's opening across finger_axis must be below {gripper.max_width:.3f} m."""


def _r(v) -> list[float]:
    return [round(float(x), 4) for x in v]


def _sections(mesh) -> list[dict]:
    """Horizontal cuts of the world-frame mesh at 10–90 % of its height."""
    z_lo, z_hi = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
    out = []
    for f in (0.1, 0.3, 0.5, 0.7, 0.9):
        z = z_lo + f * (z_hi - z_lo)
        s = mesh.section(plane_origin=[0.0, 0.0, z], plane_normal=[0.0, 0.0, 1.0])
        if s is not None:
            lo, hi = s.vertices[:, :2].min(0), s.vertices[:, :2].max(0)
            out.append({"z_m": round(z, 4), "x_min_max_m": _r([lo[0], hi[0]]), "y_min_max_m": _r([lo[1], hi[1]]),
                        "center_xy_m": _r((lo + hi) / 2)})
    return out


def evidence(*, urdf_path: Path, category: str, pos, quat, aabb, table_z: float, gripper, plans, label: dict,
             object_xy) -> dict:
    from robotsmith.grasp.transforms import quat_wxyz_to_matrix

    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_matrix(np.asarray(quat, dtype=np.float64))
    T[:3, 3] = np.asarray(pos, dtype=np.float64) - T[:3, :3] @ derive.inertial_origin(urdf_path)
    world = derive.visual_mesh(urdf_path).apply_transform(T)
    mass = float(ET.parse(urdf_path).getroot().find(".//inertial/mass").get("value"))
    by_idx = {p.metadata.get("candidate_index"): p for p in plans}
    trials = []
    for t in label.get("trials") or []:
        row = {"candidate": t["candidate_index"], "kind": t.get("kind"), "ik":
               "rejected: " + t["motion_error"] if t.get("motion_error") else "ok", "lift_m": t.get("lift_m")}
        p = by_idx.get(t["candidate_index"])
        if p is not None:
            md = p.metadata
            row |= {"closing_width_m": round(md.get("closing_width", 0.0), 4), "tip_xyz": _r(md["tip_pos"]),
                    "approach": _r(md["approach_dir"]), "finger_axis": _r(md["finger_dir"])}
        trials.append(row)
    lifts = [t["lift_m"] or 0.0 for t in trials]
    question = ("The geometric planner produced 0 candidates for this object, so no grasp was tried; diagnose."
                if not plans else
                f"The planner produced {len(plans)} candidates; {len(trials)} were tried and none lifted the object "
                f"(max lift {max(lifts, default=0.0):.3f} m); diagnose.")
    return {
        "question": question,
        "gripper": {"type": "Franka Panda parallel gripper", "max_opening_m": round(gripper.max_width, 4),
                    "finger_pad_beyond_tcp_m": gripper.pad_below_tcp},
        "object": {"category": category, "mass_kg": mass, "aabb_min_m": _r(aabb[0]), "aabb_max_m": _r(aabb[1]),
                   "horizontal_sections": _sections(world)},
        "placement": {"requested_xy": _r(object_xy), "settled_pos": _r(pos), "settled_quat_wxyz": _r(quat),
                      "settle_xy_displacement_m": round(float(np.hypot(pos[0] - object_xy[0], pos[1] - object_xy[1])), 4),
                      "table_surface_z": round(table_z, 4)},
        "planner": {"n_candidates": len(plans)},
        "trials": trials,
        "recorded_finger_final_qpos_m": label.get("finger_final_qpos") or "unknown",
        "diagnostic_tool_results": [],
    }


def write_request(out_dir: Path, *, packet: dict, context: dict, table_z: float, gripper,
                  tool_cmd: str = "try_grasp") -> Path:
    """``request.md`` for the agent and ``context.json`` for ``tool`` / ``writeback``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = [ROLE, _tool(tool_cmd, table_z, gripper), SCHEMA, "Diagnosis playbook:\n" + PLAYBOOK.read_text(encoding="utf-8"),
             "Evidence package:\n" + json.dumps(packet, indent=1)]
    (out_dir / "request.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "context.json").write_text(json.dumps({**context, "gripper": asdict(gripper)}, indent=1), encoding="utf-8")
    return out_dir / "request.md"
