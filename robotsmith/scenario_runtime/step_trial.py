"""s40 experiment: replay or replan a failed place step from the snapshot taken at its start.

Replay groups reuse the original arm commands frame for frame; the replan group plans the step
again from the same snapshot with the target's moving link kept as an obstacle during transport.
"""

from __future__ import annotations

import hashlib

import numpy as np

from robotsmith.scenario_runtime import handoff
from robotsmith.sim.franka import to_numpy
from robotsmith.sim.recorder import _probe_contacts
from robotsmith.skills import place as place_skill

FORCE_N = 5.0
VARIANTS = (
    ("baseline", "replay"),
    ("finger_force_5n", "replay"),
    ("keep_target_link_in_transport", "replan"),
)


def _replay(env, seg, ent, force: float) -> list[dict]:
    samples = []
    for q in seg:
        env.franka.control_dofs_position(q, env.motors_dof)
        if force and float(q[7]) < 0.02:
            env.franka.control_dofs_force(np.array([-force, -force]), env.finger_dof)
        env.scene.step()
        samples.append({
            "hand": to_numpy(env.end_effector.get_pos()).reshape(-1)[:3],
            "q": to_numpy(env.franka.get_dofs_position(env.motors_dof)).reshape(-1)[:7],
            "obj": to_numpy(ent.get_pos()).reshape(-1)[:3],
            "contacts": _probe_contacts(env, ent),
        })
    return samples


def _brief(h: dict) -> dict:
    out = {k: h.get(k) for k in ("done", "first_loss_frame", "release_frame", "to_target_xy_m", "seg_hash")}
    around = (h.get("probe") or {}).get("around_loss") or []
    out["contacts_before_loss"] = sorted({n for c in around[:6] for n in c["env"]})
    out["carry_env"] = (h.get("probe") or {}).get("carry_env")
    return out


SETTLE_STEPS = 30
S41_VARIANTS = (
    ("baseline", "replay"),
    ("replan_zero", "replan"),
    ("roll_90", "replan"),
    ("place_z_minus_1p5cm", "replan"),
)


def _n_contacts(c) -> int:
    from robotsmith.sim.recorder import _raw

    return int(_raw(c["link_a"]).size)


def fixed_state(env, obj: str, parent: str | None, on_predicate) -> dict:
    """The s41 definition of a fixed place: the task predicate holds, the object no longer touches
    the robot and rests on the parent."""
    ent = env.entity_map[obj]
    touch_robot = _n_contacts(ent.get_contacts(with_entity=env.franka)) > 0
    par = env.entity_map.get(parent) if parent else None
    touch_parent = _n_contacts(ent.get_contacts(with_entity=par)) > 0 if par is not None else None
    on = bool(on_predicate())
    return {"on": on, "touch_robot": touch_robot, "touch_parent": touch_parent,
            "obj_z": round(float(to_numpy(ent.get_pos()).reshape(-1)[2]), 4),
            "fixed": on and not touch_robot and touch_parent is not False}


def _settle(env, q_last, steps: int = SETTLE_STEPS) -> None:
    for _ in range(steps):
        env.franka.control_dofs_position(q_last, env.motors_dof)
        env.scene.step()


def run_place_variants(env, snapshot, *, seg, obj, parent, on_predicate, replan) -> list[dict]:
    """From the place snapshot: replay the original commands and replan three variants; judge each
    right at the end of the step and again after settling."""
    from robotsmith.grasp import planner as grasp_planner
    from robotsmith.motion import chain

    post = env.scene.get_state()
    ent = env.entity_map[obj]
    results = []
    try:
        for variable, mode in S41_VARIANTS:
            env.scene.reset(snapshot)
            chain.PLAN_SOURCES.clear()
            seg_v = seg
            if mode == "replan":
                grasp_planner.S41_ROLL_90 = variable == "roll_90"
                try:
                    seg_v = replan(place_z_delta=-0.015 if variable == "place_z_minus_1p5cm" else 0.0)
                finally:
                    grasp_planner.S41_ROLL_90 = False
            samples = _replay(env, seg_v, ent, 0.0)
            at_end = fixed_state(env, obj, parent, on_predicate)
            _settle(env, seg_v[-1])
            settled = fixed_state(env, obj, parent, on_predicate)
            cmd_hash = hashlib.md5(np.round(np.asarray(seg_v, dtype=float)[:, :7], 3).tobytes()).hexdigest()[:10]
            results.append({"variable": variable, "mode": mode, "cmd_hash": cmd_hash,
                            "plan_sources": list(chain.PLAN_SOURCES) if mode == "replan" else None,
                            "at_end": at_end, "settled": settled})
    finally:
        env.scene.reset(post)
        env.franka.control_dofs_position(seg[-1], env.motors_dof)
    return results


def run_trials(env, snapshot, *, skill, phase, seg, obj, held_start, target, replan) -> list[dict]:
    post = env.scene.get_state()
    ent = env.entity_map[obj]
    results = []
    try:
        for variable, mode in VARIANTS:
            env.scene.reset(snapshot)
            seg_v, world = seg, None
            if mode == "replan":
                place_skill.S40_KEEP_TARGET_LINK_IN_TRANSPORT = True
                try:
                    seg_v, world = replan()
                finally:
                    place_skill.S40_KEEP_TARGET_LINK_IN_TRANSPORT = False
                if not world or not world.get("collision_aware") or world.get("exempt_links"):
                    results.append({"variable": variable, "mode": mode, "world": world,
                                    "skipped": "replan did not keep the target link in a collision-aware world"})
                    continue
            samples = _replay(env, seg_v, ent, FORCE_N if variable == "finger_force_5n" else 0.0)
            h = handoff.check(skill, phase, seg_v, samples, held=obj, held_start=held_start, target=target)
            results.append({"variable": variable, "mode": mode, "world": world, "handoff": _brief(h)})
    finally:
        env.scene.reset(post)
        env.franka.control_dofs_position(seg[-1], env.motors_dof)
    return results
