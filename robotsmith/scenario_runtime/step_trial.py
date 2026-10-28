"""s40 experiment: replay or replan a failed place step from the snapshot taken at its start.

Replay groups reuse the original arm commands frame for frame; the replan group plans the step
again from the same snapshot with the target's moving link kept as an obstacle during transport.
"""

from __future__ import annotations

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
