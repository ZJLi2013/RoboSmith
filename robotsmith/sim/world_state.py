"""What success predicates judge, read from the simulator.

``collect_world_state`` only reads; ``settle_window`` is the one place that advances the simulator for
judging, and only when the caller asks for it.
"""

from __future__ import annotations

import numpy as np


def _arr(x) -> np.ndarray:
    return np.asarray(x.cpu().numpy() if hasattr(x, "cpu") else x, dtype=np.float64)


def _touches(ent, other) -> bool:
    return _arr(ent.get_contacts(with_entity=other)["link_a"]).size > 0


def _pose(ent) -> list[float]:
    return np.r_[_arr(ent.get_pos()).reshape(-1)[:3], _arr(ent.get_quat()).reshape(-1)[:4]].tolist()


def collect_world_state(env) -> dict:
    """One simulator step's view of every entity: position, pose, world collision AABB, what it touches
    (``"robot"`` for the arm), and the articulated joints. Reads only."""
    names = list(env.entity_map)
    ents = env.entity_map
    return {
        "step": getattr(env.scene, "t", None),
        "object_positions": {n: _arr(ents[n].get_pos()).reshape(-1)[:3].tolist() for n in names},
        "poses": {n: _pose(ents[n]) for n in names},
        "aabbs": {n: _arr(ents[n].get_AABB()).reshape(2, 3).tolist() for n in names},
        "contacts": {
            n: [m for m in names if m != n and _touches(ents[n], ents[m])]
            + (["robot"] if _touches(ents[n], env.franka) else [])
            for n in names
        },
        "joint_positions": env.get_joint_positions() if hasattr(env, "get_joint_positions") else {},
    }


def settle_window(env, hold, *, settle_s: float = 0.5, samples: int = 10) -> dict:
    """Hold the arm at the ``hold`` command for ``settle_s`` (not recorded) and sample every entity's pose
    at ``samples`` evenly spaced steps. ``settle_s == 0`` steps nothing and returns an empty window."""
    total = round(settle_s / float(env.scene.dt))
    marks = sorted({round((k + 1) * total / samples) for k in range(samples)}) if total > 0 else []
    window = {"steps": [], "poses": {n: [] for n in env.entity_map}}
    for i in range(1, total + 1):
        env.franka.control_dofs_position(hold, env.motors_dof)
        env.scene.step()
        if i in marks:
            window["steps"].append(getattr(env.scene, "t", i))
            for n, ent in env.entity_map.items():
                window["poses"][n].append(_pose(ent))
    return window
