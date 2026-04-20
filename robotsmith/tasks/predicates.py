"""Composable predicates for success/failure evaluation.

Each predicate is a pure function: (env_state, **params) -> bool.
Registered in PREDICATE_REGISTRY by name (str key) for serializable TaskSpec.
"""

from __future__ import annotations

from typing import Callable, Any

import numpy as np

PredicateFn = Callable[..., bool]

PREDICATE_REGISTRY: dict[str, PredicateFn] = {}


def register_predicate(name: str):
    """Decorator to register a predicate function."""
    def decorator(fn: PredicateFn) -> PredicateFn:
        PREDICATE_REGISTRY[name] = fn
        return fn
    return decorator


def evaluate_predicate(name: str, env_state: dict, params: dict) -> bool:
    """Look up and evaluate a named predicate."""
    if name not in PREDICATE_REGISTRY:
        raise KeyError(
            f"Unknown predicate '{name}'. "
            f"Available: {list(PREDICATE_REGISTRY.keys())}"
        )
    return PREDICATE_REGISTRY[name](env_state, **params)


# ---------- Built-in predicates ----------


@register_predicate("object_above")
def object_above(env_state: dict, *, object: str, reference: str, z_margin: float = 0.05) -> bool:
    """True if object's Z is at least z_margin above its initial Z.

    env_state expected keys:
      - "object_positions": dict[str, np.ndarray]  (name -> [x, y, z])
      - "initial_positions": dict[str, np.ndarray]  (name -> [x, y, z])
    """
    pos = env_state["object_positions"][object]
    initial_z = env_state["initial_positions"][object][2]
    return float(pos[2] - initial_z) >= z_margin


@register_predicate("object_in_container")
def object_in_container(
    env_state: dict,
    *,
    object: str,
    container: str,
    xy_threshold: float = 0.05,
    z_margin: float = 0.0,
) -> bool:
    """True if object is inside container (XY within threshold, Z above container bottom)."""
    obj_pos = env_state["object_positions"][object]
    cont_pos = env_state["object_positions"][container]
    xy_dist = np.linalg.norm(obj_pos[:2] - cont_pos[:2])
    above = obj_pos[2] >= cont_pos[2] + z_margin
    return float(xy_dist) < xy_threshold and above


@register_predicate("stacked")
def stacked(env_state: dict, *, objects: list[str], z_tolerance: float = 0.02) -> bool:
    """True if objects are stacked in order (each above the previous)."""
    positions = env_state["object_positions"]
    for i in range(1, len(objects)):
        lower = positions[objects[i - 1]]
        upper = positions[objects[i]]
        if upper[2] <= lower[2] + z_tolerance:
            return False
    return True
