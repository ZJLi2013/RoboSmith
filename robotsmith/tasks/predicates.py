"""Composable predicates for success/failure evaluation.

Each predicate is a pure function: (env_state, **params) -> bool.
Registered in PREDICATE_REGISTRY by name (str key) for serializable TaskSpec.
"""

from __future__ import annotations

from typing import Callable, Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from robotsmith.tasks.task_spec import SuccessNode

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


def evaluate_success(node: "SuccessNode", env_state: dict) -> bool:
    """Evaluate a success-condition tree.

    Leaves look up PREDICATE_REGISTRY; composite ops (all/any/not) recurse.
    """
    if node.op == "leaf":
        return evaluate_predicate(node.predicate, env_state, node.params)
    if node.op == "all":
        return all(evaluate_success(t, env_state) for t in node.terms)
    if node.op == "any":
        return any(evaluate_success(t, env_state) for t in node.terms)
    if node.op == "not":
        return not evaluate_success(node.terms[0], env_state)
    raise ValueError(f"unknown success op '{node.op}'")


def find_leaf(node: "SuccessNode", predicate: str) -> "SuccessNode | None":
    """Return the first leaf with the given predicate name, or None.

    Used by callers that need a specific leaf's params (e.g. layout alignment
    axis, diagnostics) without assuming a single flat predicate.
    """
    if node.op == "leaf":
        return node if node.predicate == predicate else None
    for term in node.terms:
        found = find_leaf(term, predicate)
        if found is not None:
            return found
    return None


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
    z_tol: float | None = None,
) -> bool:
    """True if object is inside a physical container or at a target marker.

    Physical containers live in ``object_positions`` and keep the single-sided
    Z check (``z >= bottom + z_margin``: getting in is enough).

    Non-physical target markers live in ``target_positions``. Their Z is the
    *resolved world height* the object should rest at (e.g. a specific shelf),
    not a container bottom. When ``z_tol`` is given, the marker Z is enforced as
    a two-sided band ``abs(obj_z - target_z) <= z_tol`` so XY-coincident but
    Z-distinct targets (stacked shelves) are distinguishable. ``z_tol=None``
    keeps the legacy XY-only behaviour for callers that have not opted in.
    """
    obj_pos = env_state["object_positions"][object]
    target_positions = env_state.get("target_positions", {})
    is_target_marker = container in target_positions
    cont_pos = (
        target_positions[container]
        if is_target_marker
        else env_state["object_positions"][container]
    )
    xy_dist = np.linalg.norm(obj_pos[:2] - cont_pos[:2])
    if is_target_marker:
        at_z = True if z_tol is None else abs(float(obj_pos[2] - cont_pos[2])) <= z_tol
    else:
        at_z = obj_pos[2] >= cont_pos[2] + z_margin
    return float(xy_dist) < xy_threshold and at_z


@register_predicate("joint_opened")
def joint_opened(
    env_state: dict, *, object: str, joint: str, threshold: float
) -> bool:
    """True if an articulated object's joint is open past *threshold*.

    "Open" means the joint position has moved toward its open limit, by
    convention the higher position value (e.g. a drawer pulled out, a lid lifted).

    env_state expected keys:
      - "joint_positions": dict[str, dict[str, float]]  (object -> joint -> qpos)
    """
    qpos = env_state["joint_positions"][object][joint]
    return float(qpos) >= threshold


@register_predicate("joint_closed")
def joint_closed(
    env_state: dict, *, object: str, joint: str, threshold: float = 0.02
) -> bool:
    """True if an articulated object's joint is at/below the closed *threshold*."""
    qpos = env_state["joint_positions"][object][joint]
    return float(qpos) <= threshold


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


@register_predicate("objects_aligned")
def objects_aligned(
    env_state: dict,
    *,
    objects: list[str],
    axis: str = "y",
    xy_threshold: float = 0.06,
) -> bool:
    """True if all objects are aligned along *axis*.

    "Aligned along Y" means they form a line parallel to Y, so their
    X coordinates (the cross-axis) must be close to each other.  Additionally,
    they must be sorted along *axis* (i.e. actually spread out, not piled up).
    """
    cross = 0 if axis == "y" else 1  # cross-axis index
    along = 1 if axis == "y" else 0

    positions = env_state["object_positions"]
    coords = [positions[name] for name in objects]

    cross_vals = [float(c[cross]) for c in coords]
    cross_spread = max(cross_vals) - min(cross_vals)
    if cross_spread > xy_threshold:
        return False

    along_vals = [float(c[along]) for c in coords]
    min_spacing = xy_threshold * 0.5
    sorted_vals = sorted(along_vals)
    for i in range(1, len(sorted_vals)):
        if sorted_vals[i] - sorted_vals[i - 1] < min_spacing:
            return False
    return True
