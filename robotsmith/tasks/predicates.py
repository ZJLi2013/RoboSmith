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


def failed_leaves(node: "SuccessNode", env_state: dict) -> list[dict]:
    """Leaves whose value works against the tree's verdict; a leaf under ``not`` fails by being true."""

    def walk(n: "SuccessNode", want: bool) -> list[dict]:
        if n.op == "leaf":
            return [] if evaluate_predicate(n.predicate, env_state, n.params) == want else [n.to_dict()]
        if n.op == "not":
            return walk(n.terms[0], not want)
        return [leaf for t in n.terms for leaf in walk(t, want)]

    return walk(node, True)


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


class JudgeInputError(ValueError):
    """The state a predicate needs is missing or not finite; a verdict would be a guess."""


def _observed(env_state: dict, key: str, name: str):
    value = env_state.get(key, {}).get(name)
    if value is None or not np.all(np.isfinite(np.asarray(value, dtype=np.float64))):
        raise JudgeInputError(f"{key}[{name!r}] is missing or not finite")
    return value


def _settled(frames: list, max_move: float, max_turn_deg: float) -> bool:
    """Every sampled frame (xyz + wxyz quat) stays within the limits of the last one."""
    frames = np.asarray(frames, dtype=np.float64)
    last = frames[-1]
    moved = np.linalg.norm(frames[:, :3] - last[:3], axis=1).max()
    dots = np.clip(np.abs(frames[:, 3:] @ last[3:]), 0.0, 1.0)
    turned = np.degrees(2.0 * np.arccos(dots)).max()
    return bool(moved <= max_move and turned <= max_turn_deg)


@register_predicate("object_in_box")
def object_in_box(
    env_state: dict,
    *,
    object: str,
    parent: str,
    box: list[list[float]],
    yaw_deg: float,
    mode: str,
    target: str | None = None,
    tol: float = 0.02,
    joint: str | None = None,
    axis: list[float] | None = None,
    max_move: float = 0.005,
    max_turn_deg: float = 5.0,
) -> bool:
    """True if the object, as the simulator last saw it, rests on (``on``) or in (``inside``) a box of ``parent``.

    ``target`` names the place target the box came from; judging ignores it, step bindings match on it.
    ``box`` is ``[lo, hi]`` in the parent frame at zero joint configuration; when ``joint`` is set the box
    rides that prismatic joint by its live value. The object's world collision AABB decides: ``on`` needs
    the AABB centre over the box and its bottom within ``tol`` of the box top; ``inside`` (the box is the
    container floor) needs the whole AABB footprint over the box and its bottom within ``tol`` of the top.
    Both need the object touching ``parent``, not touching the robot, and still through the settle window.
    """
    lo, hi = np.asarray(box[0], dtype=np.float64), np.asarray(box[1], dtype=np.float64)
    if joint is not None:
        shift = np.asarray(axis, dtype=np.float64) * float(env_state["joint_positions"][parent][joint])
        lo, hi = lo + shift, hi + shift
    a_lo, a_hi = (np.asarray(v, dtype=np.float64) for v in _observed(env_state, "aabbs", object))
    base = np.asarray(_observed(env_state, "object_positions", parent), dtype=np.float64)
    frames = env_state.get("window", {}).get("poses", {}).get(object)
    if not frames:
        raise JudgeInputError(f"no settle window for {object!r}")
    if object not in env_state.get("contacts", {}):
        raise JudgeInputError(f"contacts[{object!r}] is missing")
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))

    def local(x: float, y: float) -> tuple[float, float]:
        dx, dy = x - base[0], y - base[1]
        return c * dx + s * dy, -s * dx + c * dy

    def over(x: float, y: float) -> bool:
        lx, ly = local(x, y)
        return bool(lo[0] <= lx <= hi[0] and lo[1] <= ly <= hi[1])

    if mode == "on":
        placed = over((a_lo[0] + a_hi[0]) / 2.0, (a_lo[1] + a_hi[1]) / 2.0)
    else:
        placed = all(over(x, y) for x in (a_lo[0], a_hi[0]) for y in (a_lo[1], a_hi[1]))
    rests = abs(a_lo[2] - (base[2] + hi[2])) <= tol
    touching = env_state["contacts"][object]
    return bool(placed and rests and parent in touching and "robot" not in touching
                and _settled(frames, max_move, max_turn_deg))


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
