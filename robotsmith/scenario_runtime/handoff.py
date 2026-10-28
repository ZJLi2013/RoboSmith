"""Per-skill handoff checks: did each skill leave the world the way the next one needs it?

Judged from per-frame sim samples (hand link, held object, actual arm joints) taken on the
recording timeline, so a failed episode can be pinned to the first skill that did not finish.
Place causes are candidates (``motion?`` / ``grip?`` / ``release?``), not conclusions.
"""

from __future__ import annotations

import numpy as np

# Hand link origin sits ~0.10 m above the fingertips.
HELD_M = 0.15
LIFT_M = 0.02
TARGET_XY_M = 0.08
OPEN_FRACTION = 0.5
CLOSED_M = 0.02
LOSS_WINDOW = 10
JUMP_RAD = 0.03


def _joint_values(joints: dict, asset: str) -> np.ndarray:
    return np.asarray(list((joints or {}).get(asset, {}).values()), dtype=float)


def _release_frame(seg) -> int:
    """First frame whose finger command opens past the carry width (len(seg) if none)."""
    if not len(seg):
        return 0
    fingers = np.asarray(seg, dtype=float)[:, 7]
    opened = np.nonzero(fingers > fingers[0] + 1e-3)[0]
    return int(opened[0]) if opened.size else len(seg)


def check(skill, phase: str, seg, samples: list[dict], *, held: str | None,
          held_start=None, target=None, joints_before=None, joints_after=None,
          trace: dict | None = None, rocrobo: bool = False) -> dict:
    trace = trace or {}
    entry: dict = {"skill": skill.name, "phase": phase, "target": skill.target, "object": held}
    q = np.asarray([s["q"] for s in samples], dtype=float) if samples else np.zeros((0, 7))
    steps = np.abs(np.diff(q, axis=0)).max(axis=1) if len(q) > 1 else np.zeros(0)
    entry["max_joint_step_rad"] = round(float(steps.max()), 4) if steps.size else None

    if skill.name in ("open", "close"):
        before = _joint_values(joints_before, skill.target)
        after = _joint_values(joints_after, skill.target)
        if not after.size:
            entry["done"] = None
            return entry
        if skill.name == "open":
            moved = float(np.abs(after - before).max()) if before.size else 0.0
            need = OPEN_FRACTION * float(np.linalg.norm(trace.get("move_vec", [0.0])))
            entry.update(joint_travel_m=round(moved, 4), done=moved >= need)
        else:
            entry.update(joint_end_m=round(float(np.abs(after).max()), 4),
                         done=float(np.abs(after).max()) <= CLOSED_M)
        return entry

    if held is None or not samples or samples[-1].get("obj") is None:
        entry["done"] = None
        return entry
    obj = np.asarray([s["obj"] for s in samples], dtype=float)
    hand = np.asarray([s["hand"] for s in samples], dtype=float)
    dist = np.linalg.norm(obj - hand, axis=1)

    if skill.name == "pick":
        lift = float(obj[-1][2] - np.asarray(held_start, dtype=float)[2]) if held_start is not None else 0.0
        entry.update(dist_end_m=round(float(dist[-1]), 4), lift_m=round(lift, 4),
                     done=bool(dist[-1] <= HELD_M and lift >= LIFT_M))
        return entry

    release = min(_release_frame(seg), len(dist))
    carry = dist[:release]
    lost = np.nonzero(carry > HELD_M)[0]
    first_loss = int(lost[0]) if lost.size else None
    entry.update(
        release_frame=release if release < len(dist) else None,
        first_loss_frame=first_loss,
        max_dist_before_release_m=round(float(carry.max()), 4) if carry.size else None,
        rocrobo=rocrobo,
        cartesian=[s.get("cartesian") for s in trace.get("motion_trace", {}).get("segments", [])],
    )
    if first_loss is not None:
        window = steps[max(0, first_loss - LOSS_WINDOW):first_loss]
        jump = float(window.max()) if window.size else 0.0
        entry.update(max_joint_step_before_loss_rad=round(jump, 4), done=False,
                     cause="motion?" if jump >= JUMP_RAD else "grip?")
        return entry
    if phase == "transport" or target is None:
        entry["done"] = True
        return entry
    to_target = float(np.linalg.norm(obj[-1][:2] - np.asarray(target, dtype=float)[:2]))
    entry.update(to_target_xy_m=round(to_target, 4), done=to_target <= TARGET_XY_M)
    if not entry["done"]:
        entry["cause"] = "release?"
    return entry


def first_failure(handoffs: list[dict]) -> dict | None:
    """The first handoff that did not finish; for a failed task whose handoffs all passed, the last
    place (the object never came to rest the way the task predicate asks)."""
    failed = next((h for h in handoffs if h.get("done") is False), None)
    if failed is not None:
        return failed
    place = next((h for h in reversed(handoffs) if h["skill"] == "place"), None)
    return None if place is None else {**place, "done": False, "cause": "release?", "by_predicate": True}
