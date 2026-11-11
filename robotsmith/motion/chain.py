"""Motion-chain core — solve a waypoint chain with seed-continuous IK.

``execute_motion_chain`` solves the chain and expands it into a full step-by-step joint-space trajectory
(velocity-adaptive timing). Used by ``MotionExecutor.run``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from robotsmith.motion import gripper, move_linear
from robotsmith.motion.constants import N_ARM_JOINTS as _N_ARM_JOINTS
from robotsmith.motion.params import MotionParams
from robotsmith.motion.plan_state import Attach, Detach, PlanState, attach, detach
from robotsmith.motion.planner import MotionPlanner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeMotionWaypoint:
    """A runtime waypoint for one incoming trajectory segment."""

    pos: np.ndarray
    quat: np.ndarray
    finger_width: float
    label: str
    steps: int | None = None
    finger_only: bool = False
    # A ``move_linear`` segment and its policy on an IK branch flip; ``None`` is a joint-space move.
    linear: Literal["refuse", "keep"] | None = None
    hold_after_steps: int = 0
    allow: str | None = None
    # A ``move_to`` the planner already searched: its points, the last at this waypoint; empty stays put.
    path: tuple[np.ndarray, ...] | None = None


@dataclass
class MotionChainExecution:
    """Generated trajectory plus runtime diagnostics; ``state`` is where the chain leaves the arm and the hand."""

    trajectory: list[np.ndarray]
    q_waypoints: list[np.ndarray]
    state: PlanState
    segments: list[dict] = field(default_factory=list)


def _interpolate(a: np.ndarray, b: np.ndarray, n: int) -> list[np.ndarray]:
    """Linear interpolation in joint space (identical to old _lerp)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return [a + (b - a) * (i + 1) / max(n, 1) for i in range(n)]


def _compute_steps(
    q_from: np.ndarray,
    q_to: np.ndarray,
    vel: float,
    dt: float,
    min_steps: int = 5,
) -> tuple[int, float]:
    """Velocity-adaptive step count for a single segment.

    Returns (steps, max_joint_dist) for diagnostics.
    """
    diff = np.abs(q_to[:_N_ARM_JOINTS] - q_from[:_N_ARM_JOINTS])
    max_dist = float(np.max(diff))
    if max_dist < 1e-6:
        return min_steps, max_dist
    t_min = max_dist / vel
    steps = max(math.ceil(t_min / dt), min_steps)
    return steps, max_dist


def _solve_segment(
    i: int,
    wp: RuntimeMotionWaypoint,
    waypoints: list[RuntimeMotionWaypoint],
    q_from: np.ndarray,
    *,
    motion_planner: MotionPlanner,
) -> tuple[np.ndarray, list[np.ndarray] | None, float]:
    """Solve one segment → (endpoint qpos, dense path | None, arm displacement).

    Routing: finger-only is expanded by ``gripper`` (arm locked); a linear segment is expanded by
    ``move_linear`` (and reports its per-step continuity metric); a planned ``path`` is taken as it is;
    otherwise plain per-pose IK, interpolated in joint space.
    """
    segment_path: list[np.ndarray] | None = None
    if wp.path is not None:
        segment_path = [np.asarray(q, dtype=np.float64).copy() for q in wp.path]
        q_to = segment_path[-1].copy() if segment_path else q_from
    elif wp.finger_only:
        q_to = gripper.expand(gripper.Gripper(wp.pos, wp.quat, wp.finger_width), q_from, motion_planner.solve_ik).q_target
    elif wp.linear is not None:
        if i == 0:
            raise ValueError("cartesian segment requires a previous waypoint")
        prev_wp = waypoints[i - 1]
        path = move_linear.expand(
            move_linear.MoveLinear(prev_wp.pos, wp.pos, wp.quat, wp.finger_width, on_jump=wp.linear, label=wp.label),
            q_from,
            motion_planner.solve_ik,
        )
        segment_path = path.path
        return segment_path[-1].copy(), segment_path, path.max_joint_step
    else:
        q_to = np.asarray(
            motion_planner.solve_ik(wp.pos, wp.quat, wp.finger_width, init_qpos=q_from),
            dtype=np.float64,
        )

    max_joint_step = float(
        np.max(np.abs(q_to[:_N_ARM_JOINTS] - q_from[:_N_ARM_JOINTS]))
    )
    return q_to, segment_path, max_joint_step


def _segment_step_count(
    wp: RuntimeMotionWaypoint,
    q_from: np.ndarray,
    q_to: np.ndarray,
    segment_path: list[np.ndarray] | None,
    params: MotionParams,
) -> int:
    """Frame count for one segment (dense path length, explicit, or adaptive)."""
    if segment_path is not None:
        return len(segment_path)
    if wp.steps is not None:
        return wp.steps
    if wp.finger_only:
        return params.finger_steps
    steps, _ = _compute_steps(q_from, q_to, params.max_joint_vel, params.dt, params.min_steps)
    return steps


def _log_segments(
    waypoints: list[RuntimeMotionWaypoint],
    segments: list[dict],
    params: MotionParams,
) -> None:
    logger.debug(
        "[MotionExecutor] velocity-adaptive steps "
        "(vel=%s rad/s, dt=%.4fs):",
        params.max_joint_vel,
        params.dt,
    )
    for segment in segments:
        i = int(segment["index"])
        src = "home" if i == 0 else waypoints[i - 1].label
        logger.debug(
            "  seg %s: %s → %s  finger_only=%s  joint_dist=%.4f rad  steps=%s",
            i,
            src,
            waypoints[i].label,
            segment["finger_only"],
            segment["max_joint_step"],
            segment["steps"],
        )
    total = sum(int(seg["steps"]) + int(seg["hold_after_steps"]) for seg in segments)
    logger.debug(
        "  total trajectory: %s steps (%.2fs sim time)",
        total,
        total * params.dt,
    )


def _assemble_trajectory(
    waypoints: list[RuntimeMotionWaypoint],
    q_targets: list[np.ndarray],
    segment_paths: list[list[np.ndarray] | None],
    segments: list[dict],
) -> list[np.ndarray]:
    """Expand per-segment endpoints/paths into the full step-by-step trajectory; each segment's ``span`` is
    its ``[start, end)`` in it, its hold included, and ``reach`` the
    point where its waypoint ``pose`` is reached (``None`` when the segment adds no point)."""
    traj: list[np.ndarray] = []
    for i, wp in enumerate(waypoints):
        start = len(traj)
        path = segment_paths[i]
        if path is not None:
            traj += path
        else:
            traj += _interpolate(q_targets[i], q_targets[i + 1], int(segments[i]["steps"]))
        moved = len(traj) - start
        if wp.hold_after_steps > 0:
            traj += [q_targets[i + 1].copy() for _ in range(wp.hold_after_steps)]
        segments[i]["span"] = [start, len(traj)]
        segments[i]["pose"] = (np.asarray(wp.pos, dtype=np.float64), np.asarray(wp.quat, dtype=np.float64))
        segments[i]["reach"] = start + max(moved - 1, 0) if len(traj) > start else None
    return traj


def execute_motion_chain(
    actions: Sequence[RuntimeMotionWaypoint | Attach | Detach],
    motion_planner: MotionPlanner,
    start: PlanState,
    params: MotionParams,
) -> MotionChainExecution:
    """Execute an action list with seed-continuous IK and diagnostics.

    Per-pose IK goes through ``motion_planner.solve_ik``; nothing here checks
    collisions. Each segment starts from the plan state's ``q`` and advances it to its end configuration; a
    segment is ``holding`` when the plan state holds something while it runs. ``Attach`` / ``Detach`` change the
    holding where they stand in the list, after the previous segment and its hold.
    """
    state = PlanState(np.asarray(start.q, dtype=np.float64).copy(), start.holding)
    q_targets = [state.q]
    segment_paths: list[list[np.ndarray] | None] = []
    segments: list[dict] = []
    waypoints: list[RuntimeMotionWaypoint] = []

    for wp in actions:
        if isinstance(wp, Attach):
            state = attach(state, wp.holding)
            continue
        if isinstance(wp, Detach):
            state = detach(state)
            continue
        i = len(waypoints)
        waypoints.append(wp)
        q_from = state.q
        q_to, segment_path, max_joint_step = _solve_segment(
            i,
            wp,
            waypoints,
            q_from,
            motion_planner=motion_planner,
        )
        q_targets.append(q_to)
        segment_paths.append(segment_path)
        segments.append(
            {
                "index": i,
                "label": wp.label,
                "steps": _segment_step_count(wp, q_from, q_to, segment_path, params),
                "max_joint_step": max_joint_step,
                "cartesian": wp.linear is not None,
                "finger_only": wp.finger_only,
                "hold_after_steps": wp.hold_after_steps,
                "holding": state.holding is not None,
                "allow": wp.allow,
            }
        )
        state = PlanState(q_to, state.holding)

    _log_segments(waypoints, segments, params)
    return MotionChainExecution(
        trajectory=_assemble_trajectory(waypoints, q_targets, segment_paths, segments),
        q_waypoints=q_targets,
        state=state,
        segments=segments,
    )


def execution_trace(action: str, result: MotionChainExecution) -> dict:
    return {
        "action": action,
        "segments": result.segments,
        "q_waypoints": [
            [float(v) for v in np.asarray(q, dtype=np.float64)]
            for q in result.q_waypoints
        ],
    }
