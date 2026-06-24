"""Motion-chain core — solve a waypoint chain with seed-continuous IK.

A waypoint chain is solved once and consumed two ways:

- ``evaluate_motion_chain`` — **pre-runtime feasibility**: solve the chain and
  return IK diagnostics (joint margin, total distance, final-approach step) without
  running a rollout. Used by grasp scoring.
- ``execute_motion_chain`` — **runtime**: solve the chain and expand it into a full
  step-by-step joint-space trajectory (velocity-adaptive timing, optional
  collision-free ``plan_motion`` sub-paths). Used by ``MotionExecutor``.

Both share segment classification (``classify_segments`` / ``_same_pose``): a
finger-only waypoint repeats the previous EE pose with a different finger width,
and the *fine* segment is the final approach onto the grasp.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from robotsmith.sim.franka import Q_LOWER as _FRANKA_Q_LOWER
from robotsmith.sim.franka import Q_UPPER as _FRANKA_Q_UPPER
from robotsmith.motion.cartesian import (
    DEFAULT_CARTESIAN_MAX_JOINT_STEP,
    solve_cartesian_ik_path,
)
from robotsmith.motion.constants import N_ARM_JOINTS as _N_ARM_JOINTS
from robotsmith.motion.params import MotionParams
from robotsmith.motion.planner import MotionPlanner, Waypoint

logger = logging.getLogger(__name__)

_FRANKA_Q_RANGE = _FRANKA_Q_UPPER - _FRANKA_Q_LOWER


@dataclass(frozen=True)
class MotionChainMetrics:
    """Diagnostics from solving a waypoint chain without running a rollout."""

    ok: bool
    min_joint_margin: float
    total_joint_dist: float
    final_approach_joint_step: float


def _ik_joint_margin(q_arm: np.ndarray) -> float:
    """Return min normalized joint margin to limits across 7 arm DoFs."""
    q = np.asarray(q_arm, dtype=np.float64)[:_N_ARM_JOINTS]
    lower_dist = (q - _FRANKA_Q_LOWER) / _FRANKA_Q_RANGE
    upper_dist = (_FRANKA_Q_UPPER - q) / _FRANKA_Q_RANGE
    return float(np.min(np.minimum(lower_dist, upper_dist)))


def _same_pose(a, b, pos_atol: float = 1e-4, quat_atol: float = 1e-3) -> bool:
    """True if two waypoints share the same EE pose, differing only in finger_width."""
    return bool(
        np.allclose(a.pos, b.pos, atol=pos_atol)
        and np.allclose(a.quat, b.quat, atol=quat_atol)
    )


def classify_segments(waypoints: Sequence) -> tuple[list[bool], int]:
    """Mark finger-only waypoints and the fine (final-approach) segment.

    - **finger-only**: same EE pose as the previous waypoint, differing only in
      ``finger_width`` (the arm is locked; the gripper opens/closes in place).
    - **fine**: the arm segment immediately before the first finger-only waypoint
      (the final approach onto the grasp point); ``-1`` when there is none.
    """
    is_finger = [
        i > 0 and _same_pose(waypoints[i - 1], waypoints[i])
        for i in range(len(waypoints))
    ]
    first_finger = next((i for i, f in enumerate(is_finger) if f), len(waypoints))
    fine_idx = first_finger - 1 if first_finger > 1 else -1
    return is_finger, fine_idx


def evaluate_motion_chain(
    waypoints: Sequence,
    solve_ik: Callable,
    init_qpos: np.ndarray,
    *,
    max_final_approach_joint_step: float,
) -> MotionChainMetrics:
    """Solve a waypoint chain with seed continuity and return IK diagnostics."""
    qs = [np.asarray(init_qpos, dtype=np.float64).copy()]
    ok = True
    for i, wp in enumerate(waypoints):
        if i > 0 and _same_pose(waypoints[i - 1], wp):
            q = qs[-1].copy()
            finger_q = solve_ik(wp.pos, wp.quat, wp.finger_width)
            q[_N_ARM_JOINTS:] = finger_q[_N_ARM_JOINTS:]
        else:
            q = solve_ik(wp.pos, wp.quat, wp.finger_width, init_qpos=qs[-1])
        q = np.asarray(q, dtype=np.float64)
        if not np.all(np.isfinite(q)):
            ok = False
        qs.append(q)

    arm_qs = np.array([q[:_N_ARM_JOINTS] for q in qs[1:]], dtype=np.float64)
    margins = [_ik_joint_margin(q) for q in arm_qs]
    diffs = [
        float(np.max(np.abs(qs[i + 1][:_N_ARM_JOINTS] - qs[i][:_N_ARM_JOINTS])))
        for i in range(len(qs) - 1)
    ]

    final_approach_step = 0.0
    final_approach_ok = True
    _, fine_idx = classify_segments(waypoints)
    if fine_idx > 0:
        start_wp = waypoints[fine_idx - 1]
        end_wp = waypoints[fine_idx]
        path = solve_cartesian_ik_path(
            start_wp.pos,
            end_wp.pos,
            end_wp.quat,
            end_wp.finger_width,
            qs[fine_idx],
            solve_ik,
            max_joint_step=max_final_approach_joint_step,
        )
        final_approach_step = path.max_joint_step
        final_approach_ok = path.ok

    return MotionChainMetrics(
        ok=bool(ok and final_approach_ok),
        min_joint_margin=float(np.min(margins)) if margins else -np.inf,
        total_joint_dist=float(np.sum(diffs)),
        final_approach_joint_step=final_approach_step,
    )


# --------------------------------------------------------------------------- #
# Runtime trajectory generation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RuntimeMotionWaypoint:
    """A runtime waypoint for one incoming trajectory segment."""

    pos: np.ndarray
    quat: np.ndarray
    finger_width: float
    label: str
    segment_type: str = "normal"
    steps: int | None = None
    finger_only: bool = False
    cartesian: bool = False
    hold_after_steps: int = 0


@dataclass
class MotionChainExecution:
    """Generated trajectory plus runtime diagnostics."""

    trajectory: list[np.ndarray]
    q_waypoints: list[np.ndarray]
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
    world: list[dict] | None,
    attach: dict | None,
    params: MotionParams,
    raise_on_cartesian_discontinuity: bool,
) -> tuple[np.ndarray, list[np.ndarray] | None, float]:
    """Solve one segment → (endpoint qpos, dense path | None, arm displacement).

    Routing: finger-only locks the arm; cartesian walks a continuity-checked
    Cartesian path (and reports its per-step continuity metric); a free segment
    with a ``world`` tries collision-free ``plan_motion`` (straight-line fallback
    on miss); otherwise plain per-pose IK.
    """

    def _ik(q_seed: np.ndarray) -> np.ndarray:
        return np.asarray(
            motion_planner.solve_ik(wp.pos, wp.quat, wp.finger_width, init_qpos=q_seed),
            dtype=np.float64,
        )

    segment_path: list[np.ndarray] | None = None
    if wp.finger_only:
        q_to = q_from.copy()
        finger_q = motion_planner.solve_ik(wp.pos, wp.quat, wp.finger_width)
        q_to[_N_ARM_JOINTS:] = finger_q[_N_ARM_JOINTS:]
    elif wp.cartesian:
        if i == 0:
            raise ValueError("cartesian segment requires a previous waypoint")
        prev_wp = waypoints[i - 1]
        path = solve_cartesian_ik_path(
            prev_wp.pos,
            wp.pos,
            wp.quat,
            wp.finger_width,
            q_from,
            motion_planner.solve_ik,
            max_joint_step=DEFAULT_CARTESIAN_MAX_JOINT_STEP,
            raise_on_discontinuity=raise_on_cartesian_discontinuity,
            error_label=wp.label,
        )
        segment_path = path.path
        return segment_path[-1].copy(), segment_path, path.max_joint_step
    elif world is not None:
        plan = motion_planner.plan_motion(
            q_from,
            [Waypoint(wp.pos, wp.quat, wp.finger_width)],
            world=world,
            control_dt=params.dt,
            attach=attach,
        )
        if plan.success and plan.trajectory:
            segment_path = [np.asarray(q, dtype=np.float64) for q in plan.trajectory]
            q_to = segment_path[-1].copy()
        else:
            # Collision-blind straight-line last resort. Endpoint geometry is
            # authored to be reachable upstream (adaptive transport hover +
            # lift-then-swing), so a miss here means the goal itself is
            # unreachable/inside a keep-out — loud, because this segment then
            # loses its collision guarantee.
            logger.warning(
                "[MotionExecutor] plan_motion miss on %s (%s); "
                "collision-blind straight-line fallback",
                wp.label,
                plan.reason,
            )
            q_to = _ik(q_from)
    else:
        q_to = _ik(q_from)

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
    vel = params.fine_joint_vel if wp.segment_type == "fine" else params.max_joint_vel
    steps, _ = _compute_steps(q_from, q_to, vel, params.dt, params.min_steps)
    return steps


def _log_segments(
    waypoints: list[RuntimeMotionWaypoint],
    segments: list[dict],
    params: MotionParams,
    tail_hold_steps: int,
) -> None:
    logger.debug(
        "[MotionExecutor] velocity-adaptive steps "
        "(vel=%s/%s rad/s, dt=%.4fs):",
        params.max_joint_vel,
        params.fine_joint_vel,
        params.dt,
    )
    for segment in segments:
        i = int(segment["index"])
        src = "home" if i == 0 else waypoints[i - 1].label
        logger.debug(
            "  seg %s: %s → %s  type=%-6s  joint_dist=%.4f rad  steps=%s",
            i,
            src,
            waypoints[i].label,
            segment["type"],
            segment["max_joint_step"],
            segment["steps"],
        )
    total = sum(int(seg["steps"]) + int(seg["hold_after_steps"]) for seg in segments)
    total += tail_hold_steps
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
    tail_hold_steps: int,
) -> list[np.ndarray]:
    """Expand per-segment endpoints/paths into the full step-by-step trajectory."""
    traj: list[np.ndarray] = []
    for i, wp in enumerate(waypoints):
        path = segment_paths[i]
        if path is not None:
            traj += path
        else:
            traj += _interpolate(q_targets[i], q_targets[i + 1], int(segments[i]["steps"]))
        if wp.hold_after_steps > 0:
            traj += [q_targets[i + 1].copy() for _ in range(wp.hold_after_steps)]
    if tail_hold_steps > 0:
        traj += [q_targets[-1].copy() for _ in range(tail_hold_steps)]
    return traj


def execute_motion_chain(
    waypoints: list[RuntimeMotionWaypoint],
    motion_planner: MotionPlanner,
    start_qpos: np.ndarray,
    params: MotionParams,
    *,
    tail_hold_steps: int = 0,
    raise_on_cartesian_discontinuity: bool = True,
    world: list[dict] | None = None,
    attach: dict | None = None,
) -> MotionChainExecution:
    """Execute runtime waypoints with seed-continuous IK and diagnostics.

    The motion seam is a single ``MotionPlanner``: per-pose IK goes through
    ``motion_planner.solve_ik`` (the collision-blind ``GenesisBackend`` when no
    collision-aware backend is wired). ``raise_on_cartesian_discontinuity`` gates
    free-space Cartesian segments (pick/place) where an IK branch flip would whip
    the arm. Contact drags (open/close) set it False: the motion is
    position-controlled against a held handle, so a reconfiguration is tracked
    instead of aborting the episode.

    With a ``world``, each free-space arm segment (not finger-only, not the
    contact-drag cartesian path) is routed through ``motion_planner.plan_motion``
    for a **collision-free** sub-path that avoids ``world`` (the cabinet/drawer
    envelope) instead of straight-line interpolation. A segment the planner cannot
    solve (incl. the collision-blind fallback) uses ``solve_ik`` + interpolation.
    """
    q_targets = [np.asarray(start_qpos, dtype=np.float64).copy()]
    segment_paths: list[list[np.ndarray] | None] = []
    segments: list[dict] = []

    for i, wp in enumerate(waypoints):
        q_from = q_targets[-1]
        q_to, segment_path, max_joint_step = _solve_segment(
            i,
            wp,
            waypoints,
            q_from,
            motion_planner=motion_planner,
            world=world,
            attach=attach,
            params=params,
            raise_on_cartesian_discontinuity=raise_on_cartesian_discontinuity,
        )
        q_targets.append(q_to)
        segment_paths.append(segment_path)
        segments.append(
            {
                "index": i,
                "label": wp.label,
                "type": wp.segment_type,
                "steps": _segment_step_count(wp, q_from, q_to, segment_path, params),
                "max_joint_step": max_joint_step,
                "cartesian": wp.cartesian,
                "finger_only": wp.finger_only,
                "hold_after_steps": wp.hold_after_steps,
            }
        )

    _log_segments(waypoints, segments, params, tail_hold_steps)
    return MotionChainExecution(
        trajectory=_assemble_trajectory(
            waypoints, q_targets, segment_paths, segments, tail_hold_steps
        ),
        q_waypoints=q_targets,
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
