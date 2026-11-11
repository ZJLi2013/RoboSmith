"""MotionExecutor — runs action lists into joint-space trajectories.

``run`` is the one way an action list (the pick and place lists of ``execution.meta_actions``, the drag
segments here) reaches ``motion.chain.execute_motion_chain``. The executor knows *nothing* about object
categories or grasp semantics.

Waypoint path uses **velocity-adaptive** step counts:
  steps = max(ceil(max_joint_dist / (vel * dt)), min_steps)
so large moves get more steps and short moves fewer, automatically.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from robotsmith.execution.meta_actions import Action
from robotsmith.motion.chain import (
    MotionChainExecution,
    RuntimeMotionWaypoint,
    execute_motion_chain,
    execution_trace,
)
from robotsmith.motion.params import MotionParams
from robotsmith.motion.plan_state import PlanState
from robotsmith.motion.planner import MotionPlanner

# Closing pushes the ball handle from the pull side: start the closed gripper this
# far back (world meters) so its fingertips contact the ball, then sweep through.
_PUSH_CONTACT_GAP = 0.05


def _open_drag_waypoints(
    handle: np.ndarray,
    move: np.ndarray,
    grasp_quat: np.ndarray,
    *,
    finger_open: float,
    finger_closed: float,
    approach_height: float,
    retreat_height: float,
    params: MotionParams,
) -> list[RuntimeMotionWaypoint]:
    """Open: cage the handle neck and pull, then disengage straight up.

    approach (above handle) -> descend (vertical onto knob) -> grasp (close) ->
    drag (cartesian by ``move``) -> lift (straight up, fingers STILL closed,
    sliding off the neck with no horizontal push) -> release (open in the air).
    The lift precedes the release because opening the fingers while still caging
    the neck kicks the drawer back (stored contact energy).
    """
    target = handle + move
    back = np.array([0.0, 0.0, 1.0])  # top-down: approach/disengage straight up
    return [
        RuntimeMotionWaypoint(
            pos=handle + back * approach_height, quat=grasp_quat,
            finger_width=finger_open, label="approach",
            steps=params.approach_steps,
        ),
        RuntimeMotionWaypoint(
            pos=handle, quat=grasp_quat, finger_width=finger_open,
            # Straight-down Cartesian: descend onto the handle vertically
            # (no horizontal force, no joint-space curve); goal on the handle.
            label="descend", linear="keep",
        ),
        RuntimeMotionWaypoint(
            pos=handle, quat=grasp_quat, finger_width=finger_closed,
            label="grasp", finger_only=True,
            steps=params.grasp_hold_steps,
            hold_after_steps=params.grasp_settle_steps,
        ),
        RuntimeMotionWaypoint(
            pos=target, quat=grasp_quat, finger_width=finger_closed,
            label="drag", linear="keep",
        ),
        RuntimeMotionWaypoint(
            pos=target + back * retreat_height, quat=grasp_quat,
            finger_width=finger_closed, label="lift",
            # Straight-up Cartesian lift: keeps the gripper vertical so it
            # slides off the neck with no horizontal force on the drawer,
            # then dwell so the drawer settles before the fingers open.
            linear="keep",
            hold_after_steps=params.contact_settle_steps,
        ),
        RuntimeMotionWaypoint(
            pos=target + back * retreat_height, quat=grasp_quat,
            finger_width=finger_open, label="release",
            finger_only=True,
            steps=params.release_steps,
        ),
    ]


def _close_drag_waypoints(
    handle: np.ndarray,
    move: np.ndarray,
    grasp_quat: np.ndarray,
    *,
    finger_closed: float,
    approach_height: float,
    retreat_height: float,
    params: MotionParams,
) -> list[RuntimeMotionWaypoint]:
    """Close: push with a CLOSED gripper through the ball from the pull side.

    The spherical handle is a *pull* stop only: caging the neck and pushing slips
    off it (and over-extends the arm into the cabinet). So drive a closed gripper
    straight through the ball — the rigid ball, and the drawer behind it, is
    pushed shut by contact. No grasp/release; fingers stay closed throughout.
    approach (above contact) -> descend -> push (cartesian to ``target``) -> retreat.
    """
    target = handle + move
    back = np.array([0.0, 0.0, 1.0])  # top-down: approach/disengage straight up
    move_norm = float(np.linalg.norm(move))
    push_dir = move / move_norm if move_norm > 1e-9 else move
    contact = handle - push_dir * _PUSH_CONTACT_GAP
    return [
        RuntimeMotionWaypoint(
            pos=contact + back * approach_height, quat=grasp_quat,
            finger_width=finger_closed, label="approach",
            steps=params.approach_steps,
        ),
        RuntimeMotionWaypoint(
            pos=contact, quat=grasp_quat, finger_width=finger_closed,
            # Straight-down Cartesian: descend onto the contact point
            # vertically (no horizontal force, no joint-space curve into
            # the cabinet); the goal sits on the handle.
            label="descend", linear="keep",
        ),
        RuntimeMotionWaypoint(
            pos=target, quat=grasp_quat, finger_width=finger_closed,
            label="push", linear="keep",
            # Dwell in contact so the drawer's velocity decays before the
            # gripper backs off, else stored contact energy springs it open
            # again.
            hold_after_steps=params.contact_settle_steps,
        ),
        RuntimeMotionWaypoint(
            pos=target + back * retreat_height, quat=grasp_quat,
            finger_width=finger_closed, label="retreat",
            # Straight-up Cartesian disengage: keeps the gripper vertical
            # so backing off the handle applies no horizontal force on the
            # drawer (a planned retreat curves into the front panel).
            linear="keep",
        ),
    ]


class MotionExecutor:
    """Run action lists through the motion chain and keep the last one's trace."""

    def __init__(self) -> None:
        self.last_motion_trace: dict | None = None

    def run(
        self,
        name: str,
        actions: Sequence[Action],
        motion_planner: MotionPlanner,
        start: PlanState,
        params: MotionParams,
    ) -> MotionChainExecution:
        """Run an action list from ``start``, in order, and keep its trace as ``last_motion_trace``."""
        result = execute_motion_chain(actions, motion_planner, start, params)
        self.last_motion_trace = execution_trace(name, result)
        return result

    def drag_handle(
        self,
        handle_pos: np.ndarray,
        grasp_quat: np.ndarray,
        move_vec: np.ndarray,
        motion_planner: MotionPlanner,
        start_qpos: np.ndarray,
        params: MotionParams,
        *,
        finger_open: float = 0.04,
        finger_closed: float = 0.0,
        # Top-down approach height above the knob; at 0.10 the straight-line
        # approach punched the drawer front. descend (Cartesian) does the contact.
        approach_height: float = 0.28,
        retreat_height: float = 0.10,
        action: str = "open",
    ) -> list[np.ndarray]:
        """Move an articulated handle along ``move_vec`` (open/close a drawer).

        ``action="open"`` cages the handle neck and pulls: approach (above handle)
        -> descend -> grasp (close fingers) -> drag (cartesian by ``move_vec``) ->
        lift (straight up, fingers STILL closed, sliding off the neck without any
        horizontal push) -> release (open in the air); the ball is the
        pull-direction stop (form closure). Opening the fingers while still caging
        the neck kicks the drawer back (stored contact energy), so the lift comes
        first and the release happens clear of the handle.

        ``action="close"`` instead pushes with a CLOSED gripper: approach above a
        point one ``_PUSH_CONTACT_GAP`` back on the pull side -> descend ->
        push (cartesian through to ``target``) -> retreat. Pushing a caged ball
        slips, so closing relies on rigid contact, not a grasp.

        Every contact-adjacent segment is a Cartesian straight line: ``descend``
        (straight down onto the handle), the contact ``drag``/``push``, and the
        disengage ``lift``/``retreat`` (straight up). Keeping these vertical means
        the gripper applies no horizontal force on the drawer and a joint-space
        move never curves the wrist into the drawer front. An IK branch flip on them is
        kept (``linear="keep"``), not refused: the motion is position-controlled against
        a held handle, so a reconfiguration is tracked instead of aborting the episode.
        """
        handle = np.asarray(handle_pos, dtype=np.float64)
        move = np.asarray(move_vec, dtype=np.float64)
        if action == "close":
            runtime_wps = _close_drag_waypoints(
                handle, move, grasp_quat,
                finger_closed=finger_closed,
                approach_height=approach_height,
                retreat_height=retreat_height,
                params=params,
            )
        else:
            runtime_wps = _open_drag_waypoints(
                handle, move, grasp_quat,
                finger_open=finger_open,
                finger_closed=finger_closed,
                approach_height=approach_height,
                retreat_height=retreat_height,
                params=params,
            )
        return self.run(action, runtime_wps, motion_planner, PlanState(start_qpos), params).trajectory
