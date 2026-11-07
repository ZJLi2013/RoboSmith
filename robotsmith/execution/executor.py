"""MotionExecutor — generates joint-space trajectories from GraspPlans.

Extracted from the old PickStrategy / PickAndPlaceStrategy / StackStrategy.
The executor knows *nothing* about object categories or grasp semantics;
it only converts 6-DoF waypoints (from GraspPlan) into IK-solved joint
targets, delegating the chain solve + trajectory expansion to
``motion.chain`` (``execute_motion_chain``).

Waypoint path uses **velocity-adaptive** step counts:
  steps = max(ceil(max_joint_dist / (vel * dt)), min_steps)
so large moves get more steps and short moves fewer, automatically.
"""

from __future__ import annotations

import logging

import numpy as np

from robotsmith.grasp.planner import GraspPlan
from robotsmith.motion.chain import (
    MotionChainExecution,
    RuntimeMotionWaypoint,
    classify_segments,
    execute_motion_chain,
    execution_trace,
)
from robotsmith.motion.params import MotionParams
from robotsmith.motion.planner import MotionPlanner

logger = logging.getLogger(__name__)

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
            label="grasp", segment_type="finger", finger_only=True,
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
            segment_type="finger", finger_only=True,
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
    """Convert GraspPlan(s) + IK solver into joint-space trajectories."""

    def __init__(self) -> None:
        self.last_motion_trace: dict | None = None

    def pick(
        self,
        plan: GraspPlan,
        motion_planner: MotionPlanner,
        home_qpos: np.ndarray,
        params: MotionParams,
    ) -> MotionChainExecution:
        """waypoints → hold (velocity-adaptive step counts); the final approach segment is Cartesian.
        ``home_qpos`` is the arm already at the first waypoint, so that segment only opens the fingers, in zero
        steps when they already are."""
        if not plan.waypoints:
            raise ValueError("GraspPlan.pick requires a non-empty waypoint sequence")
        return self._pick_waypoints(plan, motion_planner, home_qpos, params)

    def _pick_waypoints(
        self,
        plan: GraspPlan,
        motion_planner: MotionPlanner,
        home_qpos: np.ndarray,
        params: MotionParams,
    ) -> MotionChainExecution:
        """Execute an ordered waypoint sequence with velocity-adaptive timing.

        Segment classification (determines velocity):
          **finger** — consecutive waypoints with same Cartesian pose but
                       different finger_width.  Arm joints are locked to the
                       previous solution; uses fixed ``finger_steps``.
          **fine**   — the last arm-movement segment *before* the first finger
                       segment (i.e. the final approach to the grasp point).
                       Uses ``fine_joint_vel``.
          **normal** — all other arm-movement segments (approach, retreat).
                       Uses ``max_joint_vel``.

        Finger segments may touch the target (``allow="grip"``); after the last one the target is
        held, and the first segment after it may still touch the support it leaves (``allow="rest"``).
        """
        wps = plan.waypoints
        assert wps, "waypoints must be non-empty"

        is_finger, fine_idx = classify_segments(wps)
        last_finger = max((i for i, f in enumerate(is_finger) if f), default=-1)
        opened = np.asarray(motion_planner.solve_ik(wps[0].pos, wps[0].quat, wps[0].finger_width), dtype=np.float64)
        open_steps = 0 if np.allclose(opened[7:], np.asarray(home_qpos, dtype=np.float64)[7:]) else None

        runtime_wps: list[RuntimeMotionWaypoint] = []
        for i in range(len(wps)):
            if is_finger[i]:
                segment_type = "finger"
            elif i == fine_idx:
                segment_type = "fine"
            else:
                segment_type = "normal"
            runtime_wps.append(
                RuntimeMotionWaypoint(
                    pos=wps[i].pos,
                    quat=wps[i].quat,
                    finger_width=wps[i].finger_width,
                    label=f"wp{i}",
                    segment_type=segment_type,
                    steps=open_steps if i == 0 else None,
                    finger_only=is_finger[i] or i == 0,
                    linear="refuse" if i == fine_idx else None,
                    hold_after_steps=(
                        params.grasp_settle_steps if i == last_finger else 0
                    ),
                    holding=0 <= last_finger < i,
                    allow=("grip" if is_finger[i]
                           else "rest" if last_finger >= 0 and i == last_finger + 1 else None),
                )
            )

        result = execute_motion_chain(
            runtime_wps,
            motion_planner,
            home_qpos,
            params,
            tail_hold_steps=params.lift_hold_steps,
        )
        self.last_motion_trace = execution_trace("pick", result)
        if last_finger >= 0 and params.grasp_settle_steps > 0:
            logger.debug(
                "  grasp_settle: %s steps after seg %s (%.2fs hold)",
                params.grasp_settle_steps,
                last_finger,
                params.grasp_settle_steps * params.dt,
            )
        return result

    def place(
        self,
        place_plan: GraspPlan,
        motion_planner: MotionPlanner,
        start_qpos: np.ndarray,
        params: MotionParams,
        *,
        phase: str = "all",
    ) -> MotionChainExecution:
        """transport → pre_place → place (open fingers) → retreat.

        Assumes the robot is holding an object (fingers closed) at start_qpos.
        Finger widths come from place_plan: finger_closed while transporting,
        finger_open on release.

        ``transport`` (to the standoff = one clearance back along the approach axis
        from the drop point) is a joint-space move. ``descend`` (insert standoff→drop)
        and ``retreat`` (drop→standoff) are Cartesian straight lines **along the
        approach axis** — the geometry is baked into ``pre_grasp_pos`` /
        ``grasp_pos`` / ``retreat_pos`` by ``plan_place``. For a top-down place
        (``approach=-Z``) the standoff is straight above and these are pure vertical
        moves (drawer / tabletop, unchanged); for a side tuck-under the same
        Cartesian segments run horizontally along the approach axis. A Cartesian
        insert keeps the contact move predictable (no planner detour grazing the
        slot) and lands the gripper at the authored release pose for the next skill.

        ``phase`` selects which execution slice to plan, so the runtime can step
        ``transport`` into the sim, re-sense the (possibly recoiled) place anchor,
        then plan ``descend`` from the settled pose (live re-anchoring):

        - ``"transport"``: the inbound free move to straight above the target only.
        - ``"descend"``: a corrective ``transport`` to the *re-anchored* point above
          the target (laps up any drift since the first transport) then the
          Cartesian-vertical descend → release → retreat. Keeping the corrective
          transport is what preserves verticality: descending from a stale x/y
          would slant the Cartesian line into the drawer wall.
        - ``"all"`` (default): the full transport → descend → release → retreat in
          one open-loop pass (legacy behavior; static anchors, tests).

        The ``place`` skill plans the free move to the standoff itself and calls this with
        ``"descend"`` from there. The object is held until the release; descend and release may rest it on the target, and the
        retreat may touch it with the fingers.
        """
        transport_wps = [RuntimeMotionWaypoint(
            pos=place_plan.pre_grasp_pos,
            quat=place_plan.pre_grasp_quat,
            finger_width=place_plan.finger_closed,
            label="transport",
            steps=params.transport_steps,
            holding=True,
        )]
        descend_wps = [
            RuntimeMotionWaypoint(
                pos=place_plan.grasp_pos,
                quat=place_plan.grasp_quat,
                finger_width=place_plan.finger_closed,
                label="descend",
                linear="keep",
                steps=params.place_descend_steps,
                holding=True,
                allow="rest",
            ),
            RuntimeMotionWaypoint(
                pos=place_plan.grasp_pos,
                quat=place_plan.grasp_quat,
                finger_width=place_plan.finger_open,
                label="release",
                segment_type="finger",
                steps=params.release_steps,
                finger_only=True,
                holding=True,
                allow="rest",
            ),
            RuntimeMotionWaypoint(
                pos=place_plan.retreat_pos,
                quat=place_plan.retreat_quat,
                finger_width=place_plan.finger_open,
                label="retreat",
                linear="keep",
                steps=params.retreat_steps,
                allow="grip",
            ),
        ]
        if phase == "transport":
            runtime_wps = transport_wps
        else:  # "descend" or "all": (corrective) transport then vertical descend
            runtime_wps = [*transport_wps, *descend_wps]
        result = execute_motion_chain(
            runtime_wps,
            motion_planner,
            start_qpos,
            params,
        )
        self.last_motion_trace = execution_trace("place", result)
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
        result = execute_motion_chain(
            runtime_wps,
            motion_planner,
            start_qpos,
            params,
        )
        self.last_motion_trace = execution_trace(action, result)
        return result.trajectory
