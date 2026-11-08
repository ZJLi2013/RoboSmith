"""The pick and place action lists, run by ``MotionExecutor.run``.

An action is a motion segment (``RuntimeMotionWaypoint``: ``finger_only`` is a ``gripper``, ``linear`` a
``move_linear``, otherwise a ``move_to``) or an ``Attach`` / ``Detach`` of the plan state, which moves nothing.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from robotsmith.execution.gate import Holding
from robotsmith.grasp.planner import GraspPlan
from robotsmith.motion import gripper
from robotsmith.motion.chain import RuntimeMotionWaypoint, classify_segments
from robotsmith.motion.params import MotionParams
from robotsmith.motion.plan_state import Attach, Detach

Action = RuntimeMotionWaypoint | Attach | Detach


def pick_actions(
    plan: GraspPlan,
    home_qpos: np.ndarray,
    params: MotionParams,
    solve_ik: Callable,
    *,
    grasp: Holding,
) -> list[Action]:
    """The plan's waypoints with velocity-adaptive step counts; the final approach segment is Cartesian.
    ``home_qpos`` is the arm already at the first waypoint, so that segment only opens the fingers, in zero
    steps when they already are. ``grasp`` is attached once the last finger segment and its settle end.

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
    if not wps:
        raise ValueError("GraspPlan.pick requires a non-empty waypoint sequence")
    is_finger, fine_idx = classify_segments(wps)
    last_finger = max((i for i, f in enumerate(is_finger) if f), default=-1)
    opened = gripper.expand(gripper.Gripper(wps[0].pos, wps[0].quat, wps[0].finger_width), home_qpos,
                            solve_ik).q_target
    open_steps = 0 if np.allclose(opened[7:], np.asarray(home_qpos, dtype=np.float64)[7:]) else None

    actions: list[Action] = []
    for i in range(len(wps)):
        if is_finger[i]:
            segment_type = "finger"
        elif i == fine_idx:
            segment_type = "fine"
        else:
            segment_type = "normal"
        actions.append(
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
                allow=("grip" if is_finger[i]
                       else "rest" if last_finger >= 0 and i == last_finger + 1 else None),
            )
        )
        if i == last_finger:
            actions.append(Attach(grasp))
    return actions


def place_actions(place_plan: GraspPlan, params: MotionParams, *, phase: str = "all") -> list[Action]:
    """transport → pre_place → place (open fingers) → retreat.

    The start state holds the object (fingers closed); it is detached once the release ends.
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

    The ``place`` skill plans the free move to the standoff itself and runs this with
    ``"descend"`` from there. The object is held until the release; descend and release may rest it on the target, and the
    retreat may touch it with the fingers.
    """
    transport: list[Action] = [RuntimeMotionWaypoint(
        pos=place_plan.pre_grasp_pos,
        quat=place_plan.pre_grasp_quat,
        finger_width=place_plan.finger_closed,
        label="transport",
        steps=params.transport_steps,
    )]
    if phase == "transport":
        return transport
    # "descend" or "all": (corrective) transport then vertical descend
    return [
        *transport,
        RuntimeMotionWaypoint(
            pos=place_plan.grasp_pos,
            quat=place_plan.grasp_quat,
            finger_width=place_plan.finger_closed,
            label="descend",
            linear="keep",
            steps=params.place_descend_steps,
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
            allow="rest",
        ),
        Detach(),
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
