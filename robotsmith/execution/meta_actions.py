"""The pick and place action lists, run by ``MotionExecutor.run``, in the order and with the meaning of the public
plans' actions (``scenarios/acceptance/plans``).

An action is a motion segment (``RuntimeMotionWaypoint``: ``finger_only`` is a ``gripper``, ``linear`` a
``move_linear``, otherwise a ``move_to``) or an ``Attach`` / ``Detach`` of the plan state, which moves nothing.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from robotsmith.execution.gate import Holding
from robotsmith.grasp.planner import GraspPlan
from robotsmith.motion import gripper
from robotsmith.motion.chain import RuntimeMotionWaypoint
from robotsmith.motion.params import MotionParams
from robotsmith.motion.plan_state import Attach, Detach

Action = RuntimeMotionWaypoint | Attach | Detach

LIFT = np.array([0.0, 0.0, 0.10])


def pick_actions(
    plan: GraspPlan,
    start_q: np.ndarray,
    start_hand: tuple[np.ndarray, np.ndarray],
    approach: Sequence[np.ndarray],
    params: MotionParams,
    solve_ik: Callable,
    *,
    grasp: Holding,
) -> list[Action]:
    """gripper open → move_to pre-grasp → move_linear to the grasp → gripper close → attach → move_linear ``LIFT``.

    The fingers open where the arm is at ``start_q`` (``start_hand`` is its hand pose there), in zero steps when they
    already are. ``approach`` is the planned joint path from there to the pre-grasp configuration, empty when
    ``start_q`` already is it. The close may touch the target (``allow="grip"``) and settles; ``grasp`` is held from
    then on, and the lift may still touch the support it leaves (``allow="rest"``) and holds for ``lift_hold_steps``.
    """
    opened = gripper.expand(gripper.Gripper(*start_hand, plan.finger_open), start_q, solve_ik).q_target
    already_open = np.allclose(opened[7:], np.asarray(start_q, dtype=np.float64)[7:])
    return [
        RuntimeMotionWaypoint(pos=start_hand[0], quat=start_hand[1], finger_width=plan.finger_open, label="open",
                              segment_type="finger", steps=0 if already_open else None, finger_only=True),
        RuntimeMotionWaypoint(pos=plan.pre_grasp_pos, quat=plan.pre_grasp_quat, finger_width=plan.finger_open,
                              label="approach", path=tuple(approach)),
        RuntimeMotionWaypoint(pos=plan.grasp_pos, quat=plan.grasp_quat, finger_width=plan.finger_open,
                              label="descend", linear="refuse"),
        RuntimeMotionWaypoint(pos=plan.grasp_pos, quat=plan.grasp_quat, finger_width=plan.finger_closed,
                              label="close", segment_type="finger", finger_only=True,
                              hold_after_steps=params.grasp_settle_steps, allow="grip"),
        Attach(grasp),
        RuntimeMotionWaypoint(pos=np.asarray(plan.grasp_pos, dtype=np.float64) + LIFT, quat=plan.grasp_quat,
                              finger_width=plan.finger_closed, label="lift", linear="keep",
                              hold_after_steps=params.lift_hold_steps, allow="rest"),
    ]


def place_actions(place_plan: GraspPlan, params: MotionParams, *, phase: str = "all") -> list[Action]:
    """transport → move_linear to the drop point → detach → gripper open → move_linear back to the standoff.

    The start state holds the object (fingers closed). Finger widths come from place_plan: finger_closed while
    transporting, finger_open on release.

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
    ``"descend"`` from there. The object is held until the descend ends, and the descend may rest it on the target;
    the release and the retreat are checked with it placed, and may touch it with the fingers.
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
        Detach(),
        RuntimeMotionWaypoint(
            pos=place_plan.grasp_pos,
            quat=place_plan.grasp_quat,
            finger_width=place_plan.finger_open,
            label="release",
            segment_type="finger",
            steps=params.release_steps,
            finger_only=True,
            allow="grip",
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
