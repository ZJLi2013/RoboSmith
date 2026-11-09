"""The pick and place parts of a plan's ``actions`` (``scenarios/acceptance/plans``), translated item by item into
the action lists ``MotionExecutor.run`` runs.

A part starts at its ``sample`` and runs to the next one. An action is a motion segment (``RuntimeMotionWaypoint``:
``finger_only`` is a ``gripper``, ``linear`` a ``move_linear``, otherwise a ``move_to``) or an ``Attach`` /
``Detach`` of the plan state, which moves nothing.
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

# The one order each part runs in; a part in any other order is refused at load.
PICK_ORDER = ("sample", "gripper open", "move_to pre", "move_linear grasp", "gripper close", "attach",
              "move_linear line")
PLACE_ORDER = ("sample", "move_to pre", "move_linear place", "detach", "gripper open", "move_linear retreat")


def _key(item: dict) -> str:
    (name, arg), = item.items()
    if name == "gripper":
        return f"gripper {arg['width']}"
    if name in ("move_to", "move_linear"):
        return f"{name} {arg.partition('.')[2]}" if isinstance(arg, str) else f"{name} line"
    return name


def check_part(part: list[dict]) -> str:
    """``"pick"`` or ``"place"``; ``ValueError`` unless ``part`` runs in that part's order."""
    kind = {"grasp": "pick", "place": "place"}[part[0]["sample"]["kind"]]
    order = PICK_ORDER if kind == "pick" else PLACE_ORDER
    keys = tuple(_key(item) for item in part)
    if keys != order:
        raise ValueError(f"a {kind} part runs {' → '.join(keys)}; the first version runs {' → '.join(order)}")
    return kind


def pick_actions(
    part: list[dict],
    plan: GraspPlan,
    start_q: np.ndarray,
    start_hand: tuple[np.ndarray, np.ndarray],
    approach: Sequence[np.ndarray],
    params: MotionParams,
    solve_ik: Callable,
    *,
    grasp: Holding,
) -> list[Action]:
    """The pick ``part`` with ``plan``'s poses: gripper open → move_to pre-grasp → move_linear to the grasp →
    gripper close → attach → move_linear along the part's direction and distance.

    The fingers open where the arm is at ``start_q`` (``start_hand`` is its hand pose there), in zero steps when they
    already are. ``approach`` is the planned joint path from there to the pre-grasp configuration, empty when
    ``start_q`` already is it. The close may touch the target (``allow="grip"``) and settles; ``grasp`` is held from
    then on, and the lift may still touch the support it leaves (``allow="rest"``) and holds for ``lift_hold_steps``.
    """
    poses = {"pre": (plan.pre_grasp_pos, plan.pre_grasp_quat), "grasp": (plan.grasp_pos, plan.grasp_quat)}
    pos, quat = start_hand
    width = plan.finger_open
    actions: list[Action] = []
    for item in part[1:]:
        key, (arg,) = _key(item), item.values()
        if key == "gripper open":
            opened = gripper.expand(gripper.Gripper(pos, quat, plan.finger_open), start_q, solve_ik).q_target
            already_open = np.allclose(opened[7:], np.asarray(start_q, dtype=np.float64)[7:])
            width = plan.finger_open
            actions.append(RuntimeMotionWaypoint(pos=pos, quat=quat, finger_width=width, label="open",
                                                 segment_type="finger", steps=0 if already_open else None,
                                                 finger_only=True))
        elif key == "move_to pre":
            pos, quat = poses["pre"]
            actions.append(RuntimeMotionWaypoint(pos=pos, quat=quat, finger_width=width, label="approach",
                                                 path=tuple(approach)))
        elif key == "move_linear grasp":
            pos, quat = poses["grasp"]
            actions.append(RuntimeMotionWaypoint(pos=pos, quat=quat, finger_width=width, label="descend",
                                                 linear="refuse"))
        elif key == "gripper close":
            width = plan.finger_closed
            actions.append(RuntimeMotionWaypoint(pos=pos, quat=quat, finger_width=width, label="close",
                                                 segment_type="finger", finger_only=True,
                                                 hold_after_steps=params.grasp_settle_steps, allow="grip"))
        elif key == "attach":
            actions.append(Attach(grasp))
        elif key == "move_linear line":
            pos = np.asarray(pos, dtype=np.float64) + _unit(arg["direction"]) * arg["dist"]
            actions.append(RuntimeMotionWaypoint(pos=pos, quat=quat, finger_width=width, label="lift",
                                                 linear="keep", hold_after_steps=params.lift_hold_steps,
                                                 allow="rest"))
    return actions


def place_actions(part: list[dict], place_plan: GraspPlan, params: MotionParams, *,
                  phase: str = "all") -> list[Action]:
    """The place ``part`` with ``place_plan``'s poses: transport → move_linear to the drop point → detach →
    gripper open → move_linear back to the standoff.

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
    poses = {"pre": (place_plan.pre_grasp_pos, place_plan.pre_grasp_quat),
             "place": (place_plan.grasp_pos, place_plan.grasp_quat),
             "retreat": (place_plan.retreat_pos, place_plan.retreat_quat)}
    width = place_plan.finger_closed
    actions: list[Action] = []
    for item in part[1:]:
        key = _key(item)
        if key == "move_to pre":
            pos, quat = poses["pre"]
            actions.append(RuntimeMotionWaypoint(pos=pos, quat=quat, finger_width=width, label="transport",
                                                 steps=params.transport_steps))
            if phase == "transport":
                return actions
        elif key == "move_linear place":
            pos, quat = poses["place"]
            actions.append(RuntimeMotionWaypoint(pos=pos, quat=quat, finger_width=width, label="descend",
                                                 linear="keep", steps=params.place_descend_steps, allow="rest"))
        elif key == "detach":
            actions.append(Detach())
        elif key == "gripper open":
            width = place_plan.finger_open
            actions.append(RuntimeMotionWaypoint(pos=pos, quat=quat, finger_width=width, label="release",
                                                 segment_type="finger", steps=params.release_steps,
                                                 finger_only=True, allow="grip"))
        elif key == "move_linear retreat":
            pos, quat = poses["retreat"]
            actions.append(RuntimeMotionWaypoint(pos=pos, quat=quat, finger_width=width, label="retreat",
                                                 linear="keep", steps=params.retreat_steps, allow="grip"))
    return actions


def _unit(direction) -> np.ndarray:
    d = np.asarray(direction, dtype=np.float64)
    return d / np.linalg.norm(d)
