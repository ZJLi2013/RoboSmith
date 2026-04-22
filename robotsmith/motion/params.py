"""MotionParams — pure timing / step-count parameters for trajectory execution.

Contains NO grasp-decision fields (orientation, finger width, heights).
Those live in GraspPlan.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MotionParams:
    """Step counts for each motion phase."""

    approach_steps: int = 40
    descend_steps: int = 30
    grasp_hold_steps: int = 20
    lift_steps: int = 30
    lift_hold_steps: int = 15
    transport_steps: int = 40
    place_descend_steps: int = 25
    release_steps: int = 15
    retreat_steps: int = 25
