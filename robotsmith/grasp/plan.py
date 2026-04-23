"""GraspPlan — the contract between Grasp Planning and Motion Execution layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class GraspPlan:
    """Complete grasp specification produced by a GraspPlanner.

    All poses are in world frame. This is the sole interface between
    "where to grasp" (planner) and "how to reach it" (executor).
    """

    grasp_pos: np.ndarray
    grasp_quat: np.ndarray          # EE orientation at grasp (wxyz)
    pre_grasp_pos: np.ndarray       # hover / approach waypoint
    pre_grasp_quat: np.ndarray
    retreat_pos: np.ndarray         # post-grasp lift waypoint
    retreat_quat: np.ndarray
    finger_open: float              # finger width while approaching
    finger_closed: float            # finger width when grasping
    quality: float = 1.0            # planner confidence (0–1)
    metadata: dict = field(default_factory=dict)

    # Mid-wall grasp: EE descends to this XY-offset position first,
    # then approaches horizontally to grasp_pos.  None for top-down.
    side_approach_pos: Optional[np.ndarray] = None
    side_approach_quat: Optional[np.ndarray] = None
