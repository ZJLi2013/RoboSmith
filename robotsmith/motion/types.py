"""Shared motion-layer data primitives (zero intra-package deps).

``Waypoint`` is a single end-effector target (pose + finger width). It is the
shared kinematic primitive used both by ``grasp`` planners (hand-authored
pre-defined waypoint sequences) and by the ``motion`` planner backends
(``plan_motion`` inputs). It lives in this leaf — not in ``grasp.planner`` — so
the general motion layer owns the primitive and ``grasp`` depends on ``motion``
(not the reverse).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Waypoint:
    """A single end-effector target in a motion sequence."""

    pos: np.ndarray
    quat: np.ndarray  # wxyz
    finger_width: float
