"""Minimal zero-dependency robot-structure constant shared across layers.

Only ``N_ARM_JOINTS`` lives here: it is the qpos slice index used pervasively by
``motion`` and ``grasp`` (``q[:N_ARM_JOINTS]`` selects the 7 arm DoFs of the 9-dim
Franka qpos). It must import from a leaf with no third-party or intra-package deps
to avoid a ``motion`` ↔ ``grasp`` import cycle. The fuller Franka hardware spec
(joint names, home pose, gains, force + joint limits) lives in
``robotsmith.sim.franka``.
"""

from __future__ import annotations

N_ARM_JOINTS = 7
