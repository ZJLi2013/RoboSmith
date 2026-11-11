"""MotionParams — timing parameters for trajectory execution.

Contains NO grasp-decision fields (orientation, finger width, heights).
Those live in GraspPlan.

Two step-allocation modes:
  - **velocity-adaptive** (joint-space segments without explicit steps):
    ``steps = max(ceil(joint_dist / (vel * dt)), min_steps)``
    Only ``max_joint_vel``, ``dt`` matter.
  - **fixed step counts** (used by ``place`` and the drag-handle builders):
    The ``*_steps`` fields below are used directly.

A straight-line (``move_linear``) segment takes as many steps as its path has points.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MotionParams:
    """Trajectory timing parameters.

    Velocity-adaptive fields:
      max_joint_vel  — move speed (rad/s)
      dt             — control timestep (s), i.e. interval between consecutive
                       control_dofs_position() calls. Genesis default: 1/30 s.
      min_steps      — lower bound per segment (avoids degenerate 1-step moves)
      finger_steps   — fixed steps for finger-only segments

    Fixed step-count fields (for place & drag-handle builders):
    """

    # -- velocity-adaptive --
    max_joint_vel: float = 1.0
    dt: float = 1.0 / 30  # Genesis control dt (fps=30), NOT physics substep
    min_steps: int = 5
    finger_steps: int = 15
    grasp_settle_steps: int = 15  # hold after finger-close before lift
    contact_settle_steps: int = 20  # zero-vel dwell before release/retreat in a
    # contact drag (lets the dragged joint's velocity decay so it does not spring
    # back when the gripper lets go).

    # -- fixed step counts --
    approach_steps: int = 40
    grasp_hold_steps: int = 20
    lift_hold_steps: int = 15
    transport_steps: int = 40
    release_steps: int = 15
