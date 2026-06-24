"""Motion layer — pure IK / collision-free motion planning (grasp-independent).

The execution glue that consumes ``GraspPlan`` lives in ``robotsmith.execution``;
this package depends on nothing in ``grasp``/``execution``.
"""

from robotsmith.motion.params import MotionParams

__all__ = [
    "MotionParams",
]
