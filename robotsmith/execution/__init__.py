"""Execution layer — turns GraspPlans + a MotionPlanner backend into trajectories.

This is the glue above ``grasp`` (where/how to grasp) and ``motion`` (pure IK /
collision-free planning): ``MotionExecutor`` consumes a ``GraspPlan`` plus a
``MotionPlanner`` and produces joint-space trajectories. It depends on both
``grasp`` and ``motion``; neither depends on it (keeps ``motion`` a pure,
grasp-independent package).
"""

from robotsmith.execution.executor import MotionExecutor

__all__ = ["MotionExecutor"]
