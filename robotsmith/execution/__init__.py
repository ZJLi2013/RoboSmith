"""Execution layer — turns action lists + a MotionPlanner backend into trajectories.

This is the glue above ``grasp`` (where/how to grasp) and ``motion`` (pure IK /
collision-free planning): ``meta_actions`` builds the pick / place lists from a ``GraspPlan``, and
``MotionExecutor.run`` expands one into a joint-space trajectory. It depends on both
``grasp`` and ``motion``; neither depends on it (keeps ``motion`` a pure,
grasp-independent package).
"""

from robotsmith.execution.executor import MotionExecutor

__all__ = ["MotionExecutor"]
