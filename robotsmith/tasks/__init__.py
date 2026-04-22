"""Task definition system: TaskSpec + composable predicates + presets."""

from robotsmith.tasks.task_spec import TaskSpec
from robotsmith.tasks.predicates import PREDICATE_REGISTRY, evaluate_predicate
from robotsmith.tasks.presets import TASK_PRESETS

# Backward-compat re-exports (deprecated — use robotsmith.grasp / robotsmith.motion)
from robotsmith.tasks.ik_strategies import IK_STRATEGIES, TrajectoryParams  # noqa: F401

__all__ = [
    "TaskSpec",
    "PREDICATE_REGISTRY",
    "evaluate_predicate",
    "TASK_PRESETS",
    # deprecated, kept for backward compat
    "IK_STRATEGIES",
    "TrajectoryParams",
]
