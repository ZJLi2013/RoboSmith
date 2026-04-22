"""Task definition system: TaskSpec + composable predicates + presets."""

from robotsmith.tasks.task_spec import TaskSpec
from robotsmith.tasks.predicates import PREDICATE_REGISTRY, evaluate_predicate
from robotsmith.tasks.presets import TASK_PRESETS

__all__ = [
    "TaskSpec",
    "PREDICATE_REGISTRY",
    "evaluate_predicate",
    "TASK_PRESETS",
]
