"""GraspPlanner ABC — base class for all grasp planning backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from robotsmith.grasp.plan import GraspPlan


class GraspPlanner(ABC):
    """Given an object pose (+ optional asset metadata), produce GraspPlan(s)."""

    @abstractmethod
    def plan(
        self,
        object_pos: np.ndarray,
        object_quat: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        *,
        category: str = "block",
        asset: Any = None,
        object_height: float | None = None,
    ) -> list[GraspPlan]:
        """Return GraspPlan(s) sorted by quality (descending)."""
        ...
