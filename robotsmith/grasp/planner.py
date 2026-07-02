"""Grasp planning contracts.

This module defines the shared data contract between grasp planners and motion
execution: ``GraspPlan`` plus the ``GraspPlanner`` backend ABC. The ``Waypoint``
primitive is owned by the motion layer (``robotsmith.motion.types``) and
re-exported here for the planners that build waypoint sequences.

It also owns the per-asset dispatch ``resolve_grasp_strategy`` — the decision of
whether an asset is grasp-planned (``"learned"``) or skipped (``"none"``) — since
that is a planning-contract concern, not a property of any one planner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from typing import Optional

import numpy as np

from robotsmith.motion.types import Waypoint


@dataclass
class GraspPlan:
    """Complete grasp specification produced by a GraspPlanner.

    All poses are in world frame. This is the sole interface between
    "where to grasp" (planner) and "how to reach it" (executor).

    Pick plans set ``waypoints``: the executor walks the ordered waypoint list.
    The fixed ``pre_grasp``/``grasp``/``retreat`` fields are used by ``place``
    (which builds its own transport/descend/retreat sequence from them).
    """

    grasp_pos: np.ndarray
    grasp_quat: np.ndarray          # EE orientation at grasp (wxyz)
    pre_grasp_pos: np.ndarray       # hover / approach waypoint
    pre_grasp_quat: np.ndarray
    retreat_pos: np.ndarray         # post-grasp lift waypoint
    retreat_quat: np.ndarray
    finger_open: float              # finger width while approaching
    finger_closed: float            # finger width when grasping
    quality: float = 1.0            # planner confidence (0-1)
    metadata: dict = field(default_factory=dict)

    # Learned planner: ordered waypoint sequence for approach-aware motion.
    # When set, MotionExecutor uses this instead of the fixed fields above.
    waypoints: Optional[list[Waypoint]] = None


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
        scale: float = 1.0,
    ) -> list[GraspPlan]:
        """Return GraspPlan(s) sorted by quality (descending)."""
        ...


# -- per-asset grasp strategy dispatch --------------------------------------
#
# Decouples *whether/how a grasp pose is acquired* from the experiment-level
# default, so scenes can mix object kinds without an all-or-nothing switch:
#
# - Articulated assets (a drawer, etc.) are manipulated by joint primitives
#   (``drag_handle``) and are **never** grasp-planned -> ``"none"``.
# - General pickable assets are grasp-planned by the learned planner (GraspGen)
#   -> ``"learned"``.
#
# ``--grasp-planner`` is the run-level *default* for general assets; an individual
# asset may pin its own strategy via ``metadata.grasp_strategy`` (reserved/unused
# until a scene needs it), e.g. ``"none"`` to skip grasp-planning one object.

GRASP_STRATEGIES = {"learned", "none"}


def resolve_grasp_strategy(asset, *, default: str = "learned") -> str:
    """Resolve the grasp strategy for one asset.

    Fixtures (static props and articulated furniture) always resolve to
    ``"none"`` — they are never grasp-planned. Otherwise an explicit
    ``metadata.grasp_strategy`` override wins, falling back to ``default``.
    """
    if getattr(asset, "is_fixture", False):
        return "none"
    override = getattr(getattr(asset, "metadata", None), "grasp_strategy", None)
    if override:
        if override not in GRASP_STRATEGIES:
            raise ValueError(
                f"unknown grasp_strategy {override!r}; expected one of {GRASP_STRATEGIES}"
            )
        return override
    return default
