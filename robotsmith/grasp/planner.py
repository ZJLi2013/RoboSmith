"""Grasp planning contracts.

This module defines the shared data contract between grasp planners and motion
execution: ``GraspPlan`` plus the ``GraspPlanner`` backend ABC. The ``Waypoint``
primitive is owned by the motion layer (``robotsmith.motion.types``) and
re-exported here for the planners that build waypoint sequences.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from typing import Optional

import numpy as np

from robotsmith.grasp.transforms import TOP_DOWN_QUAT, rotmat_to_quat_wxyz
from robotsmith.motion.types import Waypoint

_DEFAULT_APPROACH = np.array([0.0, 0.0, -1.0])


S41_ROLL_90 = False


def _ee_quat_from_approach(approach: np.ndarray) -> np.ndarray:
    """EE quat whose gripper Z axis (approach) aligns with ``approach`` (insert dir).

    Reproduces ``TOP_DOWN_QUAT`` exactly for the default ``-Z`` so drawer / tabletop
    places stay byte-identical; for a side approach builds an orthonormal frame with
    Z = approach and fingers along the most-horizontal perpendicular. Finger yaw is
    a first cut (orientation/IK tuning for genuine tuck-under is deferred — see
    docs/refactor.md R4 "Franka 近界位形").
    """
    z = np.asarray(approach, dtype=np.float64).reshape(3)
    z = z / float(np.linalg.norm(z))
    if np.allclose(z, _DEFAULT_APPROACH, atol=1e-6):
        return TOP_DOWN_QUAT.astype(np.float64).copy()
    up = np.array([0.0, 0.0, 1.0])
    x = up - np.dot(up, z) * z
    if float(np.linalg.norm(x)) < 1e-6:
        ref = np.array([1.0, 0.0, 0.0])
        x = ref - np.dot(ref, z) * z
    x = x / float(np.linalg.norm(x))
    y = np.cross(z, x)
    if S41_ROLL_90:
        x, y = y, -x
    return rotmat_to_quat_wxyz(np.column_stack([x, y, z]))


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

    place_retreat_clearance: float = 0.17

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

    def plan_place(
        self,
        place_pos: np.ndarray,
        *,
        category: str = "block",
        place_point_world: Optional[np.ndarray] = None,
        approach: Optional[np.ndarray] = None,
    ) -> GraspPlan:
        """Build a place-target GraspPlan along an approach axis.

        Place planning needs no grasp candidates, just a target pose. The caller passes
        ``place_point_world`` (the already-resolved world drop point: marker / opening / shelf world
        xyz lifted by the grasp-relative offset); the planner never re-derives a table-relative drop
        height (see docs/features/feature12_pick_place_into_drawer.md §5 W2).

        ``approach`` is the world **insert direction** (unit; default top-down ``-Z``, = grasp frame
        Z). The standoff sits one ``place_retreat_clearance`` back along ``-approach``; the executor's
        Cartesian segment then inserts along ``+approach`` and retracts back along ``-approach``.
        Top-down is just the ``-Z`` value (drawer / tabletop unchanged), side tuck-under is e.g.
        ``+X`` — same code path, no per-case branch (docs/refactor.md R4).
        """
        if place_point_world is None:
            raise ValueError(
                "plan_place requires place_point_world (the resolved world drop "
                "point); table-relative place_z was retired (feature12 §5 W2)"
            )
        place_target = np.asarray(place_point_world, dtype=np.float64).reshape(3)
        a = _DEFAULT_APPROACH if approach is None else np.asarray(approach, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(a))
        a = _DEFAULT_APPROACH if norm < 1e-9 else a / norm
        standoff = place_target - a * self.place_retreat_clearance
        quat = _ee_quat_from_approach(a)
        return GraspPlan(
            grasp_pos=place_target,
            grasp_quat=quat.copy(),
            pre_grasp_pos=standoff,
            pre_grasp_quat=quat.copy(),
            retreat_pos=standoff.copy(),
            retreat_quat=quat.copy(),
            finger_open=0.04,
            finger_closed=0.01,
            quality=1.0,
            metadata={"source": "place", "category": category, "approach_dir": a.tolist()},
        )
