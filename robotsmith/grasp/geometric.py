"""Grasp candidates from object geometry alone.

The planner reads the asset's visual geometry, takes its yaw-oriented bounding box in the world
frame, and emits top-down and side grasps for every horizontal box axis the gripper can close
across. Candidates are not scored or screened beyond that width check: which one works is decided
by trying them in simulation (``grasp_trial`` in the pick skill).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh

from robotsmith.assets import derive
from robotsmith.grasp.planner import GraspPlan, GraspPlanner
from robotsmith.grasp.transforms import quat_wxyz_to_matrix, rotmat_to_quat_wxyz
from robotsmith.motion.types import Waypoint

DOWN = np.array([0.0, 0.0, -1.0])
UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class GripperSpec:
    """Franka hand: each finger opens ``finger_open`` from the centre (``SimEnv.solve_ik`` sets both
    finger joints to it); IK targets the ``hand`` link, ``tcp_offset`` behind the fingertip centre.
    ``usable_width`` is the widest object the closed hand still lifts, measured in simulation."""

    finger_open: float = 0.04
    finger_closed: float = 0.01
    finger_length: float = 0.04
    tcp_offset: float = 0.1034
    usable_width: float = 0.07
    approach_clearance: float = 0.12
    lift: float = 0.17


def yaw_box(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Centre xy, two horizontal unit axes, their extents, and the z range of ``points``."""
    to_box, extents = trimesh.bounds.oriented_bounds_2D(points[:, :2])
    from_box = np.linalg.inv(to_box)
    axes = from_box[:2, :2].T
    return from_box[:2, 2], axes, np.asarray(extents), float(points[:, 2].min()), float(points[:, 2].max())


def _canonical(axis: np.ndarray) -> np.ndarray:
    """The two-finger grasp is symmetric about its closing axis; pick one sign deterministically."""
    flip = axis[0] < -1e-9 or (abs(axis[0]) <= 1e-9 and axis[1] < 0)
    return -axis if flip else axis


class GeometricGraspPlanner(GraspPlanner):
    def __init__(self, gripper: GripperSpec = GripperSpec()):
        self.gripper = gripper
        self.last_filter_counts: dict[str, int] = {}

    def plan(self, object_pos, object_quat=None, rng=None, *, category="block", asset: Any = None,
             object_height=None, scale=1.0) -> list[GraspPlan]:
        rot = np.eye(3) if object_quat is None else quat_wxyz_to_matrix(np.asarray(object_quat, dtype=np.float64))
        world = derive.visual_points(asset.urdf_path) * scale @ rot.T + np.asarray(object_pos, dtype=np.float64)
        plans = self.candidates(world, category)
        self.last_filter_counts = {"raw": len(plans), "accepted": len(plans)}
        return plans

    def candidates(self, world_points: np.ndarray, category: str = "") -> list[GraspPlan]:
        """Every candidate in the base wrist orientation, then the same list turned 180° about the
        approach axis: the hand is symmetric, but the two give IK different joint branches."""
        g = self.gripper
        centre, axes, extents, z_lo, z_hi = yaw_box(world_points)
        half = (z_hi - z_lo) / 2.0
        closable = sorted((w, i) for i, w in enumerate(extents) if w <= g.usable_width)
        depths = sorted({round(min(0.015, half), 4), round(min(half, g.finger_length), 4)})
        out = []
        for orientation, turn in (("base", 1.0), ("flipped", -1.0)):
            for width, i in closable:
                finger = turn * _canonical(np.r_[axes[i], 0.0])
                for d in depths:
                    out.append(self._plan("top_down", orientation, np.r_[centre, z_hi - d], DOWN, finger, width, category))
            for width, i in closable:
                finger = turn * _canonical(np.r_[axes[i], 0.0])
                for sign in (1.0, -1.0):
                    out.append(self._plan("side", orientation, np.r_[centre, z_lo + half], sign * np.r_[axes[1 - i], 0.0],
                                          finger, width, category))
        for k, p in enumerate(out):
            p.metadata["candidate_index"] = k
        return out

    def _plan(self, kind, orientation, tip, approach, finger, width, category) -> GraspPlan:
        g = self.gripper
        rot = np.stack([finger, np.cross(approach, finger), approach], axis=1)
        quat = rotmat_to_quat_wxyz(rot)
        hand = tip - approach * g.tcp_offset
        pre = hand - approach * g.approach_clearance
        lifted = hand + UP * g.lift
        return GraspPlan(
            grasp_pos=hand, grasp_quat=quat, pre_grasp_pos=pre, pre_grasp_quat=quat.copy(),
            retreat_pos=lifted, retreat_quat=quat.copy(), finger_open=g.finger_open, finger_closed=g.finger_closed,
            metadata={"source": "geometric", "kind": kind, "orientation": orientation, "category": category, "closing_width": float(width),
                      "tip_pos": tip.tolist(), "approach_dir": approach.tolist(), "finger_dir": finger.tolist()},
            waypoints=[
                Waypoint(pos=pre, quat=quat.copy(), finger_width=g.finger_open),
                Waypoint(pos=hand.copy(), quat=quat.copy(), finger_width=g.finger_open),
                Waypoint(pos=hand.copy(), quat=quat.copy(), finger_width=g.finger_closed),
                Waypoint(pos=lifted, quat=quat.copy(), finger_width=g.finger_closed),
            ],
        )
