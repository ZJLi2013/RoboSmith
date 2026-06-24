"""LearnedGraspPlanner — GraspGen-backed grasp planning with approach-aware waypoints.

Replaces per-category templates with a learned model that generalises to
arbitrary object geometries.  Uses GraspGen's predicted approach direction
to build a waypoint sequence so MotionExecutor follows the correct path
(instead of always descending vertically).

Waypoint sequence for each grasp:
    1. pre-approach — offset along -approach by ``approach_clearance``
    2. grasp (open) — at grasp point, fingers open
    3. grasp (closed) — same position, fingers closed
    4. retreat — back out along -approach by ``retreat_clearance``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from robotsmith.grasp.planner import GraspPlan, Waypoint
from robotsmith.grasp.planner import GraspPlanner
from robotsmith.grasp.transforms import (
    TOP_DOWN_QUAT,
    pose_matrix,
    quat_wxyz_to_matrix,
    rotmat_to_quat_wxyz,
)

logger = logging.getLogger(__name__)

_FRANKA_FINGER_OPEN = 0.04
_FRANKA_FINGER_CLOSED = 0.01


class LearnedGraspPlanner(GraspPlanner):
    """GraspGen-backed grasp planner.

    Generates ``GraspPlan``s with an explicit ``waypoints`` list so
    ``MotionExecutor`` follows the predicted approach direction.
    """

    def __init__(
        self,
        graspgen_model: Any,
        *,
        z_offset: float = 0.0,
        n_sample_points: int = 2000,
        min_grasp_z_margin: float = 0.02,
        approach_clearance: float = 0.12,
        retreat_clearance: float = 0.17,
        top_k: int = 100,
        max_approach_z: float = 1.0,
        fixed_orientation: bool = False,
    ):
        self._model = graspgen_model
        self._z_offset = z_offset
        self._n_points = n_sample_points
        self._min_z_margin = min_grasp_z_margin
        self._approach_clearance = approach_clearance
        self._retreat_clearance = retreat_clearance
        self._top_k = top_k
        self._max_approach_z = max_approach_z
        self._fixed_orientation = fixed_orientation

    def _predict_grasps(
        self,
        mesh_path: Path,
        *,
        asset: Any,
        object_quat: np.ndarray | None,
        scale: float,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        poses, scores = self._model.predict_from_mesh(
            mesh_path, scale=scale, num_sample_points=self._n_points,
        )
        return poses, scores, {"input_mode": "mesh_full_surface"}

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
        if asset is None:
            logger.warning("LearnedGraspPlanner requires an asset with mesh")
            return []

        mesh_path = asset.visual_mesh or asset.collision_mesh
        if mesh_path is None or not Path(mesh_path).exists():
            logger.warning("Asset %s has no mesh file", getattr(asset, "name", "?"))
            return []

        grasp_poses, grasp_scores, input_stats = self._predict_grasps(
            Path(mesh_path), asset=asset, object_quat=object_quat, scale=scale,
        )

        n_raw = len(grasp_poses)
        if n_raw == 0:
            logger.warning("GraspGen returned 0 grasps for %s", asset.name)
            return []

        # local → world
        T_world_obj = pose_matrix(
            np.asarray(object_pos, dtype=np.float64), object_quat,
        )
        world_poses = np.array([T_world_obj @ g for g in grasp_poses])

        logger.debug(
            f"[learned] {n_raw} raw grasps, scores=[{grasp_scores.min():.3f}, "
            f"{grasp_scores.max():.3f}]"
        )
        logger.debug("[learned] input: %s", input_stats)
        n_diag = min(10, n_raw)
        for j in range(n_diag):
            p = world_poses[j]
            pos = p[:3, 3]
            finger_dir = p[:3, 0]  # X axis: finger opening direction
            approach = p[:3, 2]    # Z axis: approach direction
            logger.debug(
                f"  [{j:2d}] score={grasp_scores[j]:.3f} "
                f"pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
                f"finger=({finger_dir[0]:.2f},{finger_dir[1]:.2f},{finger_dir[2]:.2f}) "
                f"approach=({approach[0]:.2f},{approach[1]:.2f},{approach[2]:.2f})"
            )

        table_z = self._z_offset
        min_z = table_z + self._min_z_margin
        n_z_reject = 0
        n_approach_reject = 0
        plans: list[GraspPlan] = []

        fixed_approach = np.array([0.0, 0.0, -1.0])
        fixed_finger = np.array([1.0, 0.0, 0.0])
        fixed_rot = np.eye(3)
        fixed_rot[:, 0] = fixed_finger
        fixed_rot[:, 1] = np.cross(fixed_approach, fixed_finger)
        fixed_rot[:, 2] = fixed_approach
        fixed_quat = rotmat_to_quat_wxyz(fixed_rot)

        mode = "fixed_ori" if self._fixed_orientation else "graspgen_ori"
        logger.debug("[learned] orientation mode: %s", mode)

        for i in range(len(world_poses)):
            pose = world_poses[i]
            grasp_pos = pose[:3, 3].astype(np.float64)

            if grasp_pos[2] < min_z:
                n_z_reject += 1
                continue

            if self._fixed_orientation:
                approach_dir = fixed_approach
                finger_dir = fixed_finger
                grasp_quat = fixed_quat.copy()
            else:
                grasp_rot = pose[:3, :3]
                approach_dir = grasp_rot[:, 2]
                if approach_dir[2] > self._max_approach_z:
                    n_approach_reject += 1
                    continue
                finger_dir = grasp_rot[:, 0]
                grasp_quat = rotmat_to_quat_wxyz(grasp_rot)

            pre_pos = grasp_pos - approach_dir * self._approach_clearance
            retreat_pos = grasp_pos - approach_dir * self._retreat_clearance

            wps = [
                Waypoint(pos=pre_pos, quat=grasp_quat.copy(), finger_width=_FRANKA_FINGER_OPEN),
                Waypoint(pos=grasp_pos.copy(), quat=grasp_quat.copy(), finger_width=_FRANKA_FINGER_OPEN),
                Waypoint(pos=grasp_pos.copy(), quat=grasp_quat.copy(), finger_width=_FRANKA_FINGER_CLOSED),
                Waypoint(pos=retreat_pos, quat=grasp_quat.copy(), finger_width=_FRANKA_FINGER_CLOSED),
            ]

            plans.append(GraspPlan(
                grasp_pos=grasp_pos,
                grasp_quat=grasp_quat,
                pre_grasp_pos=pre_pos,
                pre_grasp_quat=grasp_quat.copy(),
                retreat_pos=retreat_pos,
                retreat_quat=grasp_quat.copy(),
                finger_open=_FRANKA_FINGER_OPEN,
                finger_closed=_FRANKA_FINGER_CLOSED,
                quality=float(grasp_scores[i]),
                metadata={
                    "source": "graspgen",
                    "category": category,
                    "candidate_index": i,
                    "approach_dir": approach_dir.tolist(),
                    "finger_dir": finger_dir.tolist(),
                    "orientation_mode": mode,
                    **input_stats,
                },
                waypoints=wps,
            ))

            if len(plans) >= self._top_k:
                break

        logger.debug(
            f"[learned] filter: {n_raw} total → z_reject={n_z_reject}, "
            f"approach_reject={n_approach_reject}, accepted={len(plans)} "
            f"(mode={mode})"
        )
        if plans:
            p = plans[0]
            a = p.metadata["approach_dir"]
            f = p.metadata["finger_dir"]
            logger.debug(
                f"[learned] SELECTED #{p.metadata['candidate_index']} "
                f"score={p.quality:.3f} pos=({p.grasp_pos[0]:.4f},"
                f"{p.grasp_pos[1]:.4f},{p.grasp_pos[2]:.4f}) "
                f"approach=({a[0]:.3f},{a[1]:.3f},{a[2]:.3f}) "
                f"finger=({f[0]:.3f},{f[1]:.3f},{f[2]:.3f})"
            )
        else:
            logger.warning(
                f"[learned] WARNING: no grasps passed filters "
                f"(z_reject={n_z_reject}, approach_reject={n_approach_reject})"
            )
        return plans

    def plan_place(
        self,
        place_pos: np.ndarray,
        *,
        category: str = "block",
        place_z_override: Optional[float] = None,
        place_point_world: Optional[np.ndarray] = None,
    ) -> GraspPlan:
        """Build a place-target GraspPlan (top-down placement pose).

        Place planning doesn't need learned grasps — just a target pose.
        ``place_point_world`` (absolute world drop point) bypasses the
        table-relative ``place_z`` math for live-resolved targets (e.g. an
        articulated drawer opening).
        """
        if place_point_world is not None:
            px, py, pz_world = (float(v) for v in place_point_world)
            hover = self._retreat_clearance
            place_target = np.array([px, py, pz_world])
            pre_place_pos = np.array([px, py, pz_world + hover])
            retreat_pos = np.array([px, py, pz_world + hover])
            return GraspPlan(
                grasp_pos=place_target,
                grasp_quat=TOP_DOWN_QUAT.copy(),
                pre_grasp_pos=pre_place_pos,
                pre_grasp_quat=TOP_DOWN_QUAT.copy(),
                retreat_pos=retreat_pos,
                retreat_quat=TOP_DOWN_QUAT.copy(),
                finger_open=_FRANKA_FINGER_OPEN,
                finger_closed=_FRANKA_FINGER_CLOSED,
                quality=1.0,
                metadata={"source": "learned_place", "category": category},
            )

        px, py = float(place_pos[0]), float(place_pos[1])
        zo = self._z_offset
        pz = place_z_override if place_z_override is not None else 0.15

        pre_place_z = pz + zo + self._retreat_clearance
        place_target = np.array([px, py, pz + zo])
        pre_place_pos = np.array([px, py, pre_place_z])
        retreat_pos = np.array([px, py, pre_place_z])

        return GraspPlan(
            grasp_pos=place_target,
            grasp_quat=TOP_DOWN_QUAT.copy(),
            pre_grasp_pos=pre_place_pos,
            pre_grasp_quat=TOP_DOWN_QUAT.copy(),
            retreat_pos=retreat_pos,
            retreat_quat=TOP_DOWN_QUAT.copy(),
            finger_open=_FRANKA_FINGER_OPEN,
            finger_closed=_FRANKA_FINGER_CLOSED,
            quality=1.0,
            metadata={"source": "learned_place", "category": category},
        )
