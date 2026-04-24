"""LearnedGraspPlanner — GraspGen-backed grasp planning.

Replaces per-category templates with a learned model that generalises to
arbitrary object geometries. The pipeline:

    asset mesh → trimesh.sample → point cloud (local frame)
        → GraspGen inference → 100 candidates {SE3, score}
            → coordinate transform (local → world)
            → height filter / workspace filter / gripper width check
            → score ranking → top-K → list[GraspPlan]

Downstream MotionExecutor / run_skills see the same GraspPlan interface.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from robotsmith.grasp.plan import GraspPlan
from robotsmith.grasp.planner import GraspPlanner
from robotsmith.grasp.pointcloud_utils import asset_to_pointcloud
from robotsmith.grasp.transforms import pose_matrix, rotmat_to_quat_wxyz

logger = logging.getLogger(__name__)

_TOP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

_FRANKA_FINGER_OPEN = 0.04
_FRANKA_FINGER_CLOSED = 0.01


class LearnedGraspPlanner(GraspPlanner):
    """GraspGen-backed grasp planner.

    Args:
        graspgen_model: A GraspGenModel instance (from graspgen_wrapper.py).
        z_offset: Table surface Z (same as TemplateGraspPlanner.z_offset).
        n_sample_points: Number of points to sample from the asset mesh.
        min_grasp_z_margin: Minimum Z above table for a valid grasp.
        hover_clearance: Pre-grasp hover height above grasp point.
        retreat_clearance: Post-grasp retreat height above grasp point.
        top_k: Maximum number of GraspPlans to return.
    """

    def __init__(
        self,
        graspgen_model: Any,
        *,
        z_offset: float = 0.0,
        n_sample_points: int = 8192,
        min_grasp_z_margin: float = 0.02,
        hover_clearance: float = 0.12,
        retreat_clearance: float = 0.17,
        top_k: int = 1,
    ):
        self._model = graspgen_model
        self._z_offset = z_offset
        self._n_points = n_sample_points
        self._min_z_margin = min_grasp_z_margin
        self._hover_clearance = hover_clearance
        self._retreat_clearance = retreat_clearance
        self._top_k = top_k

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
        """Generate grasp plans using GraspGen.

        The asset's visual mesh is sampled into a point cloud in the mesh's
        local frame, passed to GraspGen, then the resulting grasp poses are
        transformed to world frame and filtered.
        """
        if asset is None:
            logger.warning(
                "LearnedGraspPlanner requires an asset with mesh; "
                "falling back to empty plan list"
            )
            return []

        # --- 1. Generate point cloud in mesh local frame ---
        pc_local = asset_to_pointcloud(
            asset, self._n_points, scale=scale,
        )

        # --- 2. GraspGen inference (local frame) ---
        grasp_poses, grasp_scores = self._model.predict(pc_local)

        if len(grasp_poses) == 0:
            logger.warning("GraspGen returned 0 grasps for %s", asset.name)
            return []

        # --- 3. Transform grasps: local frame → world frame ---
        T_world_obj = pose_matrix(
            np.asarray(object_pos, dtype=np.float64),
            object_quat,
        )
        world_poses = np.array([T_world_obj @ g for g in grasp_poses])

        # --- 4. Filter & rank ---
        table_z = self._z_offset
        min_z = table_z + self._min_z_margin

        plans: list[GraspPlan] = []
        for i in range(len(world_poses)):
            pose = world_poses[i]
            score = float(grasp_scores[i])

            grasp_pos = pose[:3, 3].astype(np.float64)
            grasp_rot = pose[:3, :3]

            # Height filter
            if grasp_pos[2] < min_z:
                continue

            grasp_quat = rotmat_to_quat_wxyz(grasp_rot)

            # Pre-grasp: directly above the grasp point
            pre_grasp_pos = grasp_pos.copy()
            pre_grasp_pos[2] = grasp_pos[2] + self._hover_clearance

            # Retreat: lift straight up
            retreat_pos = grasp_pos.copy()
            retreat_pos[2] = grasp_pos[2] + self._retreat_clearance

            plans.append(GraspPlan(
                grasp_pos=grasp_pos,
                grasp_quat=grasp_quat,
                pre_grasp_pos=pre_grasp_pos,
                pre_grasp_quat=grasp_quat.copy(),
                retreat_pos=retreat_pos,
                retreat_quat=grasp_quat.copy(),
                finger_open=_FRANKA_FINGER_OPEN,
                finger_closed=_FRANKA_FINGER_CLOSED,
                quality=score,
                metadata={
                    "source": "graspgen",
                    "category": category,
                    "candidate_index": i,
                },
            ))

            if len(plans) >= self._top_k:
                break

        if not plans:
            logger.warning(
                "All %d GraspGen candidates filtered out for %s",
                len(grasp_poses), asset.name,
            )

        return plans

    def plan_place(
        self,
        place_pos: np.ndarray,
        *,
        category: str = "block",
        place_z_override: Optional[float] = None,
    ) -> GraspPlan:
        """Build a place-target GraspPlan (reuses template-style placement).

        Place planning doesn't need learned grasps — just a target pose.
        """
        px, py = float(place_pos[0]), float(place_pos[1])
        zo = self._z_offset
        pz = place_z_override if place_z_override is not None else 0.15

        pre_place_z = pz + zo + self._retreat_clearance
        place_target = np.array([px, py, pz + zo])
        pre_place_pos = np.array([px, py, pre_place_z])
        retreat_pos = np.array([px, py, pre_place_z])

        return GraspPlan(
            grasp_pos=place_target,
            grasp_quat=_TOP_DOWN_QUAT.copy(),
            pre_grasp_pos=pre_place_pos,
            pre_grasp_quat=_TOP_DOWN_QUAT.copy(),
            retreat_pos=retreat_pos,
            retreat_quat=_TOP_DOWN_QUAT.copy(),
            finger_open=_FRANKA_FINGER_OPEN,
            finger_closed=_FRANKA_FINGER_CLOSED,
            quality=1.0,
            metadata={"source": "learned_place", "category": category},
        )
