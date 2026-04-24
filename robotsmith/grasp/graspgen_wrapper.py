"""GraspGen inference wrapper for RoboSmith.

Thin wrapper around GraspGen's GraspGenSampler that handles:
  - Model loading from a gripper config YAML
  - Point cloud → grasp poses + scores inference
  - GPU ↔ numpy conversion

Requires: ``pip install -e /path/to/GraspGen --no-deps``
plus ROCm-patched pointnet2_ops and torch-cluster.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class GraspGenModel:
    """Lazy-loaded GraspGen inference model.

    Keeps the heavy imports (torch, grasp_gen) behind __init__ so that
    code that only uses TemplateGraspPlanner doesn't pay the import cost.
    """

    def __init__(
        self,
        gripper_config: str | Path,
        *,
        grasp_threshold: float = -1.0,
        num_grasps: int = 200,
        topk_num_grasps: int = 100,
        remove_outliers: bool = True,
    ):
        self._gripper_config = str(gripper_config)
        self._grasp_threshold = grasp_threshold
        self._num_grasps = num_grasps
        self._topk_num_grasps = topk_num_grasps
        self._remove_outliers = remove_outliers
        self._sampler: Optional[object] = None

    def _ensure_loaded(self) -> None:
        if self._sampler is not None:
            return
        from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

        logger.info("Loading GraspGen model from %s", self._gripper_config)
        cfg = load_grasp_cfg(self._gripper_config)
        self._sampler = GraspGenSampler(cfg)
        self._run_inference = GraspGenSampler.run_inference
        logger.info("GraspGen model loaded (gripper: %s)", cfg.data.gripper_name)

    def predict(
        self,
        point_cloud: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run GraspGen inference on a point cloud.

        Args:
            point_cloud: (N, 3) object point cloud in whatever frame
                         you want the output grasps expressed in.

        Returns:
            grasp_poses: (M, 4, 4) SE3 homogeneous transforms.
            grasp_scores: (M,) confidence scores (higher = better).
        """
        self._ensure_loaded()

        grasps, scores = self._run_inference(
            point_cloud,
            self._sampler,
            grasp_threshold=self._grasp_threshold,
            num_grasps=self._num_grasps,
            topk_num_grasps=self._topk_num_grasps,
            remove_outliers=self._remove_outliers,
        )

        if len(grasps) == 0:
            return (
                np.empty((0, 4, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        poses = grasps.cpu().numpy().astype(np.float32)
        confs = scores.cpu().numpy().astype(np.float32)

        poses[:, 3, :] = [0, 0, 0, 1]

        sort_idx = np.argsort(-confs)
        return poses[sort_idx], confs[sort_idx]
