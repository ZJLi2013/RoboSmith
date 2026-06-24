"""GraspGen inference wrapper for RoboSmith.

Thin wrapper around GraspGen's GraspGenSampler that handles:
  - Model loading from a full config YAML (with checkpoint paths)
  - HIP cdist monkey-patch for ROCm compatibility
  - Point cloud → grasp poses + scores inference
  - GPU ↔ numpy conversion

NOTE: The Franka Panda checkpoint uses the PTV3 backbone. On ROCm this
requires spconv_rocm; the old non-Franka retarget path is intentionally
not used for Franka evaluation.

Requires: GraspGen source on PYTHONPATH and spconv_rocm installed.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh.transformations as tra

from robotsmith.assets.geometry import sample_mesh_pointcloud

logger = logging.getLogger(__name__)


def _patch_pointnet2_import_for_rocm() -> None:
    """Avoid GraspGen's CUDA pointnet2 JIT on ROCm-only PTV3 checkpoints.

    The Franka Panda config uses the PTV3 backbone, but GraspGen imports its
    PointNet2 modules at module import time.  On ROCm, the CUDA fallback JIT can
    fail before PTV3 is reached.  A stub keeps import-time side effects out of
    the PTV3 path and fails loudly if a PointNet2 op is actually executed.
    """
    if "pointnet2_ops" in sys.modules:
        return

    class _PointNet2OpsStub:
        def __getattr__(self, name: str):
            raise RuntimeError(
                "pointnet2_ops CUDA extension is unavailable on ROCm; "
                f"attempted to call {name!r}. Use a PTV3-backed GraspGen config."
            )

    mod = types.ModuleType("pointnet2_ops")
    mod._ext = _PointNet2OpsStub()  # type: ignore[attr-defined]
    sys.modules["pointnet2_ops"] = mod
    logger.info("Patched pointnet2_ops import for ROCm PTV3 path")


def _patch_knn_for_rocm() -> None:
    """Monkey-patch knn_points to use chunked cdist, avoiding HIP kernel crash."""
    try:
        import torch
        import grasp_gen.utils.point_cloud_utils as pcu_mod
    except ImportError:
        return

    if getattr(pcu_mod, "_rocm_patched", False):
        return

    def _chunked_knn(X: torch.Tensor, K: int, norm: int):
        N = X.shape[0]
        chunk = 2048
        all_dists = torch.full((N, K), float("inf"), device=X.device)
        all_idxs = torch.zeros((N, K), dtype=torch.long, device=X.device)
        for i in range(0, N, chunk):
            end_i = min(i + chunk, N)
            row = X[i:end_i]
            dists_row = []
            for j in range(0, N, chunk):
                end_j = min(j + chunk, N)
                d = torch.cdist(row, X[j:end_j], p=norm)
                if i == j:
                    eye = torch.eye(
                        end_i - i, end_j - j, device=X.device, dtype=torch.bool
                    )
                    d.masked_fill_(eye, float("inf"))
                dists_row.append(d)
            full_row = torch.cat(dists_row, dim=1)
            topk_d, topk_i = torch.topk(full_row, K, dim=1, largest=False)
            all_dists[i:end_i] = topk_d
            all_idxs[i:end_i] = topk_i
        return all_dists, all_idxs

    pcu_mod.knn_points = _chunked_knn
    pcu_mod._rocm_patched = True
    logger.info("Patched knn_points for ROCm (chunked cdist)")


class GraspGenModel:
    """Lazy-loaded GraspGen inference model.

    Keeps the heavy imports (torch, grasp_gen) behind __init__ so that
    code that doesn't run learned grasping doesn't pay the import cost.

    ``config_yaml`` should be the **full** inference config YAML (with
    ``eval.checkpoint``, ``discriminator.checkpoint``, etc.) — not just
    the minimal gripper definition YAML.
    """

    DEFAULT_CONFIG = "graspgen_franka_panda"

    def __init__(
        self,
        config_yaml: str | Path | None = None,
        *,
        gripper_config: str | Path | None = None,
        grasp_threshold: float = -1.0,
        num_grasps: int = 200,
        topk_num_grasps: int = 100,
        remove_outliers: bool = True,
    ):
        self._config_yaml = str(config_yaml or gripper_config or self.DEFAULT_CONFIG)
        self._grasp_threshold = grasp_threshold
        self._num_grasps = num_grasps
        self._topk_num_grasps = topk_num_grasps
        self._remove_outliers = remove_outliers
        self._sampler: Optional[object] = None

    def _ensure_loaded(self) -> None:
        if self._sampler is not None:
            return

        _patch_knn_for_rocm()
        _patch_pointnet2_import_for_rocm()

        import omegaconf
        from grasp_gen.grasp_server import GraspGenSampler

        cfg_path = Path(self._config_yaml)
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"GraspGen config not found: {cfg_path}. "
                "Provide a full inference YAML with eval.checkpoint + "
                "discriminator.checkpoint paths."
            )

        logger.info("Loading GraspGen config from %s", cfg_path)
        cfg = omegaconf.OmegaConf.load(str(cfg_path))

        ckpt_dir = cfg_path.parent
        if not Path(cfg.eval.checkpoint).is_absolute():
            cfg.eval.checkpoint = str(ckpt_dir / cfg.eval.checkpoint)
        if not Path(cfg.discriminator.checkpoint).is_absolute():
            cfg.discriminator.checkpoint = str(
                ckpt_dir / cfg.discriminator.checkpoint
            )

        # Prefer spconv_rocm-converted checkpoints (5D→3D weight format)
        for attr in ("eval.checkpoint", "discriminator.checkpoint"):
            orig = omegaconf.OmegaConf.select(cfg, attr)
            rocm_path = orig.replace(".pth", "_rocm.pth")
            if Path(rocm_path).exists():
                omegaconf.OmegaConf.update(cfg, attr, rocm_path)
                logger.info("Using ROCm-converted checkpoint: %s", rocm_path)

        self._sampler = GraspGenSampler(cfg)
        self._run_inference = GraspGenSampler.run_inference
        logger.info(
            "GraspGen model loaded (gripper: %s, backbone: %s)",
            cfg.data.gripper_name,
            cfg.diffusion.obs_backbone,
        )

    def predict(
        self,
        point_cloud: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run GraspGen inference on a point cloud.

        Args:
            point_cloud: (N, 3) object point cloud, **already centered**
                         (mean-subtracted) as GraspGen expects.

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

    def predict_from_mesh(
        self,
        mesh_path: str | Path,
        scale: float = 1.0,
        num_sample_points: int = 2000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load a mesh, sample points, run inference — follows GraspGen demo_object_mesh.py.

        Handles centering (mean-subtraction) and un-centering internally so
        the returned grasps are in the **original mesh frame**.

        NOTE: This samples the **full mesh surface** (all faces visible),
        producing a full-view point cloud. Real sensors only observe partial
        surfaces. For sim-to-real transfer, consider using ``predict()``
        directly with a depth-camera partial point cloud instead.

        Args:
            mesh_path: Path to mesh file (.obj, .stl, .ply).
            scale: Scale factor applied to the mesh before sampling.
            num_sample_points: Number of surface points to sample (default
                2000, matching GraspGen training).

        Returns:
            grasp_poses: (M, 4, 4) SE3 in the original mesh frame.
            grasp_scores: (M,) confidence scores (higher = better).
        """
        import trimesh.transformations as tra

        xyz = sample_mesh_pointcloud(
            mesh_path,
            num_sample_points,
            scale=scale,
        )

        T_center = tra.translation_matrix(-xyz.mean(axis=0))
        xyz_centered = tra.transform_points(xyz, T_center)

        poses_centered, scores = self.predict(xyz_centered)

        if len(poses_centered) == 0:
            return poses_centered, scores

        T_uncenter = tra.inverse_matrix(T_center)
        poses = np.array([T_uncenter @ g for g in poses_centered])
        return poses, scores

    def predict_from_points(
        self,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on an object-frame point cloud and return object-frame poses."""
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"point cloud must have shape (N, 3), got {points.shape}")
        if len(points) == 0:
            return (
                np.empty((0, 4, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        xyz = np.asarray(points, dtype=np.float32)
        T_center = tra.translation_matrix(-xyz.mean(axis=0))
        xyz_centered = tra.transform_points(xyz, T_center)

        poses_centered, scores = self.predict(xyz_centered)
        if len(poses_centered) == 0:
            return poses_centered, scores

        T_uncenter = tra.inverse_matrix(T_center)
        poses = np.array([T_uncenter @ g for g in poses_centered])
        return poses, scores
