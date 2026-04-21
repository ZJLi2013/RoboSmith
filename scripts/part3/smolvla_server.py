"""SmolVLA model server for vla-eval-harness.

Loads a SmolVLA checkpoint and serves predictions via vla-eval's ModelServer protocol.

Usage:
    vla-eval serve scripts/part3/smolvla_server.py \
        --checkpoint /path/to/checkpoint \
        --dataset-id user/dataset_name
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vla_eval.model_servers.base import ModelServer
from vla_eval.types import Action, Observation

JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "joint5", "joint6", "joint7",
    "finger_joint1", "finger_joint2",
]


class SmolVLAServer(ModelServer):
    """SmolVLA inference server for RoboSmith eval."""

    def __init__(
        self,
        checkpoint: str,
        dataset_id: str,
        dataset_root: str | None = None,
        action_horizon: int = 1,
        device: str = "cuda",
        **kwargs: Any,
    ):
        self._checkpoint = checkpoint
        self._dataset_id = dataset_id
        self._dataset_root = dataset_root
        self._action_horizon = action_horizon
        self._device = device
        self._action_queue: list[np.ndarray] = []
        self._n_dofs = len(JOINT_NAMES)

        self._load_model()

    def _load_model(self) -> None:
        try:
            from lerobot.configs.types import FeatureType
            from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
            from lerobot.common.datasets.utils import dataset_to_policy_features
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.policies.factory import make_pre_post_processors
        except ImportError:
            from lerobot.configs.types import FeatureType
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
            from lerobot.datasets.utils import dataset_to_policy_features
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.policies.factory import make_pre_post_processors

        metadata = LeRobotDatasetMetadata(self._dataset_id, root=self._dataset_root)
        features = dataset_to_policy_features(metadata.features)
        output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {k: ft for k, ft in features.items() if k not in output_features}

        ckpt_dir = Path(self._checkpoint)
        import json
        config_path = ckpt_dir / "config.json"
        saved_cfg = {}
        if config_path.exists():
            saved_cfg = json.loads(config_path.read_text(encoding="utf-8"))
        chunk_size = saved_cfg.get("chunk_size", 50)

        cfg = SmolVLAConfig(
            input_features=input_features,
            output_features=output_features,
            chunk_size=chunk_size,
            n_action_steps=chunk_size,
        )

        try:
            policy = SmolVLAPolicy.from_pretrained(str(ckpt_dir), config=cfg, strict=False)
        except Exception:
            from safetensors.torch import load_file
            policy = SmolVLAPolicy(cfg)
            st = load_file(str(ckpt_dir / "model.safetensors"))
            policy.load_state_dict(st, strict=False)

        policy.eval().to(self._device)
        self._policy = policy

        self._preprocessor, self._postprocessor = make_pre_post_processors(
            cfg, dataset_stats=metadata.stats,
        )

        self._task_instruction = ""

        total_params = sum(p.numel() for p in policy.parameters())
        print(f"[smolvla_server] Loaded: {total_params:,} params, chunk_size={chunk_size}")

    async def on_observation(self, obs: Observation, ctx: Any) -> None:
        if ctx.is_first:
            self._action_queue.clear()
            task_info = getattr(ctx, "task", None) or {}
            if isinstance(task_info, dict):
                self._task_instruction = task_info.get("name", "")

        if self._action_queue:
            action_arr = self._action_queue.pop(0)
            await ctx.send_action({"actions": action_arr})
            return

        state = obs.get("state")
        images = obs.get("images", {})
        task_desc = obs.get("task_description", self._task_instruction)

        state_np = np.asarray(state, dtype=np.float32) if state is not None else np.zeros(self._n_dofs, dtype=np.float32)

        chunk = self._predict(state_np, images, task_desc)

        horizon = min(self._action_horizon, len(chunk))
        for i in range(1, horizon):
            self._action_queue.append(chunk[i])

        await ctx.send_action({"actions": chunk[0]})

    @torch.no_grad()
    def _predict(
        self,
        state_np: np.ndarray,
        images: dict[str, np.ndarray],
        task: str,
    ) -> np.ndarray:
        obs = {
            "observation.state": torch.from_numpy(state_np).unsqueeze(0).float().to(self._device),
            "task": [task],
        }

        for cam_name, img in images.items():
            img_arr = np.asarray(img, dtype=np.float32)
            if img_arr.ndim == 3 and img_arr.shape[2] in (3, 4):
                img_t = torch.from_numpy(img_arr).permute(2, 0, 1)
            else:
                img_t = torch.from_numpy(img_arr)
            img_t = img_t.unsqueeze(0).float().to(self._device) / 255.0
            obs[f"observation.images.{cam_name}"] = img_t

        obs = self._preprocessor(obs)
        raw = self._policy.select_action(obs)
        raw = self._postprocessor(raw)

        if isinstance(raw, dict):
            actions = raw["action"]
        else:
            actions = raw

        actions = actions.detach().cpu().numpy()
        if actions.ndim == 3:
            actions = actions[0]
        elif actions.ndim == 1:
            actions = actions[np.newaxis, :]

        return actions[:, :self._n_dofs]

    async def on_episode_start(self, config: dict[str, Any], ctx: Any) -> None:
        self._action_queue.clear()
        task_info = config.get("task", {})
        if isinstance(task_info, dict):
            self._task_instruction = task_info.get("name", "")

    async def on_episode_end(self, result: dict[str, Any], ctx: Any) -> None:
        self._action_queue.clear()


def create_model_server(**kwargs: Any) -> SmolVLAServer:
    """Factory function for vla-eval model server config."""
    return SmolVLAServer(**kwargs)
