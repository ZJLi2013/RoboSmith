"""TripoSG backend: image to high-fidelity 3D mesh.

VAST-AI TripoSG (1.5B params, MoE Transformer).
Successor to TripoSR with significantly better mesh quality.
Baked texture output (not PBR), can be combined with Hunyuan3D-Paint for PBR.

Requires: pip install torch  (+ TripoSG repo dependencies)
See: https://github.com/VAST-AI-Research/TripoSG

ROCm status: pure PyTorch, likely compatible — needs verification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from robotsmith.gen.backend import GenBackend, BackendInfo, register_backend


@register_backend("triposg")
class TripoSGBackend(GenBackend):

    def __init__(self, device: str = "auto"):
        self._device = device

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            name="triposg",
            model_name="TripoSG 1.5B (VAST-AI)",
            version="1.0",
            has_pbr=False,
            min_vram_gb=6.0,
            rocm_status="likely",
            description=(
                "High-fidelity mesh generation via MoE Transformer + Rectified Flow. "
                "Successor to TripoSR with much better quality. "
                "Low VRAM (≥6GB). Baked texture only (not PBR). "
                "Can combine with Hunyuan3D-Paint for PBR texture."
            ),
            install_hint=(
                "git clone https://github.com/VAST-AI-Research/TripoSG && "
                "cd TripoSG && pip install -r requirements.txt"
            ),
        )

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            return False  # not yet installed
        except ImportError:
            return False

    def generate(
        self,
        prompt: str,
        output_path: Optional[str | Path] = None,
        **kwargs,
    ) -> "trimesh.Trimesh":
        raise NotImplementedError(
            "TripoSG backend not yet implemented. "
            "ROCm verification needed. See docs/design.md.\n"
            "Install: https://github.com/VAST-AI-Research/TripoSG"
        )
