"""TRELLIS.2 backend: image to 3D mesh with full PBR materials.

Microsoft TRELLIS.2 (4B params, CVPR'25 Spotlight).
O-Voxel sparse representation, handles complex topology, full PBR output.

Requires: CUDA (official) or ROCm community fork.
See: https://github.com/microsoft/TRELLIS.2
ROCm fork: https://github.com/Lamothe/TRELLIS.2_rocm

ROCm status: community fork exists (experimental), CUDA deps (FlexiBox, Kaolin, nvdiffrast).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from robotsmith.gen.backend import GenBackend, BackendInfo, register_backend


@register_backend("trellis2")
class Trellis2Backend(GenBackend):

    def __init__(self, device: str = "auto", resolution: int = 512):
        self._device = device
        self._resolution = resolution

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            name="trellis2",
            model_name="TRELLIS.2 4B (Microsoft)",
            version="2.0",
            has_pbr=True,
            min_vram_gb=24.0,
            rocm_status="community_fork",
            description=(
                "State-of-the-art mesh quality (CVPR'25 Spotlight). "
                "O-Voxel representation, handles open surfaces and non-manifold geometry. "
                "Full PBR: base color, metallic, roughness, opacity. "
                "~3s@512³, ~17s@1024³ on H100. "
                "ROCm: community fork exists but experimental (CUDA deps: FlexiBox, Kaolin, nvdiffrast)."
            ),
            install_hint=(
                "git clone https://github.com/microsoft/TRELLIS.2 && "
                "cd TRELLIS.2 && pip install -r requirements.txt\n"
                "ROCm fork: https://github.com/Lamothe/TRELLIS.2_rocm"
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
            "TRELLIS.2 backend not yet implemented. "
            "ROCm verification needed (heavy CUDA deps). See docs/design.md.\n"
            "Install: https://github.com/microsoft/TRELLIS.2\n"
            "ROCm fork: https://github.com/Lamothe/TRELLIS.2_rocm"
        )
