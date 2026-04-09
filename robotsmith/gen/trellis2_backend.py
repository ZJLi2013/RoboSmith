"""TRELLIS.2 backend: image to 3D mesh with full PBR materials.

Microsoft TRELLIS.2 (4B params) — state-of-the-art image-to-3D with O-Voxel
sparse representation.  Handles complex topology (open surfaces, non-manifold),
outputs PBR materials (base color, metallic, roughness, opacity).

ROCm fork: https://github.com/ZJLi2013/TRELLIS.2/tree/rocm
Verified on MI300X (gfx942) + ROCm 6.4.

Performance (H100 reference):
  512³  ~3s  |  1024³  ~17s  |  1536³  ~60s

Usage::

    backend = get_backend("trellis2", resolution=512)
    mesh = backend.generate("red ceramic mug", image_path="mug.png")
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

from robotsmith.gen.backend import GenBackend, BackendInfo, register_backend

_TRELLIS2_REPO_ENV = "TRELLIS2_REPO_PATH"
_HF_MODEL_ID = "microsoft/TRELLIS.2-4B"


def _find_trellis2_repo() -> Optional[Path]:
    """Locate the cloned TRELLIS.2 repo on disk."""
    if _TRELLIS2_REPO_ENV in os.environ:
        p = Path(os.environ[_TRELLIS2_REPO_ENV])
        if p.is_dir() and (p / "trellis2").is_dir():
            return p

    search_names = ["TRELLIS.2", "TRELLIS2", "trellis2"]
    search_roots = [Path.cwd(), Path.home(), Path("/data"), Path("/data/shared"), Path("/tmp")]

    for root in search_roots:
        for name in search_names:
            c = root / name
            if c.is_dir() and (c / "trellis2").is_dir():
                return c
    return None


@register_backend("trellis2")
class Trellis2Backend(GenBackend):

    def __init__(
        self,
        device: str = "auto",
        texture: bool = True,
        resolution: int = 512,
        repo_path: Optional[str | Path] = None,
        model_id: Optional[str] = None,
        decimation_target: int = 1_000_000,
        texture_size: int = 4096,
    ):
        self._device = device
        self._texture = texture
        self._resolution = resolution
        self._decimation_target = decimation_target
        self._texture_size = texture_size
        self._pipeline = None

        if repo_path is not None:
            self._repo_path = Path(repo_path)
        else:
            self._repo_path = _find_trellis2_repo()

        self._model_id = model_id or _HF_MODEL_ID

    @property
    def info(self) -> BackendInfo:
        res = self._resolution
        return BackendInfo(
            name="trellis2",
            model_name=f"TRELLIS.2-4B (Microsoft, {res}³)",
            version="2.0",
            has_pbr=self._texture,
            min_vram_gb=24.0,
            rocm_status="verified",
            description=(
                "State-of-the-art 3D generation (4B params). "
                "O-Voxel representation, handles open surfaces and non-manifold geometry. "
                "Full PBR: base color, metallic, roughness, opacity. "
                f"Resolution: {res}³. "
                "ROCm: verified on MI300X via ZJLi2013/TRELLIS.2@rocm."
            ),
            install_hint=(
                "git clone -b rocm https://github.com/ZJLi2013/TRELLIS.2.git --recursive && "
                "cd TRELLIS.2 && . ./setup.sh --basic --flash-attn --nvdiffrast "
                "--nvdiffrec --cumesh --o-voxel --flexgemm"
            ),
        )

    def is_available(self) -> bool:
        if self._repo_path is None or not self._repo_path.is_dir():
            return False
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return

        if self._repo_path is None:
            raise RuntimeError(
                f"TRELLIS.2 repo not found. Either:\n"
                f"  1. Set env var {_TRELLIS2_REPO_ENV}=/path/to/TRELLIS.2\n"
                f"  2. Clone: git clone -b rocm https://github.com/ZJLi2013/TRELLIS.2.git --recursive\n"
                f"  3. Pass repo_path= to the backend constructor"
            )

        import torch

        repo_root = str(self._repo_path)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if torch.cuda.is_available() and hasattr(torch.version, "hip"):
            os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")

        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        device = self._device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[trellis2] Repo: {self._repo_path}")
        print(f"[trellis2] Loading pipeline from {self._model_id} ...")
        self._pipeline = Trellis2ImageTo3DPipeline.from_pretrained(self._model_id)
        self._pipeline.to(device)
        self._pipeline_device = device
        print(f"[trellis2] Pipeline ready on {device}")

    def generate(
        self,
        prompt: str,
        output_path: Optional[str | Path] = None,
        image_path: Optional[str | Path] = None,
        **kwargs,
    ) -> "trimesh.Trimesh":
        """Generate a 3D mesh with PBR textures from an image.

        Args:
            prompt: Text description (used for cataloging/tagging).
            output_path: Optional path to export the mesh (GLB/OBJ).
            image_path: Path to input image (required).
        """
        import trimesh
        from PIL import Image

        self._ensure_pipeline()

        if image_path is None:
            raise ValueError(
                "TRELLIS.2 requires an input image. "
                "Pass image_path='/path/to/image.png'.\n"
                "Tip: generate a reference image first via SDXL-Turbo T2I."
            )

        image_path_str = str(Path(image_path).resolve())
        print(f"[trellis2] Generating from {image_path_str} (resolution={self._resolution}³) ...")

        image = Image.open(image_path_str)
        results = self._pipeline.run(image)
        mesh_output = results[0] if isinstance(results, (list, tuple)) else results

        n_verts = len(mesh_output.vertices) if hasattr(mesh_output, "vertices") else 0
        n_faces = len(mesh_output.faces) if hasattr(mesh_output, "faces") else 0
        print(f"[trellis2] Raw mesh: {n_verts:,} vertices, {n_faces:,} faces")

        if hasattr(mesh_output, "simplify"):
            mesh_output.simplify(min(n_faces, 16_777_216))

        mesh = self._export_glb_and_load(mesh_output)

        if output_path is not None:
            out = Path(output_path)
            suffix = out.suffix.lower()
            if suffix == ".glb":
                mesh.export(str(out), file_type="glb")
            elif suffix == ".obj":
                mesh.export(str(out), file_type="obj")
            else:
                mesh.export(str(out))
            print(f"[trellis2] Exported to {out}")

        return mesh

    def _export_glb_and_load(self, mesh_output) -> "trimesh.Trimesh":
        """Export TRELLIS.2 mesh to GLB via o_voxel, then load as trimesh."""
        import trimesh

        has_ovoxel_attrs = all(
            hasattr(mesh_output, a) for a in ("attrs", "coords", "layout", "voxel_size")
        )

        if has_ovoxel_attrs and self._texture:
            import o_voxel

            print(f"[trellis2] Exporting PBR GLB (decimation={self._decimation_target:,}, "
                  f"tex={self._texture_size}) ...")

            glb = o_voxel.postprocess.to_glb(
                vertices=mesh_output.vertices,
                faces=mesh_output.faces,
                attr_volume=mesh_output.attrs,
                coords=mesh_output.coords,
                attr_layout=mesh_output.layout,
                voxel_size=mesh_output.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=self._decimation_target,
                texture_size=self._texture_size,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=True,
            )

            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                glb.export(tmp_path, extension_webp=True)
                loaded = trimesh.load(tmp_path, process=False)
                if isinstance(loaded, trimesh.Scene):
                    meshes = list(loaded.geometry.values())
                    if meshes:
                        mesh = trimesh.util.concatenate(meshes)
                    else:
                        mesh = self._fallback_trimesh(mesh_output)
                else:
                    mesh = loaded
                print(f"[trellis2] PBR GLB loaded: {len(mesh.vertices):,} verts, textured=True")
                return mesh
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        return self._fallback_trimesh(mesh_output)

    @staticmethod
    def _fallback_trimesh(mesh_output) -> "trimesh.Trimesh":
        """Convert raw mesh output to trimesh without PBR."""
        import trimesh
        import numpy as np

        if isinstance(mesh_output, trimesh.Trimesh):
            return mesh_output

        verts = mesh_output.vertices
        faces = mesh_output.faces
        if not isinstance(verts, np.ndarray):
            verts = np.array(verts)
        if not isinstance(faces, np.ndarray):
            faces = np.array(faces)

        return trimesh.Trimesh(vertices=verts, faces=faces)
