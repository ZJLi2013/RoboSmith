"""Hunyuan3D-2.1 backend: image to high-fidelity 3D mesh with PBR textures.

Tencent Hunyuan3D-2.1 — production-ready, fully open-source (weights + training code).
Two-stage: DiT shape gen (3.3B, 10GB VRAM) + PBR texture paint (2B, 21GB VRAM).

ROCm status: ✅ VERIFIED on MI300X (gfx942) + ROCm 6.4.
  - Shape gen: 60s, 344K verts, AOTriton Flash Attention
  - custom_rasterizer: builds with --no-build-isolation
  - DifferentiableRenderer: builds with pybind11

Supports two repo layouts:
  - Hunyuan3D-2.1 repo (hy3dshape): https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
  - Hunyuan3D-2   repo (hy3dgen):  https://github.com/Tencent-Hunyuan/Hunyuan3D-2

HuggingFace weights: tencent/Hunyuan3D-2.1  or  tencent/Hunyuan3D-2
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

from robotsmith.gen.backend import GenBackend, BackendInfo, register_backend

_HUNYUAN3D_REPO_ENV = "HUNYUAN3D_REPO_PATH"
_HF_MODEL_ID_V21 = "tencent/Hunyuan3D-2.1"
_HF_MODEL_ID_V2 = "tencent/Hunyuan3D-2"


def _detect_repo(repo_path: Path) -> Optional[str]:
    """Detect which Hunyuan3D repo layout is present.

    Returns "v2.1" for the standalone Hunyuan3D-2.1 repo (hy3dshape/hy3dpaint),
    "v2" for the main Hunyuan3D-2 repo (hy3dgen), or None.
    """
    if (repo_path / "hy3dshape").is_dir():
        return "v2.1"
    if (repo_path / "hy3dgen").is_dir():
        return "v2"
    return None


def _find_hunyuan3d_repo() -> Optional[Path]:
    """Locate the cloned Hunyuan3D repo on disk."""
    if _HUNYUAN3D_REPO_ENV in os.environ:
        p = Path(os.environ[_HUNYUAN3D_REPO_ENV])
        if p.is_dir() and _detect_repo(p) is not None:
            return p

    search_names = ["Hunyuan3D-2.1", "Hunyuan3D-2"]
    search_roots = [Path.cwd(), Path.home(), Path("/data"), Path("/tmp")]

    for root in search_roots:
        for name in search_names:
            c = root / name
            if c.is_dir() and _detect_repo(c) is not None:
                return c
    return None


@register_backend("hunyuan3d")
class Hunyuan3DBackend(GenBackend):

    def __init__(
        self,
        device: str = "auto",
        texture: bool = True,
        repo_path: Optional[str | Path] = None,
        model_id: Optional[str] = None,
    ):
        self._device = device
        self._texture = texture
        self._pipeline = None
        self._paint_pipeline = None
        self._repo_variant: Optional[str] = None

        if repo_path is not None:
            self._repo_path = Path(repo_path)
        else:
            self._repo_path = _find_hunyuan3d_repo()

        if self._repo_path is not None:
            self._repo_variant = _detect_repo(self._repo_path)

        if model_id is not None:
            self._model_id = model_id
        elif self._repo_variant == "v2":
            self._model_id = _HF_MODEL_ID_V2
        else:
            self._model_id = _HF_MODEL_ID_V21

    @property
    def info(self) -> BackendInfo:
        mode = "shape + PBR texture" if self._texture else "shape only"
        return BackendInfo(
            name="hunyuan3d",
            model_name=f"Hunyuan3D-2.1 (Tencent, {mode})",
            version="2.1",
            has_pbr=self._texture,
            min_vram_gb=29.0 if self._texture else 10.0,
            rocm_status="verified",
            description=(
                "Production-ready image-to-3D. "
                "Shape (3.3B, ~60s) + PBR Paint (2B, ~30-60s). "
                "Default: shape + PBR → textured GLB. "
                "Use texture=False for shape-only white mesh."
            ),
            install_hint=(
                "git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 && "
                "cd Hunyuan3D-2.1 && pip install -r requirements.txt && "
                "cd hy3dpaint/custom_rasterizer && pip install -e . --no-build-isolation"
            ),
        )

    def is_available(self) -> bool:
        if self._repo_path is None or not self._repo_path.is_dir():
            return False
        if self._repo_variant is None:
            return False
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return

        if self._repo_path is None or self._repo_variant is None:
            raise RuntimeError(
                f"Hunyuan3D repo not found. Either:\n"
                f"  1. Set env var {_HUNYUAN3D_REPO_ENV}=/path/to/Hunyuan3D-2.1\n"
                f"  2. Clone: git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1\n"
                f"  3. Or:    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2\n"
                f"  4. Pass repo_path= to the backend constructor"
            )

        import torch

        if self._repo_variant == "v2.1":
            shape_dir = str(self._repo_path / "hy3dshape")
            if shape_dir not in sys.path:
                sys.path.insert(0, shape_dir)
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        else:
            repo_root = str(self._repo_path)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        device = self._device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[hunyuan3d] Repo variant: {self._repo_variant} ({self._repo_path.name})")
        print(f"[hunyuan3d] Loading shape pipeline from {self._model_id} ...")
        self._pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            self._model_id
        )
        self._pipeline_device = device
        print(f"[hunyuan3d] Shape pipeline ready on {device}")

        if self._texture:
            self._ensure_paint_pipeline()

    def _ensure_paint_pipeline(self):
        """Load the PBR texture painting pipeline (Hunyuan3D-Paint)."""
        if self._paint_pipeline is not None:
            return

        if self._repo_variant == "v2.1":
            paint_dir = str(self._repo_path / "hy3dpaint")
            if paint_dir not in sys.path:
                sys.path.insert(0, paint_dir)
            from textureGenPipeline import Hunyuan3DPaintPipeline
        else:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

        print(f"[hunyuan3d] Loading PBR paint pipeline ...")
        self._paint_pipeline = Hunyuan3DPaintPipeline()
        print(f"[hunyuan3d] Paint pipeline ready (PBR textures enabled)")

    def generate(
        self,
        prompt: str,
        output_path: Optional[str | Path] = None,
        image_path: Optional[str | Path] = None,
        num_inference_steps: int = 50,
        **kwargs,
    ) -> "trimesh.Trimesh":
        """Generate a 3D mesh (with PBR textures by default) from an image.

        When texture=True (default), runs Shape → Paint → textured GLB.
        When texture=False, runs Shape only → white mesh.

        Args:
            prompt: Text description (used for cataloging / tagging).
            output_path: Optional path to export the mesh (GLB/OBJ).
            image_path: Path to input image for image-to-3D generation (required).
            num_inference_steps: Diffusion sampling steps (default 50).
        """
        import trimesh

        self._ensure_pipeline()

        if image_path is None:
            raise ValueError(
                "Hunyuan3D-2.1 requires an input image. "
                "Pass image_path='/path/to/image.png'.\n"
                "Tip: generate a reference image first via Stable Diffusion / FLUX, "
                "then pass it here."
            )

        image_path_str = str(Path(image_path).resolve())
        print(f"[hunyuan3d] Generating shape from {image_path_str} ...")

        pipeline_kwargs = {"image": image_path_str}
        if num_inference_steps != 50:
            pipeline_kwargs["num_inference_steps"] = num_inference_steps
        for k, v in kwargs.items():
            if k not in ("image_path",):
                pipeline_kwargs[k] = v

        result = self._pipeline(**pipeline_kwargs)
        mesh_output = result[0] if isinstance(result, (list, tuple)) else result
        mesh = self._to_trimesh(mesh_output)

        n_verts = len(mesh.vertices)
        n_faces = len(mesh.faces)
        print(f"[hunyuan3d] Shape: {n_verts:,} vertices, {n_faces:,} faces")

        if self._texture and self._paint_pipeline is not None:
            mesh = self._apply_paint(mesh, image_path_str)

        if output_path is not None:
            out = Path(output_path)
            suffix = out.suffix.lower()
            if suffix == ".glb":
                mesh.export(str(out), file_type="glb")
            elif suffix == ".obj":
                mesh.export(str(out), file_type="obj")
            else:
                mesh.export(str(out))
            print(f"[hunyuan3d] Exported to {out}")

        return mesh

    def _apply_paint(self, mesh, image_path: str) -> "trimesh.Trimesh":
        """Run the PBR texture painting stage on a white mesh.

        Uses save_glb=False so bpy is never invoked (Blender has no Python
        3.12 wheel).  The paint pipeline outputs a textured OBJ with PBR
        maps (diffuse, metallic, roughness, normal).  We load the OBJ with
        materials intact (textures read into memory), then the downstream
        mesh_to_urdf exports a self-contained GLB with embedded textures.
        """
        import shutil
        import trimesh

        print("[hunyuan3d] Painting PBR textures ...")

        tmpdir = tempfile.mkdtemp(prefix="hy3d_paint_")
        shape_glb = os.path.join(tmpdir, "shape.glb")
        output_obj = os.path.join(tmpdir, "textured.obj")
        try:
            mesh.export(shape_glb, file_type="glb")
            self._paint_pipeline(
                mesh_path=shape_glb,
                image_path=image_path,
                output_mesh_path=output_obj,
                save_glb=False,
            )

            textured = self._load_textured_mesh(tmpdir, output_obj)
            if textured is not None:
                return textured

            print("[hunyuan3d] Paint produced no output, using untextured mesh")
            return mesh
        except Exception as e:
            print(f"[hunyuan3d] Paint failed ({e}), falling back to untextured mesh")
            return mesh
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @staticmethod
    def _load_textured_mesh(tmpdir: str, output_obj: str):
        """Load textured OBJ preserving materials so GLB export embeds textures."""
        import trimesh

        candidates = [output_obj]
        candidates += [
            str(f) for f in Path(tmpdir).iterdir()
            if f.suffix in (".obj", ".glb") and "textured" in f.stem
        ]

        for path in candidates:
            if not os.path.exists(path):
                continue
            loaded = trimesh.load(path, process=False)
            if isinstance(loaded, trimesh.Scene):
                meshes = list(loaded.geometry.values())
                if meshes:
                    textured = trimesh.util.concatenate(meshes)
                else:
                    continue
            else:
                textured = loaded

            tex_maps = list(Path(tmpdir).glob("textured*.jpg"))
            has_tex = (
                hasattr(textured, "visual")
                and hasattr(textured.visual, "material")
                and textured.visual.material is not None
            )
            print(f"[hunyuan3d] PBR textures applied "
                  f"({len(tex_maps)} map(s), {len(textured.vertices):,} verts, "
                  f"textured={has_tex})")
            return textured

        return None

    @staticmethod
    def _to_trimesh(mesh_output) -> "trimesh.Trimesh":
        """Convert various pipeline output types to trimesh.Trimesh."""
        import trimesh
        import numpy as np

        if isinstance(mesh_output, trimesh.Trimesh):
            return mesh_output
        if isinstance(mesh_output, trimesh.Scene):
            return trimesh.util.concatenate(mesh_output.dump())
        if hasattr(mesh_output, "vertices") and hasattr(mesh_output, "faces"):
            verts = mesh_output.vertices
            faces = mesh_output.faces
            if not isinstance(verts, np.ndarray):
                verts = np.array(verts)
            if not isinstance(faces, np.ndarray):
                faces = np.array(faces)
            return trimesh.Trimesh(vertices=verts, faces=faces)
        if hasattr(mesh_output, "export"):
            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                mesh_output.export(tmp_path)
                return trimesh.load(tmp_path, force="mesh")
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        raise TypeError(
            f"Unexpected pipeline output type: {type(mesh_output)}. "
            f"Expected trimesh.Trimesh or object with .vertices/.faces"
        )
