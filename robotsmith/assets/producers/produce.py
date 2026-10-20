"""Produce a sim-ready asset directory from a generation prompt.

Pipeline: 3D gen backend -> trimesh -> mesh_to_urdf -> asset directory.

It writes files only (model.urdf, visual mesh, collision.obj, reference.png,
generation.json) into an output directory; ``AssetLibrary`` loads the result.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from robotsmith.assets.producers.backend import get_backend
from robotsmith.assets.producers.mesh_to_urdf import mesh_to_urdf

CANONICAL_ASSET_CONSTRAINT = (
    "Use a robotics-ready canonical asset frame: the object stands upright, "
    "+Z is semantic up, +X is semantic front when applicable, the object is "
    "centered with a clean origin, and the mesh is meter-scale."
)


def _with_canonical_asset_constraint(prompt: str) -> str:
    """Append generation-time frame constraints without duplicating them."""
    if "canonical asset frame" in prompt.lower() or "+z is semantic up" in prompt.lower():
        return prompt
    return f"{prompt.strip()}. {CANONICAL_ASSET_CONSTRAINT}"


def generate_asset(
    prompt: str,
    output_dir: str | Path,
    *,
    backend: str = "trellis2",
    name: Optional[str] = None,
    target_size_m: float = 0.1,
    mass_kg: Optional[float] = None,
    density_kg_m3: float = 800.0,
    texture: bool = True,
    texture_size: Optional[int] = None,
    decimation_target: Optional[int] = None,
    **gen_kwargs,
) -> Path:
    """Generate a 3D mesh and convert it into a sim-ready asset directory.

    Args:
        prompt: Text description for 3D generation.
        output_dir: Directory to write the asset files into (created if needed).
        backend: Generation backend name ("trellis2", "hunyuan3d").
        name: Asset name used for URDF/link naming (defaults to output_dir name).
        target_size_m: Scale mesh so longest edge equals this (meters).
        mass_kg: Override mass; None for auto-estimate from volume * density.
        density_kg_m3: Density for mass estimation.
        texture: Whether to keep PBR textures (GLB) vs untextured (OBJ).
        texture_size: PBR texture resolution (backend default if None).
        decimation_target: Target face count for decimation (backend default if None).
        **gen_kwargs: Extra args passed to the backend's generate() (e.g. image_path).

    Returns:
        The output directory (Path) containing model.urdf and meshes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_name = name or output_dir.name

    t0 = time.time()

    backend_kwargs: dict = {"texture": texture}
    if texture_size is not None:
        backend_kwargs["texture_size"] = texture_size
    if decimation_target is not None:
        backend_kwargs["decimation_target"] = decimation_target

    gen_backend = get_backend(backend, **backend_kwargs)
    backend_info = gen_backend.info
    print(f"[produce] Using backend: {backend_info.model_name}")
    print(f"[produce]   PBR textures: {'enabled -> GLB' if backend_info.has_pbr else 'off -> OBJ'}")

    generation_prompt = _with_canonical_asset_constraint(prompt)
    mesh = gen_backend.generate(generation_prompt, **gen_kwargs)
    t_gen = time.time() - t0

    ref_src = gen_kwargs.get("image_path")
    if ref_src is not None:
        import shutil

        ref_dst = output_dir / "reference.png"
        if not ref_dst.exists():
            shutil.copy2(str(ref_src), str(ref_dst))

    mesh_to_urdf(
        mesh,
        output_dir,
        name=asset_name,
        target_size_m=target_size_m,
        mass_kg=mass_kg,
        density_kg_m3=density_kg_m3,
    )
    total = time.time() - t0

    manifest = {
        "name": asset_name,
        "prompt": prompt,
        "backend": backend,
        "model_name": backend_info.model_name,
        "target_size_m": target_size_m,
        "density_kg_m3": density_kg_m3,
        "texture": texture,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "gen_seconds": round(t_gen, 1),
        "total_seconds": round(total, 1),
    }
    (output_dir / "generation.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"[produce] Produced {asset_name} in {total:.1f}s "
          f"(gen={t_gen:.1f}s, backend={backend}) -> {output_dir}")

    return output_dir
