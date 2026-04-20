"""Orchestrator: 3D gen -> mesh_to_urdf -> catalog -> return Asset.

Uses the pluggable GenBackend registry. Available backends are discovered
automatically at import time from robotsmith.gen.backend.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from robotsmith.assets.library import AssetLibrary
    from robotsmith.assets.schema import Asset


def generate_and_catalog(
    library: AssetLibrary,
    prompt: str,
    backend: str = "trellis2",
    target_size_m: float = 0.1,
    mass_kg: Optional[float] = None,
    density_kg_m3: float = 800.0,
    texture: bool = True,
    texture_size: Optional[int] = None,
    decimation_target: Optional[int] = None,
    **gen_kwargs,
) -> Asset:
    """Full pipeline: generate mesh -> convert to URDF -> catalog -> register in library.

    Args:
        library: AssetLibrary to register the new asset in.
        prompt: Text description for 3D generation.
        backend: Generation backend name (e.g. "trellis2", "hunyuan3d", "triposg").
        target_size_m: Scale mesh so longest edge equals this (meters).
        mass_kg: Override mass; None for auto-estimate from volume * density.
        density_kg_m3: Density for mass estimation.
        texture_size: PBR texture resolution (default 1024; 512 for speed, 4096 for quality).
        decimation_target: Target face count for mesh decimation (default 200K).
        **gen_kwargs: Extra args passed to the backend's generate() method.

    Returns:
        The newly created and cataloged Asset.
    """
    from robotsmith.gen.backend import get_backend, list_backends
    from robotsmith.gen.catalog import name_from_prompt, catalog_asset
    from robotsmith.gen.mesh_to_urdf import mesh_to_urdf

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    asset_name = f"{name_from_prompt(prompt)}_{timestamp}"
    output_dir = library.root / "generated" / asset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    backend_kwargs: dict = {"texture": texture}
    if texture_size is not None:
        backend_kwargs["texture_size"] = texture_size
    if decimation_target is not None:
        backend_kwargs["decimation_target"] = decimation_target
    gen_backend = get_backend(backend, **backend_kwargs)
    backend_info = gen_backend.info
    print(f"[robotsmith] Using backend: {backend_info.model_name}")
    print(f"[robotsmith]   PBR textures: {'enabled → GLB' if backend_info.has_pbr else 'off → OBJ'}")

    mesh = gen_backend.generate(prompt, **gen_kwargs)

    t_gen = time.time() - t0

    # Save T2I reference image if provided
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
    t_convert = time.time() - t0 - t_gen

    asset = catalog_asset(output_dir, prompt, mass_kg=mass_kg or 0.1)
    library.add(asset)

    total = time.time() - t0
    print(f"[robotsmith] Generated {asset_name} in {total:.1f}s "
          f"(gen={t_gen:.1f}s, convert={t_convert:.1f}s, backend={backend})")

    return asset
