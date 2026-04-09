#!/usr/bin/env python3
"""RoboSmith Demo — End-to-end pipeline: [text/image] → 3D mesh → URDF → catalog.

Usage:
    # Image mode (provide reference image)
    python demo/run_pipeline.py --image path/to/my_object.png

    # Text mode (auto-generates reference image via SDXL-Turbo T2I)
    python demo/run_pipeline.py --prompt "red ceramic mug"

    # Default: auto-detect sample image, or T2I from prompt
    python demo/run_pipeline.py

    # List available backends
    python demo/run_pipeline.py --list-backends

Requires:
    - Hunyuan3D-2.1 or Hunyuan3D-2 repo cloned
      (set HUNYUAN3D_REPO_PATH or clone into current dir)
    - GPU with ≥10GB VRAM (shape gen only)
    - For text mode: diffusers, transformers, accelerate (SDXL-Turbo, +8GB VRAM)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

T2I_PROMPT_TEMPLATE = (
    "a single {obj}, centered, isolated object, "
    "pure white background, no surface, no ground, no table, "
    "front view, slight top-down angle, "
    "clean silhouette, sharp edges, opaque, matte finish, "
    "uniform soft lighting, no shadow, no reflection, "
    "minimalist, object-only, high detail"
)

T2I_NEGATIVE_PROMPT = (
    "table, surface, floor, ground, shadow, reflection, mirror, glossy, specular, "
    "studio setup, product display, pedestal, platform, "
    "background texture, gradient background, pattern background, "
    "environment, scene, room, wall, "
    "dramatic lighting, cinematic lighting, "
    "glass, transparent, translucent, "
    "depth of field, blur, bokeh"
)


def text_to_image(prompt: str, output_path: Path, size: int = 512) -> Path:
    """Generate a 3D-reconstruction-friendly reference image via SDXL-Turbo.

    Prompt is optimized for downstream Image-to-3D: front orthographic view,
    symmetrical composition, clean background, sharp edges, no reflections.
    SDXL-Turbo uses guidance_scale=0.0 so negative prompts have no effect.
    """
    import torch
    from diffusers import AutoPipelineForText2Image

    full_prompt = T2I_PROMPT_TEMPLATE.format(obj=prompt)
    print(f"  T2I prompt: {full_prompt[:80]}...")

    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe = pipe.to("cuda")

    t0 = time.time()
    image = pipe(
        full_prompt,
        height=size,
        width=size,
        guidance_scale=0.0,
        num_inference_steps=4,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
    elapsed = time.time() - t0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"  T2I: {elapsed:.1f}s → {output_path} ({output_path.stat().st_size // 1024} KB)")

    del pipe
    torch.cuda.empty_cache()
    import gc; gc.collect()

    return output_path


def find_sample_image() -> Path | None:
    """Locate a sample image for demo purposes."""
    candidates = [
        REPO_ROOT / "demo" / "sample_images" / "mug.png",
        REPO_ROOT / "demo" / "sample_images" / "demo.png",
    ]

    hy3d_repo = os.environ.get("HUNYUAN3D_REPO_PATH", "")
    if hy3d_repo:
        candidates.append(Path(hy3d_repo) / "assets" / "demo.png")

    for cwd_hy3d in [Path.cwd() / "Hunyuan3D-2.1", REPO_ROOT / "Hunyuan3D-2.1"]:
        candidates.append(cwd_hy3d / "assets" / "demo.png")

    for c in candidates:
        if c.exists():
            return c
    return None


def run_demo(args):
    from robotsmith.assets.library import AssetLibrary
    from robotsmith.assets.builtin import bootstrap_builtin_assets

    assets_root = REPO_ROOT / "assets"

    print("=" * 60)
    print("  RoboSmith Demo — End-to-End Pipeline")
    print("=" * 60)
    print()

    # Step 1: Bootstrap built-in assets
    print("[Step 1] Bootstrapping built-in assets...")
    builtin = bootstrap_builtin_assets(assets_root)
    print(f"  ✓ {len(builtin)} built-in assets ready")
    print()

    # Step 2: Initialize library
    lib = AssetLibrary(assets_root)
    print(f"[Step 2] Asset library loaded: {len(lib)} assets")
    print()

    # Step 3: Search (should hit)
    print("[Step 3] Searching for 'red mug'...")
    results = lib.search("red mug", top_k=3)
    for r in results:
        print(f"  → {r.name} (tags: {r.tags[:5]})")
    print()

    # Step 4: Generate new asset
    backend = args.backend
    prompt = args.prompt

    gen_kwargs = {}
    image_path = args.image

    if image_path is not None:
        image_path = str(Path(image_path).resolve())
        print(f"[Step 4] Generating 3D asset with {backend} (image mode)...")
        print(f"  Input image: {image_path}")
    else:
        sample = find_sample_image()
        if sample is not None and not args.use_t2i:
            image_path = str(sample.resolve())
            print(f"[Step 4] Generating 3D asset with {backend} (sample image)...")
            print(f"  Input image: {image_path}")
        else:
            print(f"[Step 4] Generating 3D asset with {backend} (text → T2I → 3D)...")
            print(f"  Text: {prompt!r}")
            t2i_output = REPO_ROOT / "demo" / "sample_images" / "t2i_generated.png"
            image_path = str(text_to_image(prompt, t2i_output, size=512))
            print(f"  Generated reference image: {image_path}")

    gen_kwargs["image_path"] = image_path

    print(f"  Prompt: {prompt!r}")
    print(f"  Backend: {backend}")
    print()

    t0 = time.time()
    try:
        asset = lib.generate(
            prompt,
            backend=backend,
            target_size_m=args.size,
            texture=not getattr(args, 'no_texture', False),
            **gen_kwargs,
        )
    except Exception as e:
        print(f"\n[ERROR] Generation failed: {e}")
        print("\nTroubleshooting:")
        if "hy3dshape" in str(e) or "hy3dgen" in str(e) or "HUNYUAN3D" in str(e):
            print("  - Set HUNYUAN3D_REPO_PATH to your Hunyuan3D-2.1 clone dir")
            print("  - Or run: bash demo/setup_env.sh")
        else:
            print(f"  - {e}")
        sys.exit(1)

    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print("  Pipeline Results")
    print("=" * 60)
    print(f"  Asset:    {asset.name}")
    print(f"  URDF:     {asset.urdf_path}")
    if asset.visual_mesh:
        print(f"  Visual:   {asset.visual_mesh.name}")
    if asset.collision_mesh:
        print(f"  Collision: {asset.collision_mesh.name}")
    print(f"  Tags:     {asset.tags}")
    print(f"  Mass:     {asset.metadata.mass_kg:.3f} kg")
    print(f"  Size:     {asset.metadata.size_cm} cm")
    print(f"  Time:     {elapsed:.1f}s")
    print(f"  Backend:  {backend}")
    print()

    # Step 5: Verify URDF
    print("[Step 5] Verifying URDF...")
    urdf_path = asset.urdf_path
    if urdf_path.exists():
        urdf_text = urdf_path.read_text()
        has_visual = "<visual>" in urdf_text
        has_collision = "<collision>" in urdf_text
        has_inertial = "<inertial>" in urdf_text
        print(f"  URDF file: {urdf_path.stat().st_size} bytes")
        print(f"  Has <visual>:    {'✓' if has_visual else '✗'}")
        print(f"  Has <collision>: {'✓' if has_collision else '✗'}")
        print(f"  Has <inertial>:  {'✓' if has_inertial else '✗'}")
    else:
        print(f"  [WARNING] URDF not found at {urdf_path}")
    print()

    # Step 6: Mesh stats
    print("[Step 6] Mesh statistics...")
    if asset.visual_mesh and asset.visual_mesh.exists():
        import trimesh
        mesh = trimesh.load(str(asset.visual_mesh), force="mesh")
        print(f"  Vertices:   {len(mesh.vertices):,}")
        print(f"  Faces:      {len(mesh.faces):,}")
        print(f"  Watertight: {'yes' if mesh.is_watertight else 'no'}")
        print(f"  Bounds:     {mesh.bounds[0]} → {mesh.bounds[1]}")
        extents = mesh.bounding_box.extents
        print(f"  Extents:    {extents[0]:.4f} x {extents[1]:.4f} x {extents[2]:.4f} m")
    print()

    # Step 7: PyBullet validation (optional)
    try:
        import pybullet
        print("[Step 7] PyBullet validation...")
        from robotsmith.validate.pybullet_check import validate_urdf
        result = validate_urdf(urdf_path)
        status = "PASS" if result.loaded_ok else "FAIL"
        print(f"  Load: {status}")
        if result.final_position is not None:
            print(f"  Final Z: {result.final_position[2]:.3f}")
        print()
    except ImportError:
        print("[Step 7] PyBullet not installed, skipping validation")
        print("  Install with: pip install pybullet")
        print()

    # Step 8: Library summary
    print("[Step 8] Updated library:")
    print(f"  Total assets: {len(lib)}")
    for a in lib.list_all():
        marker = " ★" if a.name == asset.name else ""
        print(f"    {a.name:30s}  {a.metadata.mass_kg:.2f} kg{marker}")
    print()

    print("✓ Demo complete!")
    print()
    print("Next steps:")
    print(f"  - View in browser:   robotsmith view --asset {asset.name}")
    print(f"  - List backends:     robotsmith generate --list-backends")
    print(f"  - Search:            robotsmith search \"mug\"")


def main():
    parser = argparse.ArgumentParser(
        description="RoboSmith Demo — End-to-end pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend", default="hunyuan3d",
        choices=["hunyuan3d", "trellis2", "triposg"],
        help="3D generation backend (default: hunyuan3d)",
    )
    parser.add_argument(
        "--prompt", default="red ceramic mug",
        help="Text prompt for generation / cataloging (default: 'red ceramic mug')",
    )
    parser.add_argument(
        "--image", default=None,
        help="Input image for image-to-3D (auto-detected if not provided)",
    )
    parser.add_argument(
        "--size", type=float, default=0.1,
        help="Target size in meters (default: 0.1)",
    )
    parser.add_argument(
        "--use-t2i", action="store_true",
        help="Force text-to-image generation even if sample images exist",
    )
    parser.add_argument(
        "--no-texture", action="store_true",
        help="Disable PBR textures (shape-only, faster, no bpy dependency)",
    )
    parser.add_argument(
        "--list-backends", action="store_true",
        help="List available backends and exit",
    )

    args = parser.parse_args()

    if args.list_backends:
        from robotsmith.gen.backend import list_backend_info
        print("Available 3D generation backends:\n")
        for info in list_backend_info():
            avail = "✓ verified" if info.rocm_status == "verified" else info.rocm_status
            pbr = "PBR" if info.has_pbr else "no PBR"
            print(f"  {info.name:12s}  {info.model_name:35s}  {pbr:8s}  VRAM≥{info.min_vram_gb:.0f}GB  ROCm: {avail}")
        return

    run_demo(args)


if __name__ == "__main__":
    main()
