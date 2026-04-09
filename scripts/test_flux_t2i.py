"""T2I bridge verification on ROCm.

Tests multiple ungated T2I models for generating reference images
suitable as Hunyuan3D-2.1 input (single object, white bg, 512x512).

Model priority:
  1. SDXL-Turbo (fast, 1 step, ungated) — for speed test
  2. SDXL base (high quality, ungated) — for quality test
  3. FLUX.1-schnell (if HF token available) — best quality
"""
import time
import os
import sys
import torch

print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory
    print(f"VRAM: {total_mem / 1e9:.1f} GB")

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

test_objects = [
    ("red_mug", "red ceramic mug"),
    ("wooden_fork", "wooden fork"),
    ("blue_bottle", "blue plastic bottle"),
]

os.makedirs("/data/t2i_output", exist_ok=True)


def test_model(model_id, short_name, num_steps, guidance_scale, use_refiner=False, resolution=512):
    """Test a T2I model and generate reference images."""
    from diffusers import AutoPipelineForText2Image

    print(f"\n{'='*60}")
    print(f"Model: {model_id}")
    print(f"Steps: {num_steps}, Guidance: {guidance_scale}, Resolution: {resolution}")
    print(f"{'='*60}")

    t0 = time.time()
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe = pipe.to("cuda")
    load_time = time.time() - t0
    print(f"Pipeline loaded in {load_time:.1f}s")

    mem_after_load = torch.cuda.max_memory_allocated() / 1e9
    print(f"VRAM after load: {mem_after_load:.1f} GB")

    for obj_name, obj_desc in test_objects:
        prompt = T2I_PROMPT_TEMPLATE.format(obj=obj_desc)
        print(f"\n--- Generating: {obj_name} ---")

        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        gen_kwargs = dict(
            prompt=prompt,
            height=resolution,
            width=resolution,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=torch.Generator("cuda").manual_seed(42),
        )
        if guidance_scale > 0:
            gen_kwargs["negative_prompt"] = T2I_NEGATIVE_PROMPT
        image = pipe(**gen_kwargs).images[0]
        gen_time = time.time() - t1

        out_path = f"/data/t2i_output/{short_name}_{obj_name}_{resolution}.png"
        image.save(out_path)
        file_size = os.path.getsize(out_path)
        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        print(f"  Time: {gen_time:.1f}s")
        print(f"  Size: {image.size}")
        print(f"  File: {file_size / 1024:.0f} KB")
        print(f"  Peak VRAM: {peak_mem:.1f} GB")

    del pipe
    torch.cuda.empty_cache()
    import gc; gc.collect()


# --- Test 1: SDXL-Turbo (fastest, 1-4 steps) ---
print("\n" + "=" * 60)
print("TEST 1: SDXL-Turbo (stabilityai/sdxl-turbo)")
print("=" * 60)
try:
    test_model(
        "stabilityai/sdxl-turbo",
        "sdxl_turbo",
        num_steps=4,
        guidance_scale=0.0,
    )
    print("\n✅ SDXL-Turbo: PASS")
except Exception as e:
    print(f"\n❌ SDXL-Turbo: FAIL — {e}")

# --- Test 2: SDXL Base (higher quality, CFG≤5, 768px to avoid collapse) ---
print("\n" + "=" * 60)
print("TEST 2: SDXL Base (stabilityai/stable-diffusion-xl-base-1.0) — CFG=4.5, 768px")
print("=" * 60)
try:
    test_model(
        "stabilityai/stable-diffusion-xl-base-1.0",
        "sdxl_base",
        num_steps=25,
        guidance_scale=4.5,
        resolution=768,
    )
    print("\n✅ SDXL Base: PASS")
except Exception as e:
    print(f"\n❌ SDXL Base: FAIL — {e}")

print("\n=== All tests complete ===")
print("Output dir: /data/t2i_output/")
os.system("ls -la /data/t2i_output/")
