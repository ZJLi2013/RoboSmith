"""RS-9: E2E comparison — SDXL-Turbo vs SDXL-Base → Hunyuan3D-2.1 → URDF.

Runs full pipeline twice with the same prompt to compare:
  A) SDXL-Turbo  (guidance=0.0, 512px, 4 steps)
  B) SDXL-Base   (guidance=4.5, 768px, 25 steps, + negative prompt)

Both reference images → Hunyuan3D-2.1 shape gen → mesh_to_urdf → compare.
"""
import gc
import os
import sys
import time

import torch

print(f"PyTorch: {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

HUNYUAN_REPO = "/data/Hunyuan3D-2.1"
OUTPUT_ROOT = "/data/e2e_compare"

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

PROMPT_TEXT = "red ceramic mug"

MODELS = [
    {
        "name": "turbo",
        "model_id": "stabilityai/sdxl-turbo",
        "guidance_scale": 0.0,
        "num_steps": 4,
        "resolution": 512,
        "use_negative": False,
    },
    {
        "name": "base",
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "guidance_scale": 4.5,
        "num_steps": 25,
        "resolution": 768,
        "use_negative": True,
    },
]

results = {}

for cfg in MODELS:
    tag = cfg["name"]
    out_dir = os.path.join(OUTPUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)

    # ===== Stage 1: T2I =====
    print(f"\n{'='*60}")
    print(f"[{tag.upper()}] STAGE 1: Text → Image ({cfg['model_id'].split('/')[-1]}, {cfg['resolution']}px)")
    print(f"{'='*60}")

    from diffusers import AutoPipelineForText2Image

    t0 = time.time()
    t2i_pipe = AutoPipelineForText2Image.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.float16,
        variant="fp16",
    )
    t2i_pipe = t2i_pipe.to("cuda")
    print(f"T2I loaded in {time.time()-t0:.1f}s")

    prompt = T2I_PROMPT_TEMPLATE.format(obj=PROMPT_TEXT)
    print(f"Prompt ({len(prompt.split())} words): {prompt[:80]}...")

    gen_kwargs = dict(
        prompt=prompt,
        height=cfg["resolution"],
        width=cfg["resolution"],
        guidance_scale=cfg["guidance_scale"],
        num_inference_steps=cfg["num_steps"],
        generator=torch.Generator("cuda").manual_seed(42),
    )
    if cfg["use_negative"]:
        gen_kwargs["negative_prompt"] = T2I_NEGATIVE_PROMPT

    t1 = time.time()
    image = t2i_pipe(**gen_kwargs).images[0]
    t2i_time = time.time() - t1

    ref_path = os.path.join(out_dir, "reference.png")
    image.save(ref_path)
    print(f"T2I: {t2i_time:.1f}s → {ref_path} ({os.path.getsize(ref_path)/1024:.0f} KB)")

    del t2i_pipe
    torch.cuda.empty_cache()
    gc.collect()

    # ===== Stage 2: Hunyuan3D-2.1 =====
    print(f"\n[{tag.upper()}] STAGE 2: Image → 3D (Hunyuan3D-2.1)")

    if "hy3dshape" not in sys.modules:
        sys.path.insert(0, os.path.join(HUNYUAN_REPO, "hy3dshape"))
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    t2 = time.time()
    shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2.1",
    )
    print(f"Shape pipe loaded in {time.time()-t2:.1f}s")

    from PIL import Image as PILImage
    ref_image = PILImage.open(ref_path).convert("RGBA")

    t3 = time.time()
    mesh_output = shape_pipe(image=ref_image)
    shape_time = time.time() - t3
    print(f"Shape gen: {shape_time:.1f}s")

    glb_path = os.path.join(out_dir, "mesh.glb")
    if isinstance(mesh_output, list):
        mesh_output[0].export(glb_path)
    elif hasattr(mesh_output, "export"):
        mesh_output.export(glb_path)
    else:
        mesh_output.meshes[0].export(glb_path)
    print(f"GLB: {glb_path} ({os.path.getsize(glb_path)/1e6:.1f} MB)")

    del shape_pipe
    torch.cuda.empty_cache()
    gc.collect()

    # ===== Stage 3: mesh → URDF =====
    print(f"\n[{tag.upper()}] STAGE 3: Mesh → URDF")

    import numpy as np
    import trimesh

    t4 = time.time()
    mesh = trimesh.load(glb_path, force="mesh")
    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, watertight={mesh.is_watertight}")

    target_size_m = 0.12
    current_size = mesh.bounding_box.extents.max()
    if current_size > 0:
        mesh.apply_scale(target_size_m / current_size)
    mesh.apply_translation(-mesh.centroid)

    visual_path = os.path.join(out_dir, "visual.obj")
    mesh.export(visual_path)

    collision = mesh.convex_hull
    collision_path = os.path.join(out_dir, "collision.obj")
    collision.export(collision_path)

    density = 800.0
    volume = mesh.volume if mesh.is_watertight else mesh.convex_hull.volume
    mass = density * volume
    inertia = mesh.moment_inertia if mesh.is_watertight else np.eye(3) * mass * 0.01

    urdf_path = os.path.join(out_dir, "model.urdf")
    urdf_content = f"""<?xml version="1.0" ?>
<robot name="generated_asset_{tag}">
  <link name="base_link">
    <visual>
      <geometry><mesh filename="visual.obj"/></geometry>
    </visual>
    <collision>
      <geometry><mesh filename="collision.obj"/></geometry>
    </collision>
    <inertial>
      <mass value="{mass:.4f}"/>
      <inertia ixx="{inertia[0,0]:.6f}" ixy="0" ixz="0"
               iyy="{inertia[1,1]:.6f}" iyz="0"
               izz="{inertia[2,2]:.6f}"/>
    </inertial>
  </link>
</robot>
"""
    with open(urdf_path, "w") as f:
        f.write(urdf_content)

    convert_time = time.time() - t4
    total = t2i_time + shape_time + convert_time

    results[tag] = {
        "t2i_time": t2i_time,
        "shape_time": shape_time,
        "convert_time": convert_time,
        "total": total,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "watertight": mesh.is_watertight,
        "mass": mass,
    }

    print(f"\n[{tag.upper()}] Done: {total:.1f}s total, {len(mesh.vertices)} verts")

# ===== Summary =====
print(f"\n{'='*60}")
print("COMPARISON SUMMARY")
print(f"{'='*60}")
print(f"{'Metric':<25} {'Turbo (512)':<20} {'Base (768)':<20}")
print(f"{'-'*65}")
for metric in ["t2i_time", "shape_time", "convert_time", "total", "vertices", "faces", "watertight", "mass"]:
    tv = results.get("turbo", {}).get(metric, "N/A")
    bv = results.get("base", {}).get(metric, "N/A")
    if isinstance(tv, float):
        tv = f"{tv:.2f}"
        bv = f"{bv:.2f}"
    print(f"{metric:<25} {str(tv):<20} {str(bv):<20}")

print(f"\nResults in: {OUTPUT_ROOT}/turbo/ and {OUTPUT_ROOT}/base/")
os.system(f"ls -la {OUTPUT_ROOT}/turbo/ {OUTPUT_ROOT}/base/")
