"""RS-7: End-to-end text → T2I → Hunyuan3D-2.1 → URDF pipeline.

Full pipeline:
  1. Text prompt ("red ceramic mug")
  2. SDXL-Turbo generates 512×512 reference image
  3. Hunyuan3D-2.1 shape gen (image→3D mesh)
  4. mesh_to_urdf (trimesh → URDF + collision hull)
  5. Validate with PyBullet
"""
import gc
import os
import sys
import time

import torch

print(f"PyTorch: {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

HUNYUAN_REPO = "/data/Hunyuan3D-2.1"
OUTPUT_DIR = "/data/e2e_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# ===== Stage 1: T2I =====
print("\n" + "=" * 60)
print("STAGE 1: Text → Image (SDXL-Turbo, 512×512)")
print("=" * 60)

from diffusers import AutoPipelineForText2Image

t0 = time.time()
t2i_pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
)
t2i_pipe = t2i_pipe.to("cuda")
t2i_load = time.time() - t0
print(f"T2I pipeline loaded in {t2i_load:.1f}s")

prompt_text = "red ceramic mug"
prompt = T2I_PROMPT_TEMPLATE.format(obj=prompt_text)
print(f"Prompt: {prompt}")

t1 = time.time()
image = t2i_pipe(
    prompt,
    height=512,
    width=512,
    guidance_scale=0.0,
    num_inference_steps=4,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
t2i_time = time.time() - t1

ref_path = os.path.join(OUTPUT_DIR, "reference.png")
image.save(ref_path)
print(f"T2I time: {t2i_time:.1f}s → {ref_path} ({os.path.getsize(ref_path)/1024:.0f} KB)")

del t2i_pipe
torch.cuda.empty_cache()
gc.collect()
print("T2I pipeline unloaded, VRAM freed")

# ===== Stage 2: Hunyuan3D-2.1 shape gen =====
print("\n" + "=" * 60)
print("STAGE 2: Image → 3D Mesh (Hunyuan3D-2.1)")
print("=" * 60)

sys.path.insert(0, os.path.join(HUNYUAN_REPO, "hy3dshape"))
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

t2 = time.time()
shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2.1",
)
shape_load = time.time() - t2
print(f"Shape pipeline loaded in {shape_load:.1f}s")

from PIL import Image as PILImage
ref_image = PILImage.open(ref_path).convert("RGBA")

t3 = time.time()
mesh_list = shape_pipe(image=ref_image)
shape_time = time.time() - t3
print(f"Shape gen time: {shape_time:.1f}s")

glb_path = os.path.join(OUTPUT_DIR, "mesh.glb")
if isinstance(mesh_list, list):
    mesh_list[0].export(glb_path)
elif hasattr(mesh_list, "export"):
    mesh_list.export(glb_path)
else:
    mesh_list.meshes[0].export(glb_path)
print(f"GLB exported: {glb_path} ({os.path.getsize(glb_path)/1e6:.1f} MB)")

del shape_pipe
torch.cuda.empty_cache()
gc.collect()

# ===== Stage 3: mesh → URDF =====
print("\n" + "=" * 60)
print("STAGE 3: Mesh → URDF (trimesh + convex hull)")
print("=" * 60)

import numpy as np
import trimesh

t4 = time.time()
mesh = trimesh.load(glb_path, force="mesh")
print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
print(f"Watertight: {mesh.is_watertight}")

target_size_m = 0.12
current_size = mesh.bounding_box.extents.max()
if current_size > 0:
    scale = target_size_m / current_size
    mesh.apply_scale(scale)
mesh.apply_translation(-mesh.centroid)
print(f"Scaled to {target_size_m}m, bbox: {mesh.bounding_box.extents}")

visual_path = os.path.join(OUTPUT_DIR, "visual.obj")
mesh.export(visual_path)

collision = mesh.convex_hull
collision_path = os.path.join(OUTPUT_DIR, "collision.obj")
collision.export(collision_path)
print(f"Collision hull: {len(collision.vertices)} verts")

density = 800.0
if mesh.is_watertight:
    volume = mesh.volume
else:
    volume = mesh.convex_hull.volume
mass = density * volume
inertia = mesh.moment_inertia if mesh.is_watertight else np.eye(3) * mass * 0.01

urdf_path = os.path.join(OUTPUT_DIR, "model.urdf")
urdf_content = f"""<?xml version="1.0" ?>
<robot name="generated_asset">
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
print(f"URDF written: {urdf_path}")
print(f"Convert time: {convert_time:.1f}s")
print(f"Mass: {mass:.4f} kg")

# ===== Stage 4: PyBullet validation =====
print("\n" + "=" * 60)
print("STAGE 4: PyBullet Validation")
print("=" * 60)

try:
    import pybullet as p
    t5 = time.time()
    physics_client = p.connect(p.DIRECT)
    body_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.5], useFixedBase=False)
    for _ in range(240):
        p.stepSimulation()
    pos, _ = p.getBasePositionAndOrientation(body_id)
    p.disconnect()
    pybullet_time = time.time() - t5
    print(f"PyBullet: LOAD OK, final_z={pos[2]:.3f}, time={pybullet_time:.1f}s")
except Exception as e:
    print(f"PyBullet: SKIP — {e}")

# ===== Summary =====
total_time = t2i_time + shape_time + convert_time
print("\n" + "=" * 60)
print("PIPELINE SUMMARY")
print("=" * 60)
print(f"Input:          text = '{prompt_text}'")
print(f"T2I time:       {t2i_time:.1f}s (SDXL-Turbo, 512×512)")
print(f"Shape gen time: {shape_time:.1f}s (Hunyuan3D-2.1)")
print(f"Convert time:   {convert_time:.1f}s (trimesh → URDF)")
print(f"Total:          {total_time:.1f}s")
print(f"Output URDF:    {urdf_path}")
print(f"Vertices:       {len(mesh.vertices)}")
print(f"Mass:           {mass:.4f} kg")

os.system(f"ls -la {OUTPUT_DIR}/")
