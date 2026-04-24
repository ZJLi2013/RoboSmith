"""Check Genesis set_pos/set_quat behavior: Box vs URDF entity.

Compares absolute positioning for Box primitive vs URDF free-floating
entity to determine if Genesis handles them differently.
"""
import sys
from pathlib import Path
import genesis as gs
import torch
import inspect
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

gs.init(backend=gs.gpu, logging_level="error")

# ---------- Scene with Box + URDF bowl ----------
scene = gs.Scene(
    show_viewer=False,
    sim_options=gs.options.SimOptions(dt=1/30, substeps=4),
    rigid_options=gs.options.RigidOptions(enable_collision=True),
)
scene.add_entity(gs.morphs.Plane())

box = scene.add_entity(
    gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.5, 0.0, 1.0)),
    material=gs.materials.Rigid(friction=1.0),
)

assets_root = Path(__file__).resolve().parent.parent.parent / "assets"
bowl_urdf = str(assets_root / "objects" / "bowl_02" / "model.urdf")
bowl = scene.add_entity(
    gs.morphs.URDF(file=bowl_urdf, pos=(0.3, 0.1, 1.0), scale=0.5),
    material=gs.materials.Rigid(friction=1.0),
)

scene.build()
for _ in range(30):
    scene.step()

# ---------- Signatures ----------
print("=== set_pos signature ===")
print(f"  box:  {inspect.signature(box.set_pos)}")
print(f"  bowl: {inspect.signature(bowl.set_pos)}")
print("=== set_quat signature ===")
print(f"  box:  {inspect.signature(box.set_quat)}")
print(f"  bowl: {inspect.signature(bowl.set_quat)}")

# ---------- Test Box ----------
print("\n--- Box entity ---")
p0 = box.get_pos().cpu().numpy().flatten()
print(f"pre:  {p0}")

box.set_pos(torch.tensor([[0.2, 0.3, 0.8]]), zero_velocity=True)
for _ in range(5):
    scene.step()
p1 = box.get_pos().cpu().numpy().flatten()
print(f"set_pos([0.2,0.3,0.8]) -> {p1}")
print(f"  delta from target: {p1 - np.array([0.2, 0.3, 0.8])}")

# ---------- Test URDF bowl ----------
print("\n--- URDF bowl entity ---")
bp0 = bowl.get_pos().cpu().numpy().flatten()
print(f"pre:  {bp0}")

target = [0.55, 0.0, 0.8]
bowl.set_pos(torch.tensor([target]), zero_velocity=True)
for _ in range(5):
    scene.step()
bp1 = bowl.get_pos().cpu().numpy().flatten()
print(f"set_pos({target}) -> {bp1}")
print(f"  delta from target: {bp1 - np.array(target)}")

# Check if the delta matches the initial morph pos
morph_pos = np.array([0.3, 0.1, 1.0])
print(f"  morph init pos:    {morph_pos}")
print(f"  result - target:   {bp1 - np.array(target)}")
print(f"  result - morph:    {bp1 - morph_pos}")

# ---------- Test URDF bowl with relative=True ----------
print("\n--- URDF bowl: explicit relative=True ---")
bowl.set_pos(torch.tensor([[0.0, 0.0, 0.0]]), zero_velocity=True, relative=True)
for _ in range(5):
    scene.step()
bp2 = bowl.get_pos().cpu().numpy().flatten()
print(f"set_pos([0,0,0], relative=True) -> {bp2}")
print(f"  (should be morph init pos if relative adds to init)")

# ---------- Test set_quat ----------
print("\n--- URDF bowl set_quat ---")
q0 = bowl.get_quat().cpu().numpy().flatten()
print(f"pre quat:  {q0}")

identity_q = [1.0, 0.0, 0.0, 0.0]
bowl.set_quat(torch.tensor([identity_q]), zero_velocity=True)
for _ in range(5):
    scene.step()
q1 = bowl.get_quat().cpu().numpy().flatten()
print(f"set_quat({identity_q}) -> {q1}")
print(f"  (if absolute: should be ~identity; if relative to init: rotated)")
