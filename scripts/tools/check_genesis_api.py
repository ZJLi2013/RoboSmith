"""Check Genesis set_pos/set_quat behavior: relative vs absolute."""
import genesis as gs
import torch
import inspect

gs.init(backend=gs.gpu, logging_level="error")
scene = gs.Scene(show_viewer=False,
                 sim_options=gs.options.SimOptions(dt=1/30, substeps=4),
                 rigid_options=gs.options.RigidOptions(enable_collision=True))
scene.add_entity(gs.morphs.Plane())

box = scene.add_entity(
    gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.5, 0.0, 1.0)),
    material=gs.materials.Rigid(friction=1.0),
)
scene.build()

print("=== set_pos signature ===")
print(inspect.signature(box.set_pos))
print("=== set_quat signature ===")
print(inspect.signature(box.set_quat))

# Test: what does set_pos(absolute, relative=False) do?
p0 = box.get_pos().cpu().numpy().flatten()
print(f"\ninitial pos: {p0}")

box.set_pos(torch.tensor([[0.2, 0.3, 0.5]]), zero_velocity=True, relative=False)
for _ in range(5):
    scene.step()
p1 = box.get_pos().cpu().numpy().flatten()
print(f"after set_pos([0.2,0.3,0.5], relative=False): {p1}")

box.set_pos(torch.tensor([[0.2, 0.3, 0.5]]), zero_velocity=True, relative=True)
for _ in range(5):
    scene.step()
p2 = box.get_pos().cpu().numpy().flatten()
print(f"after set_pos([0.2,0.3,0.5], relative=True): {p2}")
print("  (expected: initial + [0.2,0.3,0.5] if relative)")
