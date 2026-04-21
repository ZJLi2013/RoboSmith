"""Quick check: Genesis link API for EE pose retrieval."""
import genesis as gs
import numpy as np

gs.init(backend=gs.gpu, logging_level="warning")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=1/30, substeps=4),
    rigid_options=gs.options.RigidOptions(enable_collision=True, enable_joint_limit=True),
    show_viewer=False,
)
scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
scene.build()

ee = franka.get_link("hand")

# check available methods
link_methods = [m for m in dir(ee) if not m.startswith("_")]
print("Link methods:", link_methods)

pos = ee.get_pos()
print(f"pos: {pos}, shape: {pos.shape}")

quat = ee.get_quat()
print(f"quat: {quat}, shape: {quat.shape}")

# check inverse_kinematics signature
import inspect
sig = inspect.signature(franka.inverse_kinematics)
print(f"IK signature: {sig}")
