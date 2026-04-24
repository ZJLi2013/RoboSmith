"""E1: Determine whether Genesis ent.set_quat() expects wxyz or xyzw.

Strategy:
  1. Spawn bowl with rot_z_neg90 via URDF(quat=xyzw) — confirmed upside-down.
  2. After settle, call set_quat with metadata quat in wxyz format, settle, screenshot.
  3. Call set_quat with metadata quat in xyzw format, settle, screenshot.

Metadata quat [0.006, -0.006, -0.731, -0.683] (wxyz) is confirmed opening-up
via URDF spawn. So whichever set_quat variant produces opening-up tells us the
convention.

Expected outcomes (one of two):
  A) set_quat expects wxyz → wxyz screenshot = opening-up, xyzw = wrong orientation
  B) set_quat expects xyzw → xyzw screenshot = opening-up, wxyz = wrong orientation
"""
import numpy as np
from PIL import Image


META_QUAT_WXYZ = [0.005753, -0.006157, -0.730616, -0.682736]

ROT_Z_NEG90_WXYZ = [0.7071, 0, 0, -0.7071]


def to_xyzw(wxyz):
    w, x, y, z = wxyz
    return [x, y, z, w]


def render_save(cam, path):
    rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
    arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.array(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    Image.fromarray(arr.astype(np.uint8)).save(path)
    print(f"  saved {path}")


def main():
    import genesis as gs
    import torch

    gs.init(backend=gs.gpu, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 30, substeps=4),
        rigid_options=gs.options.RigidOptions(enable_collision=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.Box(size=(1.2, 0.8, 0.05), pos=(0.5, 0.0, 0.75), fixed=True),
        material=gs.materials.Rigid(friction=1.0),
    )
    table_z = 0.775
    bowl_z = table_z + 0.05

    # Spawn with rot_z_neg90 (the only quat that is upside-down via URDF spawn)
    spawn_xyzw = to_xyzw(ROT_Z_NEG90_WXYZ)
    bowl = scene.add_entity(
        morph=gs.morphs.URDF(
            file="assets/objects/bowl_02/model.urdf",
            pos=(0.5, 0.0, bowl_z),
            quat=spawn_xyzw,
            scale=0.5,
        ),
        material=gs.materials.Rigid(friction=1.5),
    )

    cam_side = scene.add_camera(
        res=(640, 480),
        pos=(0.9, 0.0, table_z + 0.15),
        lookat=(0.5, 0.0, table_z + 0.03),
        fov=45, GUI=False,
    )

    scene.build()

    # --- Phase A: after spawn (should be upside-down) ---
    print("[Phase A] After URDF spawn with rot_z_neg90 (expect upside-down)")
    for _ in range(60):
        scene.step()
    q_a = bowl.get_quat().cpu().numpy().flatten()
    print(f"  quat_out = {[round(float(v), 4) for v in q_a]}")
    render_save(cam_side, "outputs/e1_a_spawn_rotz.png")

    # --- Phase B: set_quat with metadata quat as WXYZ ---
    print("[Phase B] set_quat(metadata_wxyz) — if opening-up → set_quat expects wxyz")
    meta_wxyz = META_QUAT_WXYZ
    bowl.set_pos(torch.tensor([0.5, 0.0, bowl_z], dtype=torch.float32, device=gs.device).unsqueeze(0))
    bowl.set_quat(torch.tensor(meta_wxyz, dtype=torch.float32, device=gs.device).unsqueeze(0))
    bowl.zero_all_dofs_velocity()
    for _ in range(60):
        scene.step()
    q_b = bowl.get_quat().cpu().numpy().flatten()
    print(f"  quat_out = {[round(float(v), 4) for v in q_b]}")
    render_save(cam_side, "outputs/e1_b_setquat_wxyz.png")

    # --- Phase C: set_quat with metadata quat as XYZW ---
    print("[Phase C] set_quat(metadata_xyzw) — if opening-up → set_quat expects xyzw")
    meta_xyzw = to_xyzw(META_QUAT_WXYZ)
    bowl.set_pos(torch.tensor([0.5, 0.0, bowl_z], dtype=torch.float32, device=gs.device).unsqueeze(0))
    bowl.set_quat(torch.tensor(meta_xyzw, dtype=torch.float32, device=gs.device).unsqueeze(0))
    bowl.zero_all_dofs_velocity()
    for _ in range(60):
        scene.step()
    q_c = bowl.get_quat().cpu().numpy().flatten()
    print(f"  quat_out = {[round(float(v), 4) for v in q_c]}")
    render_save(cam_side, "outputs/e1_c_setquat_xyzw.png")

    print("\n=== E1 Summary ===")
    print("Phase A (spawn rot_z_neg90): should be upside-down (control)")
    print("Phase B (set_quat wxyz):     if opening-up → set_quat expects wxyz")
    print("Phase C (set_quat xyzw):     if opening-up → set_quat expects xyzw")
    print("Compare outputs/e1_a_spawn_rotz.png, e1_b_setquat_wxyz.png, e1_c_setquat_xyzw.png")


if __name__ == "__main__":
    main()
