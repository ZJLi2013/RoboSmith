"""E3: Spawn with metadata_xyzw (opening-up), then reset with wxyz vs xyzw.

Isolates whether set_quat after a metadata-spawned bowl preserves opening-up.

Phase A: After spawn + settle (should be opening-up, same as test_bowl_quats)
Phase B: set_quat(metadata_wxyz) + settle
Phase C: set_quat(metadata_xyzw) + settle

If Phase B = opening-up: set_quat expects wxyz, and bug is elsewhere
If Phase C = opening-up: set_quat expects xyzw, and sim_env.reset needs conversion
If both opening-up: the problem is not in set_quat at all
If both upside-down: settle dynamics are flipping the bowl
"""
import numpy as np
from PIL import Image


META_QUAT_WXYZ = [0.005753, -0.006157, -0.730616, -0.682736]


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
    z_offset = 0.093152
    bowl_z = table_z + z_offset * 0.5  # scaled

    spawn_xyzw = to_xyzw(META_QUAT_WXYZ)
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

    # --- Phase A: after spawn with metadata_xyzw ---
    print("[Phase A] After URDF spawn with metadata_xyzw + 60 steps settle")
    for _ in range(60):
        scene.step()
    q_a = bowl.get_quat().cpu().numpy().flatten()
    p_a = bowl.get_pos().cpu().numpy().flatten()
    print(f"  pos_z={p_a[2]:.4f}  quat={[round(float(v), 4) for v in q_a]}")
    render_save(cam_side, "outputs/e3_a_spawn.png")

    # --- Phase B: set_quat(metadata as wxyz) ---
    print("[Phase B] set_quat(metadata_wxyz) + 60 steps settle")
    bowl.set_pos(torch.tensor([0.5, 0.0, bowl_z], dtype=torch.float32, device=gs.device).unsqueeze(0))
    bowl.set_quat(torch.tensor(META_QUAT_WXYZ, dtype=torch.float32, device=gs.device).unsqueeze(0))
    bowl.zero_all_dofs_velocity()
    for _ in range(60):
        scene.step()
    q_b = bowl.get_quat().cpu().numpy().flatten()
    p_b = bowl.get_pos().cpu().numpy().flatten()
    print(f"  pos_z={p_b[2]:.4f}  quat={[round(float(v), 4) for v in q_b]}")
    render_save(cam_side, "outputs/e3_b_wxyz.png")

    # --- Phase C: set_quat(metadata as xyzw) ---
    print("[Phase C] set_quat(metadata_xyzw) + 60 steps settle")
    meta_xyzw = to_xyzw(META_QUAT_WXYZ)
    bowl.set_pos(torch.tensor([0.5, 0.0, bowl_z], dtype=torch.float32, device=gs.device).unsqueeze(0))
    bowl.set_quat(torch.tensor(meta_xyzw, dtype=torch.float32, device=gs.device).unsqueeze(0))
    bowl.zero_all_dofs_velocity()
    for _ in range(60):
        scene.step()
    q_c = bowl.get_quat().cpu().numpy().flatten()
    p_c = bowl.get_pos().cpu().numpy().flatten()
    print(f"  pos_z={p_c[2]:.4f}  quat={[round(float(v), 4) for v in q_c]}")
    render_save(cam_side, "outputs/e3_c_xyzw.png")

    print("\n=== E3 Summary ===")
    print(f"A (spawn meta_xyzw): z={p_a[2]:.4f} q={[round(float(v),4) for v in q_a]}")
    print(f"B (set_quat wxyz):   z={p_b[2]:.4f} q={[round(float(v),4) for v in q_b]}")
    print(f"C (set_quat xyzw):   z={p_c[2]:.4f} q={[round(float(v),4) for v in q_c]}")


if __name__ == "__main__":
    main()
