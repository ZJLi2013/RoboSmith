"""Minimal bowl orientation test: spawn bowl, settle, screenshot."""
import numpy as np


def main():
    import genesis as gs
    import torch
    from PIL import Image

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

    # bowl_02 stable_pose from metadata (wxyz)
    q_wxyz = [0.005753, -0.006157, -0.730616, -0.682736]
    w, x, y, z = q_wxyz
    q_xyzw = (x, y, z, w)

    z_offset = 0.093152
    bowl_z = table_z + z_offset * 0.5

    # --- Test 1: URDF spawn with xyzw (genesis_loader path) ---
    bowl = scene.add_entity(
        morph=gs.morphs.URDF(
            file="assets/objects/bowl_02/model.urdf",
            pos=(0.55, 0.0, bowl_z),
            quat=q_xyzw,
            scale=0.5,
            default_armature=0.0,
        ),
        material=gs.materials.Rigid(friction=1.5),
    )

    cam = scene.add_camera(
        res=(640, 480),
        pos=(0.55, 0.55, table_z + 0.55),
        lookat=(0.55, 0.0, table_z + 0.05),
        fov=45, GUI=False,
    )

    scene.build()

    for _ in range(60):
        scene.step()

    pos = bowl.get_pos().cpu().numpy().flatten()
    quat_out = bowl.get_quat().cpu().numpy().flatten()
    print(f"[spawn] pos_z={pos[2]:.4f}  quat_out={[round(v, 4) for v in quat_out]}")

    rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
    arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.array(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    Image.fromarray(arr.astype(np.uint8)).save("outputs/bowl_spawn.png")
    print("[spawn] saved outputs/bowl_spawn.png")

    # --- Test 2: set_quat with wxyz (sim_env.reset path, after fix) ---
    bowl.set_pos(torch.tensor(
        [0.55, 0.0, bowl_z], dtype=torch.float32, device=gs.device
    ).unsqueeze(0))
    bowl.set_quat(torch.tensor(
        q_wxyz, dtype=torch.float32, device=gs.device
    ).unsqueeze(0))

    for _ in range(60):
        scene.step()

    pos2 = bowl.get_pos().cpu().numpy().flatten()
    quat2 = bowl.get_quat().cpu().numpy().flatten()
    print(f"[reset-wxyz] pos_z={pos2[2]:.4f}  quat_out={[round(v, 4) for v in quat2]}")

    rgb2, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
    arr2 = rgb2.cpu().numpy() if hasattr(rgb2, "cpu") else np.array(rgb2)
    if arr2.ndim == 4:
        arr2 = arr2[0]
    Image.fromarray(arr2.astype(np.uint8)).save("outputs/bowl_reset_wxyz.png")
    print("[reset-wxyz] saved outputs/bowl_reset_wxyz.png")

    # --- Test 3: set_quat with xyzw (old broken reset path) ---
    bowl.set_pos(torch.tensor(
        [0.55, 0.0, bowl_z], dtype=torch.float32, device=gs.device
    ).unsqueeze(0))
    bowl.set_quat(torch.tensor(
        list(q_xyzw), dtype=torch.float32, device=gs.device
    ).unsqueeze(0))

    for _ in range(60):
        scene.step()

    pos3 = bowl.get_pos().cpu().numpy().flatten()
    quat3 = bowl.get_quat().cpu().numpy().flatten()
    print(f"[reset-xyzw] pos_z={pos3[2]:.4f}  quat_out={[round(v, 4) for v in quat3]}")

    rgb3, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
    arr3 = rgb3.cpu().numpy() if hasattr(rgb3, "cpu") else np.array(rgb3)
    if arr3.ndim == 4:
        arr3 = arr3[0]
    Image.fromarray(arr3.astype(np.uint8)).save("outputs/bowl_reset_xyzw.png")
    print("[reset-xyzw] saved outputs/bowl_reset_xyzw.png")

    print("\n=== Summary ===")
    print(f"spawn (URDF xyzw):     z={pos[2]:.4f}")
    print(f"reset (set_quat wxyz): z={pos2[2]:.4f}")
    print(f"reset (set_quat xyzw): z={pos3[2]:.4f}")
    print("Check outputs/bowl_spawn.png, bowl_reset_wxyz.png, bowl_reset_xyzw.png")


if __name__ == "__main__":
    main()
