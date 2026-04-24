"""E6: Full collect_data scene (table URDF + Franka) — spawn bowl, settle, side-cam screenshot.

Tests different reset z values to find one that doesn't flip the bowl.
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

    # Same table as collect_data: URDF table
    table = scene.add_entity(
        gs.morphs.URDF(
            file="assets/furniture/table_01/model.urdf",
            pos=(0.45, 0.0, 0.0),
            fixed=True,
        ),
    )

    table_z = 0.775  # table_height(0.75) + table_thickness(0.05)/2

    # Franka (same as collect_data)
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0, 0, table_z),
        ),
    )

    spawn_xyzw = to_xyzw(META_QUAT_WXYZ)

    # Test 3 different z values
    z_values = {
        "z_original": table_z + 0.093152 * 0.5,   # 0.822 — original (too high)
        "z_low":      table_z + 0.02,               # 0.795 — modest
        "z_flush":    table_z + 0.005,               # 0.780 — nearly flush
    }

    bowls = {}
    x_pos = [0.45, 0.55, 0.65]
    for i, (label, bz) in enumerate(z_values.items()):
        bowls[label] = scene.add_entity(
            morph=gs.morphs.URDF(
                file="assets/objects/bowl_02/model.urdf",
                pos=(x_pos[i], -0.1, bz),
                quat=spawn_xyzw,
                scale=0.5,
            ),
            material=gs.materials.Rigid(friction=1.5),
        )
        print(f"[spawn] {label}: z={bz:.4f}")

    cam_side = scene.add_camera(
        res=(960, 480),
        pos=(0.55, 0.5, table_z + 0.12),
        lookat=(0.55, -0.1, table_z + 0.02),
        fov=50, GUI=False,
    )

    scene.build()

    # Initial settle (like sim_env.build)
    print("[build settle] 30 steps...")
    for step in range(30):
        scene.step()
        if step % 10 == 0 or step == 29:
            for label, bowl in bowls.items():
                q = bowl.get_quat().cpu().numpy().flatten()
                p = bowl.get_pos().cpu().numpy().flatten()
                print(f"  [{label:12s}] step {step:2d}: z={p[2]:.4f} q={[round(float(v),4) for v in q]}")

    render_save(cam_side, "outputs/e6_after_build_settle.png")

    # Now simulate reset: set_quat(wxyz) + settle
    print("\n[reset] set_quat(metadata_wxyz) + 30 steps settle...")
    for label, bowl in bowls.items():
        bz = list(z_values.values())[list(z_values.keys()).index(label)]
        bowl.set_pos(torch.tensor([x_pos[list(z_values.keys()).index(label)], -0.1, bz],
                                  dtype=torch.float32, device=gs.device).unsqueeze(0))
        bowl.set_quat(torch.tensor(META_QUAT_WXYZ, dtype=torch.float32, device=gs.device).unsqueeze(0))
        bowl.zero_all_dofs_velocity()

    for step in range(30):
        scene.step()
        if step % 10 == 0 or step == 29:
            for label, bowl in bowls.items():
                q = bowl.get_quat().cpu().numpy().flatten()
                p = bowl.get_pos().cpu().numpy().flatten()
                print(f"  [{label:12s}] step {step:2d}: z={p[2]:.4f} q={[round(float(v),4) for v in q]}")

    render_save(cam_side, "outputs/e6_after_reset_settle.png")
    print("\nCheck outputs/e6_after_build_settle.png and e6_after_reset_settle.png")


if __name__ == "__main__":
    main()
