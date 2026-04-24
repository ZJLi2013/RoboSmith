"""E6: Full scene (table_simple + Franka) — spawn bowls at different z, settle, side-cam screenshot.

Goal: find a z value that keeps the bowl opening-up after settle.
"""
import numpy as np
from PIL import Image


META_QUAT_WXYZ = [0.005753, -0.006157, -0.730616, -0.682736]

TABLE_HEIGHT = 0.75
TABLE_THICKNESS = 0.05
TABLE_Z = TABLE_HEIGHT + TABLE_THICKNESS / 2.0  # 0.775

BOWL_SCALE = 0.5
BOWL_HEIGHT_M = 5.5 / 100.0 * BOWL_SCALE  # 0.0275


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

    table = scene.add_entity(
        gs.morphs.URDF(
            file="assets/objects/table_simple/model.urdf",
            pos=(0.3, 0.0, 0.0),
            fixed=True,
        ),
    )

    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0, 0, TABLE_Z),
        ),
    )

    spawn_xyzw = to_xyzw(META_QUAT_WXYZ)

    # Test 4 different z values
    z_configs = {
        "z_original":   TABLE_Z + 0.093152 * BOWL_SCALE,  # 0.822 — original metadata z*scale
        "z_half_h":     TABLE_Z + BOWL_HEIGHT_M / 2.0,     # 0.789 — half bowl height
        "z_low":        TABLE_Z + 0.005,                    # 0.780 — 5mm above table
        "z_quarter_h":  TABLE_Z + BOWL_HEIGHT_M / 4.0,     # 0.782 — quarter bowl height
    }

    bowls = {}
    x_positions = [0.40, 0.50, 0.60, 0.70]
    for i, (label, bz) in enumerate(z_configs.items()):
        bowls[label] = scene.add_entity(
            morph=gs.morphs.URDF(
                file="assets/objects/bowl_02/model.urdf",
                pos=(x_positions[i], -0.12, bz),
                quat=spawn_xyzw,
                scale=BOWL_SCALE,
            ),
            material=gs.materials.Rigid(friction=1.5),
        )
        print(f"[spawn] {label}: z={bz:.4f} (above_table={bz-TABLE_Z:.4f})")

    cam_side = scene.add_camera(
        res=(1280, 480),
        pos=(0.55, 0.45, TABLE_Z + 0.08),
        lookat=(0.55, -0.12, TABLE_Z + 0.02),
        fov=50, GUI=False,
    )
    cam_up = scene.add_camera(
        res=(1280, 480),
        pos=(0.55, -0.12, TABLE_Z + 0.50),
        lookat=(0.55, -0.12, TABLE_Z),
        fov=50, GUI=False,
    )

    scene.build()

    print("\n[build settle] 60 steps...")
    for step in range(60):
        scene.step()
        if step % 20 == 0 or step == 59:
            for label, bowl in bowls.items():
                q = bowl.get_quat().cpu().numpy().flatten()
                p = bowl.get_pos().cpu().numpy().flatten()
                print(f"  [{label:14s}] step {step:2d}: z={p[2]:.4f} q={[round(float(v),4) for v in q]}")

    render_save(cam_side, "outputs/e6_build_side.png")
    render_save(cam_up, "outputs/e6_build_top.png")

    # Now simulate a reset: set_pos + set_quat(wxyz) + settle
    print("\n[reset] set_pos + set_quat(metadata_wxyz) + 60 steps...")
    for i, (label, bz) in enumerate(z_configs.items()):
        bowl = bowls[label]
        bowl.set_pos(torch.tensor(
            [x_positions[i], -0.12, bz],
            dtype=torch.float32, device=gs.device).unsqueeze(0))
        bowl.set_quat(torch.tensor(
            META_QUAT_WXYZ, dtype=torch.float32, device=gs.device).unsqueeze(0))
        bowl.zero_all_dofs_velocity()

    for step in range(60):
        scene.step()
        if step % 20 == 0 or step == 59:
            for label, bowl in bowls.items():
                q = bowl.get_quat().cpu().numpy().flatten()
                p = bowl.get_pos().cpu().numpy().flatten()
                print(f"  [{label:14s}] step {step:2d}: z={p[2]:.4f} q={[round(float(v),4) for v in q]}")

    render_save(cam_side, "outputs/e6_reset_side.png")
    render_save(cam_up, "outputs/e6_reset_top.png")

    print("\nDone. Check outputs/e6_*.png")


if __name__ == "__main__":
    main()
