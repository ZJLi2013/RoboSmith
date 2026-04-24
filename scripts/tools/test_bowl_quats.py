"""Test multiple quaternions to find the correct opening-up orientation."""
import numpy as np
from PIL import Image


QUATS_WXYZ = {
    "metadata": [0.005753, -0.006157, -0.730616, -0.682736],
    "identity": [1, 0, 0, 0],
    "rot_x_neg90": [0.7071, -0.7071, 0, 0],  # Y->Z  (opening toward +Z = up)
    "rot_x_pos90": [0.7071, 0.7071, 0, 0],   # Y->-Z (opening toward -Z = down)
    "rot_z_neg90": [0.7071, 0, 0, -0.7071],
    "flip_180_x": [0, 1, 0, 0],
}


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

    bowls = {}
    x_positions = [0.3, 0.5, 0.7, 0.3, 0.5, 0.7]
    y_positions = [-0.15, -0.15, -0.15, 0.15, 0.15, 0.15]

    for i, (name, q_wxyz) in enumerate(QUATS_WXYZ.items()):
        w, x, y, z = q_wxyz
        q_xyzw = (x, y, z, w)
        bowls[name] = scene.add_entity(
            morph=gs.morphs.URDF(
                file="assets/objects/bowl_02/model.urdf",
                pos=(x_positions[i], y_positions[i], table_z + 0.10),
                quat=q_xyzw,
                scale=0.5,
            ),
            material=gs.materials.Rigid(friction=1.5),
        )

    cam = scene.add_camera(
        res=(960, 720),
        pos=(0.5, 0.7, table_z + 0.6),
        lookat=(0.5, 0.0, table_z),
        fov=55, GUI=False,
    )

    cam_side = scene.add_camera(
        res=(960, 720),
        pos=(1.1, 0.0, table_z + 0.15),
        lookat=(0.5, 0.0, table_z + 0.05),
        fov=55, GUI=False,
    )

    scene.build()

    for _ in range(90):
        scene.step()

    for name, bowl in bowls.items():
        pos = bowl.get_pos().cpu().numpy().flatten()
        quat = bowl.get_quat().cpu().numpy().flatten()
        print(f"[{name:16s}] z={pos[2]:.4f} quat={[round(float(v), 4) for v in quat]}")

    rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
    arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.array(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    img = Image.fromarray(arr.astype(np.uint8))

    # label the positions
    names = list(QUATS_WXYZ.keys())
    print("\nLayout (from cam perspective, left to right, back to front):")
    print(f"  Back row:  {names[0]:16s}  {names[1]:16s}  {names[2]:16s}")
    print(f"  Front row: {names[3]:16s}  {names[4]:16s}  {names[5]:16s}")

    img.save("outputs/bowl_quats_top.png")
    print("saved outputs/bowl_quats_top.png")

    rgb2, _, _, _ = cam_side.render(rgb=True, depth=False, segmentation=False, normal=False)
    arr2 = rgb2.cpu().numpy() if hasattr(rgb2, "cpu") else np.array(rgb2)
    if arr2.ndim == 4:
        arr2 = arr2[0]
    Image.fromarray(arr2.astype(np.uint8)).save("outputs/bowl_quats_side.png")
    print("saved outputs/bowl_quats_side.png")


if __name__ == "__main__":
    main()
