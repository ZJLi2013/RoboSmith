"""Render bowl_02 with several candidate quats and save screenshots.

Usage: python scripts/tools/test_bowl_render.py
Output: /tmp/bowl_quat_*.png
"""
import numpy as np


def main():
    import genesis as gs
    gs.init(backend=gs.gpu, logging_level="warning")

    candidates = {
        "identity":    [1.0, 0.0, 0.0, 0.0],
        "x_pos90":     [0.707107, 0.707107, 0.0, 0.0],     # +90 around X
        "x_neg90":     [0.707107, -0.707107, 0.0, 0.0],     # -90 around X
        "z_pos90":     [0.707107, 0.0, 0.0, 0.707107],      # +90 around Z
        "y_pos90":     [0.707107, 0.0, 0.707107, 0.0],      # +90 around Y
        "x_pos180":    [0.0, 1.0, 0.0, 0.0],                # 180 around X
    }

    for label, quat in candidates.items():
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1/30, substeps=4),
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

        scene.add_entity(
            morph=gs.morphs.URDF(
                file="assets/objects/bowl_02/model.urdf",
                pos=(0.5, 0.0, bowl_z),
                quat=tuple(quat),
                scale=0.5,
                fixed=True,
            ),
            material=gs.materials.Rigid(friction=1.5),
        )

        cam = scene.add_camera(
            res=(640, 480),
            pos=(0.5, -0.5, 1.2),
            lookat=(0.5, 0.0, 0.8),
            fov=45,
        )

        scene.build()
        scene.step()

        cam.start_recording()
        result = cam.render(rgb=True)

        if isinstance(result, tuple):
            img = result[0]
        else:
            img = result

        if hasattr(img, 'cpu'):
            img = img.cpu().numpy()
        if img.ndim == 4:
            img = img[0]
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        from PIL import Image
        pil_img = Image.fromarray(img)
        out_path = f"/tmp/bowl_quat_{label}.png"
        pil_img.save(out_path)
        print(f"[{label:12s}] saved -> {out_path}  (quat={[round(v,4) for v in quat]})")


if __name__ == "__main__":
    main()
