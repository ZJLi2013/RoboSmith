"""Minimal Genesis test: find the correct quat for bowl_02 opening-up.

Spawns bowl_02 with 4 candidate quaternions, lets each settle,
then prints the final Z of the bowl's highest vertex (proxy for
"is the rim on top?").
"""
import numpy as np


def main():
    import genesis as gs
    import torch

    gs.init(backend=gs.gpu, logging_level="warning")

    candidates = {
        "identity":          [1.0, 0.0, 0.0, 0.0],
        "x+90_wxyz":         [0.707107, 0.707107, 0.0, 0.0],
        "x-90_wxyz":         [0.707107, -0.707107, 0.0, 0.0],
        "x+90_xyzw":         [0.707107, 0.0, 0.0, 0.707107],
        "x-90_xyzw":         [-0.707107, 0.0, 0.0, 0.707107],
        "old_broken":        [0.005753, -0.006157, -0.730616, -0.682736],
    }

    for label, quat in candidates.items():
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1.0 / 30, substeps=4),
            rigid_options=gs.options.RigidOptions(enable_collision=True),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())

        # Table at z=0.75
        table = scene.add_entity(
            gs.morphs.Box(size=(1.2, 0.8, 0.05), pos=(0.5, 0.0, 0.75), fixed=True),
            material=gs.materials.Rigid(friction=1.0),
        )

        table_surface_z = 0.775
        bowl_z = table_surface_z + 0.027086 * 0.5  # z_offset * scale

        ent = scene.add_entity(
            morph=gs.morphs.URDF(
                file="assets/objects/bowl_02/model.urdf",
                pos=(0.5, 0.0, bowl_z),
                quat=tuple(quat),
                scale=0.5,
            ),
            material=gs.materials.Rigid(friction=1.5),
        )
        scene.build()

        for _ in range(120):
            scene.step()

        pos = ent.get_pos().cpu().numpy().flatten()
        q_out = ent.get_quat().cpu().numpy().flatten()
        print(f"[{label:16s}] quat_in={[round(v,4) for v in quat]}  "
              f"pos_z={pos[2]:.4f}  quat_out={[round(v,4) for v in q_out]}")


if __name__ == "__main__":
    main()
