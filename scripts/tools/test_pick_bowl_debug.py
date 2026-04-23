"""Reproduce exact collect_data.py flow for pick_bowl, render after reset.

This will confirm whether bowl is flipped after SimEnv.build + reset.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from robotsmith.gen.sim_env import ensure_display, SimEnv, render_cam
from robotsmith.gen.franka import to_numpy
from robotsmith.scenes.presets import SCENE_PRESETS
from robotsmith.tasks.presets import TASK_PRESETS


def main():
    ensure_display()

    task_spec = TASK_PRESETS["pick_bowl"]
    scene_config = SCENE_PRESETS[task_spec.scene]

    env = SimEnv.build(scene_config, fps=30, cpu=False, use_videos=False)

    print(f"[debug] table_surface_z = {env.table_surface_z}")
    for name, po in env.placed_map.items():
        print(f"[debug] placed '{name}': pos={po.position}, quat={po.quaternion}, "
              f"object_height_m={po.object_height_m}")

    for name, ent in env.entity_map.items():
        pos = to_numpy(ent.get_pos())
        quat = to_numpy(ent.get_quat())
        print(f"[debug] BEFORE reset '{name}': pos={pos}, quat={quat}")

    # Save screenshot before reset
    img_before = render_cam(env.cam_up)
    from PIL import Image
    Image.fromarray(img_before).save("/tmp/pick_bowl_before_reset.png")
    print("[debug] saved /tmp/pick_bowl_before_reset.png")

    # Reset with a fixed position
    env.reset({"bowl_02_0": (0.5, 0.0)})

    for name, ent in env.entity_map.items():
        pos = to_numpy(ent.get_pos())
        quat = to_numpy(ent.get_quat())
        print(f"[debug] AFTER reset '{name}': pos={pos}, quat={quat}")

    img_after = render_cam(env.cam_up)
    Image.fromarray(img_after).save("/tmp/pick_bowl_after_reset.png")
    print("[debug] saved /tmp/pick_bowl_after_reset.png")


if __name__ == "__main__":
    main()
