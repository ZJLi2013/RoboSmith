"""Diagnose pick_bowl scene: verify bowl entity exists, prints key state.

Run on GPU node:
  python scripts/tools/diag_pick_bowl.py

Prints entity_map keys, placed_map, object_heights, and post-reset
positions/quats for all objects — no ad-hoc code edits needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from robotsmith.tasks import TASK_PRESETS
from robotsmith.scenes.presets import SCENE_PRESETS
from robotsmith.gen.sim_env import SimEnv, ensure_display

TASK_NAME = "pick_bowl"


def main():
    task = TASK_PRESETS[TASK_NAME]
    scene_config = SCENE_PRESETS[task.scene]

    pick_names = [s.target for s in task.skills if s.name == "pick"]
    place_names = [s.target for s in task.skills if s.name == "place"]
    print(f"[diag] task={TASK_NAME}")
    print(f"[diag] pick_names={pick_names}  place_names={place_names}")
    print(f"[diag] scene objects spec:")
    for o in scene_config.objects:
        print(f"  asset_query={o.asset_query!r} count={o.count} "
              f"name_override={o.name_override!r} scale={o.scale}")

    ensure_display()
    env = SimEnv.build(scene_config, seed=42, use_videos=False)

    print(f"\n[diag] entity_map keys : {sorted(env.entity_map.keys())}")
    print(f"[diag] placed_map keys : {sorted(env.placed_map.keys())}")
    print(f"[diag] object_heights  : {env.object_heights}")
    print(f"[diag] table_surface_z : {env.table_surface_z}")

    # Check name matching
    for name in pick_names:
        if name in env.entity_map:
            print(f"[diag] OK: '{name}' found in entity_map")
        else:
            print(f"[diag] MISSING: '{name}' NOT in entity_map!")
            print(f"[diag]   available keys: {list(env.entity_map.keys())}")

    # Reset with bowl at workspace center and inspect state
    import torch
    obj_xy = {}
    for name in env.entity_map:
        obj_xy[name] = (0.55, 0.0)

    # Pre-reset: show build-settle state
    for name, ent in env.entity_map.items():
        pos = ent.get_pos().cpu().numpy().flatten()
        quat = ent.get_quat().cpu().numpy().flatten()
        print(f"[diag] pre-reset {name}: "
              f"pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] "
              f"quat=[{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")

    # Manual reset: step-by-step to isolate the problem
    print(f"\n[diag] manual reset steps:")
    target_pos = [0.55, 0.0, env.get_initial_z("bowl")]
    ent = env.entity_map["bowl"]
    po = env.placed_map["bowl"]
    q_wxyz = po.quaternion

    print(f"[diag] target pos={target_pos}  quat={q_wxyz}")
    ent.set_pos(torch.tensor([target_pos], dtype=torch.float32, device=env._device),
                zero_velocity=True)
    ent.set_quat(torch.tensor([q_wxyz], dtype=torch.float32, device=env._device),
                 zero_velocity=True)

    # Render helpers
    from robotsmith.gen.sim_env import render_cam
    from PIL import Image
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    def snap(tag):
        img = render_cam(env.cam_up)
        path = out_dir / f"diag_{tag}_up.png"
        Image.fromarray(img).save(str(path))
        print(f"[diag] saved {path}")

    # Check IMMEDIATELY after set (no stepping)
    p_imm = ent.get_pos().cpu().numpy().flatten()
    q_imm = ent.get_quat().cpu().numpy().flatten()
    print(f"[diag] after set (0 steps): "
          f"pos=[{p_imm[0]:.4f}, {p_imm[1]:.4f}, {p_imm[2]:.4f}] "
          f"quat=[{q_imm[0]:.4f}, {q_imm[1]:.4f}, {q_imm[2]:.4f}, {q_imm[3]:.4f}]")
    snap("step0")

    # Step-by-step settle with snapshots
    checkpoints = [1, 5, 10, 30, 60]
    prev = 0
    for step in checkpoints:
        for _ in range(step - prev):
            env.scene.step()
        prev = step
        p = ent.get_pos().cpu().numpy().flatten()
        q = ent.get_quat().cpu().numpy().flatten()
        print(f"[diag] step {step:3d}: "
              f"pos=[{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}] "
              f"quat=[{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
        if step in (5, 10):
            snap(f"step{step}")


if __name__ == "__main__":
    main()
