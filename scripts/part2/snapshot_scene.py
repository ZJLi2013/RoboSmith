"""Render scene snapshots for visual layout verification.

Builds the Genesis scene (table + random objects + Franka at home pose),
steps physics briefly to settle, then saves camera images as PNG.

Usage:
  python scripts/part2/snapshot_scene.py --seed 1 --out /output/snapshots
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import numpy as np


HOME_QPOS = np.array([0, -0.3, 0, -2.2, 0, 2.0, 0.79, 0.04, 0.04], dtype=np.float32)
JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "joint5", "joint6", "joint7",
    "finger_joint1", "finger_joint2",
]


def ensure_display():
    if os.environ.get("DISPLAY"):
        return
    xvfb = subprocess.run(["which", "Xvfb"], capture_output=True)
    if xvfb.returncode != 0:
        return
    subprocess.Popen(
        ["Xvfb", ":99", "-screen", "0", "1280x1024x24", "-ac", "+extension", "GLX"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = ":99"
    time.sleep(2)
    print("[display] Xvfb started on :99")


def render_one_seed(seed: int, scene_name: str, assets_root: str, out_dir: Path, gs, settle_steps: int):
    from robotsmith.assets.library import AssetLibrary
    from robotsmith.scenes.backend import ProgrammaticSceneBackend

    library = AssetLibrary(assets_root)
    if scene_name == "tabletop_simple":
        from robotsmith.scenes.presets.tabletop_simple import tabletop_simple
        scene_config = tabletop_simple
    else:
        raise ValueError(f"Unknown scene preset: {scene_name}")

    backend = ProgrammaticSceneBackend(seed=seed)
    resolved = backend.resolve(scene_config, library)
    print(f"\n[seed={seed}] {resolved.summary()}")

    from robotsmith.scenes.genesis_loader import load_resolved_scene
    handle = load_resolved_scene(resolved, gs_module=gs, fps=30, box_box_detection=False)
    scene = handle.scene
    franka = handle.franka

    ws = scene_config.workspace_xy
    cx = (ws[0][0] + ws[1][0]) / 2.0
    cy = (ws[0][1] + ws[1][1]) / 2.0
    table_z = scene_config.table_height + scene_config.table_size[2] / 2.0
    # Franka base is at x=0; look at midpoint between base and workspace
    scene_center_x = cx / 2.0

    cam_overview = scene.add_camera(
        res=(960, 720),
        pos=(scene_center_x + 1.0, cy + 0.8, table_z + 0.8),
        lookat=(scene_center_x, cy, table_z),
        fov=55, GUI=False,
    )
    cam_topdown = scene.add_camera(
        res=(960, 720),
        pos=(scene_center_x, cy, table_z + 1.4),
        lookat=(scene_center_x, cy, table_z),
        fov=55, GUI=False,
    )

    scene.build()

    motors_dof = [franka.get_joint(j).dofs_idx_local[0] for j in JOINT_NAMES]
    franka.set_dofs_position(HOME_QPOS, motors_dof)

    for _ in range(settle_steps):
        scene.step()

    for cam_name, cam in [("overview", cam_overview), ("topdown", cam_topdown)]:
        rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
        arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.array(rgb)
        if arr.ndim == 4:
            arr = arr[0]
        arr = arr.astype(np.uint8)

        from PIL import Image
        img = Image.fromarray(arr)
        fname = out_dir / f"seed{seed}_{cam_name}.png"
        img.save(str(fname))
        print(f"  saved {fname.name} ({arr.shape[1]}x{arr.shape[0]})")

    del scene, handle


def main():
    ap = argparse.ArgumentParser(description="Render scene snapshots for layout review")
    ap.add_argument("--scene", default="tabletop_simple")
    ap.add_argument("--assets-root", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="/output/snapshots")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--settle-steps", type=int, default=60,
                    help="Physics settle steps before rendering")
    args = ap.parse_args()

    ensure_display()

    import genesis as gs
    gs.init(backend=(gs.cpu if args.cpu else gs.gpu), logging_level="warning")

    assets_root = args.assets_root or str(
        Path(__file__).resolve().parent.parent / "assets"
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    render_one_seed(args.seed, args.scene, assets_root, out_dir, gs, args.settle_steps)

    pngs = sorted(out_dir.glob("*.png"))
    print(f"\n[done] {len(pngs)} images saved to {out_dir}")
    for p in pngs:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
