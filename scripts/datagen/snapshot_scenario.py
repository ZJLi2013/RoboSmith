"""Render static layout snapshots for an authoring-ready CAP scenario.

Builds the Genesis scene from a generated scenario's lowered SceneConfig
(table + objects + Franka at home pose), steps physics briefly to settle, then
saves overview/topdown camera images as PNG. This is the static layout-review
counterpart to ``run_generated_scenario.py`` (which records rollout videos).

Usage:
  python scripts/datagen/snapshot_scenario.py \
    --scenario output/<exp>/generated_scenario.py \
    --out output/<exp>/snapshots
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from robotsmith.scenario_runtime import materialize_candidate

HOME_QPOS = np.array([0, -0.3, 0, -2.2, 0, 2.0, 0.79, 0.04, 0.04], dtype=np.float32)
JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "joint5", "joint6", "joint7",
    "finger_joint1", "finger_joint2",
]


def _camera_views(scene_config, style: str):
    ws = scene_config.workspace_xy
    cx = (ws[0][0] + ws[1][0]) / 2.0
    cy = (ws[0][1] + ws[1][1]) / 2.0
    table_z = scene_config.table_height + scene_config.table_size[2] / 2.0
    scene_center_x = cx / 2.0

    if style == "debug":
        return {
            "overview": {
                "pos": (scene_center_x + 1.0, cy + 0.8, table_z + 0.8),
                "lookat": (scene_center_x, cy, table_z),
                "fov": 55,
            },
            "topdown": {
                "pos": (scene_center_x, cy, table_z + 1.4),
                "lookat": (scene_center_x, cy, table_z),
                "fov": 55,
            },
        }

    # Presentation framing: tighter, higher, and less dominated by the floor.
    return {
        "overview": {
            "pos": (cx + 0.72, cy + 0.52, table_z + 0.52),
            "lookat": (cx, cy, table_z + 0.08),
            "fov": 38,
        },
        "topdown": {
            "pos": (cx, cy, table_z + 1.15),
            "lookat": (cx, cy, table_z),
            "fov": 42,
        },
    }


def _postprocess_snapshot(img, style: str):
    if style == "debug":
        return img

    from PIL import ImageEnhance

    img = ImageEnhance.Brightness(img).enhance(1.08)
    img = ImageEnhance.Contrast(img).enhance(1.06)
    img = ImageEnhance.Color(img).enhance(1.05)
    return img


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


def render_scenario(
    scenario_path: Path,
    seed: int,
    assets_root: str,
    out_dir: Path,
    gs,
    settle_steps: int,
    style: str,
):
    from robotsmith.assets.library import AssetLibrary
    from robotsmith.scenes.backend import ProgrammaticSceneBackend
    from robotsmith.scenes.genesis_loader import load_resolved_scene

    package = materialize_candidate(scenario_path)
    scene_config = package.legacy_scene
    name = package.legacy_task.name

    library = AssetLibrary(assets_root)
    resolved = ProgrammaticSceneBackend(seed=seed).resolve(scene_config, library)
    print(f"\n[{name} seed={seed}] {resolved.summary()}")

    handle = load_resolved_scene(resolved, gs_module=gs, fps=30, box_box_detection=False)
    scene = handle.scene
    franka = handle.franka

    views = _camera_views(scene_config, style)
    cam_overview = scene.add_camera(res=(960, 720), GUI=False, **views["overview"])
    cam_topdown = scene.add_camera(res=(960, 720), GUI=False, **views["topdown"])

    scene.build()

    motors_dof = [franka.get_joint(j).dofs_idx_local[0] for j in JOINT_NAMES]
    franka.set_dofs_position(HOME_QPOS, motors_dof)

    for _ in range(settle_steps):
        scene.step()

    from PIL import Image
    for cam_name, cam in [("overview", cam_overview), ("topdown", cam_topdown)]:
        rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
        arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.array(rgb)
        if arr.ndim == 4:
            arr = arr[0]
        arr = arr.astype(np.uint8)

        img = _postprocess_snapshot(Image.fromarray(arr), style)
        fname = out_dir / f"{name}_seed{seed}_{cam_name}.png"
        img.save(str(fname))
        print(f"  saved {fname.name} ({arr.shape[1]}x{arr.shape[0]})")

    del scene, handle


def main():
    ap = argparse.ArgumentParser(description="Render layout snapshots for a CAP scenario")
    ap.add_argument("--scenario", type=Path, required=True,
                    help="Path to an authoring-ready generated_scenario.py")
    ap.add_argument("--assets-root", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="output/scenario_snapshots")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--settle-steps", type=int, default=60,
                    help="Physics settle steps before rendering")
    ap.add_argument(
        "--style",
        choices=["presentation", "debug"],
        default="presentation",
        help="Snapshot visual style. 'debug' preserves the old wide camera.",
    )
    args = ap.parse_args()

    ensure_display()

    import genesis as gs
    gs.init(backend=(gs.cpu if args.cpu else gs.gpu), logging_level="warning")

    assets_root = args.assets_root or str(REPO_ROOT / "assets")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    render_scenario(
        args.scenario, args.seed, assets_root, out_dir, gs,
        args.settle_steps, args.style,
    )

    pngs = sorted(out_dir.glob("*.png"))
    print(f"\n[done] {len(pngs)} images saved to {out_dir}")
    for p in pngs:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
