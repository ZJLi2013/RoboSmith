"""Render all object assets on one tabletop for frame/scale QA.

This script is intentionally simple: it lays every non-support asset on a fixed
grid using the canonical upright pose, renders overview images, and writes a JSON
manifest so the grid can be audited asset by asset.

Usage:
  python scripts/assets/render_asset_table.py --cpu
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import trimesh.transformations as tf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CANDIDATE_COLS = 6
GRID_SPACING_M = 0.14
TABLE_SURFACE_Z = 0.75 + 0.05 / 2.0

from robotsmith.assets.library import AssetLibrary
from robotsmith.assets.audit import SUPPORT_ASSETS
from robotsmith.scenes.backend import _height_for_quat, _resolve_upright_pose
from robotsmith.scenes.genesis_loader import _quat_wxyz_to_xyzw


def ensure_display() -> None:
    """Start Xvfb on headless Linux hosts when available."""
    if os.environ.get("DISPLAY"):
        return
    xvfb_path = shutil.which("Xvfb")
    if xvfb_path is None:
        return
    subprocess.Popen(
        [xvfb_path, ":99", "-screen", "0", "1280x1024x24", "-ac", "+extension", "GLX"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = ":99"
    time.sleep(2)
    print("[display] Xvfb started on :99")


def render_cam(cam) -> np.ndarray:
    rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
    arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.array(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.uint8)


def save_png(path: Path, arr: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(arr).save(str(path))
    print(f"saved {path} ({arr.shape[1]}x{arr.shape[0]})")


def project_points(
    points: list[list[float]],
    *,
    cam_pos: tuple[float, float, float],
    cam_target: tuple[float, float, float],
    fov_deg: float,
    image_size: tuple[int, int],
) -> list[tuple[int, int]]:
    """Project world-space points into the oblique camera image."""
    width, height = image_size
    eye = np.array(cam_pos, dtype=float)
    target = np.array(cam_target, dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    focal = (height / 2.0) / math.tan(math.radians(fov_deg) / 2.0)

    pixels = []
    for point in points:
        rel = np.array(point, dtype=float) - eye
        z = float(np.dot(rel, forward))
        x = float(np.dot(rel, right))
        y = float(np.dot(rel, up))
        pixels.append((
            int(width / 2.0 + focal * x / z),
            int(height / 2.0 - focal * y / z),
        ))
    return pixels


def draw_candidate_indices(
    arr: np.ndarray,
    label_positions: list[tuple[int, int]],
    labels: list[str] | None = None,
) -> np.ndarray:
    """Overlay candidate indices at their projected table-grid locations."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(arr.copy())
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    if labels is None:
        labels = [str(idx) for idx in range(len(label_positions))]
    for label, (x, y) in zip(labels, label_positions):
        draw.rectangle([x - 14, y - 14, x + 34, y + 20], fill=(255, 255, 255), outline=(0, 0, 0))
        draw.text((x - 6, y - 12), label, fill=(0, 0, 0), font=font)
    return np.array(img)


def object_assets(library: AssetLibrary):
    return sorted(
        [asset for asset in library.list_all() if asset.name not in SUPPORT_ASSETS],
        key=lambda asset: asset.name,
    )


def grid_positions(count: int, cols: int, center: tuple[float, float], spacing: float):
    rows = math.ceil(count / cols)
    x0 = center[0] - (cols - 1) * spacing / 2.0
    y0 = center[1] + (rows - 1) * spacing / 2.0
    for idx in range(count):
        row, col = divmod(idx, cols)
        yield [x0 + col * spacing, y0 - row * spacing]


def orientation_candidates() -> list[list[float]]:
    """Unique 90-degree right-handed rotations, represented as wxyz quats."""
    candidates: list[list[float]] = []
    seen: set[tuple[float, ...]] = set()
    quarter_turns = (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)
    for rx in quarter_turns:
        for ry in quarter_turns:
            for rz in quarter_turns:
                quat = np.array(tf.quaternion_from_euler(rx, ry, rz, axes="sxyz"))
                quat = quat / np.linalg.norm(quat)
                first = next((v for v in quat if abs(v) > 1e-9), 1.0)
                if first < 0:
                    quat *= -1.0
                key = tuple(round(float(v), 6) for v in quat)
                if key not in seen:
                    seen.add(key)
                    candidates.append(list(key))
    return candidates


def apply_upright_candidates(library: AssetLibrary, asset_name: str, indices: list[int]) -> None:
    """Write selected visual QA candidates into metadata.task_poses."""
    asset = library.get(asset_name)
    if asset is None:
        raise ValueError(f"Unknown asset: {asset_name}")
    candidates = orientation_candidates()
    if not indices:
        raise ValueError("at least one upright candidate index is required")
    for index in indices:
        if index < 0 or index >= len(candidates):
            raise ValueError(f"candidate index must be 0..{len(candidates) - 1}, got {index}")
    meta = asset.metadata
    meta.task_poses = dict(meta.task_poses or {})
    candidate_entries = [
        {
            "quat_object": candidates[index],
            "source": "manual_visual_qa",
            "candidate_index": index,
            "verified": True,
        }
        for index in indices
    ]
    primary = dict(candidate_entries[0])
    primary["candidate_indices"] = indices
    primary["description"] = "Primary upright pose; see upright_candidates for all visually valid poses."
    meta.task_poses["upright"] = primary
    meta.task_poses["upright_candidates"] = candidate_entries
    (asset.root_dir / "metadata.json").write_text(
        json.dumps(asdict(meta), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"updated {asset.root_dir / 'metadata.json'} with upright candidates {indices}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render all object assets on one table")
    parser.add_argument("--assets-root", default=None)
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output directory. Defaults to output/asset_table_overview for all "
            "assets, or <asset_dir>/qa/render_asset_table/<mode> for --asset."
        ),
    )
    parser.add_argument("--asset", default=None,
                        help="Render or annotate one exact asset name")
    parser.add_argument("--candidate-poses", action="store_true",
                        help="Render 24 right-angle orientation candidates for --asset")
    parser.add_argument("--upright-candidates", action="store_true",
                        help="Render metadata.task_poses.upright_candidates for --asset")
    parser.add_argument("--apply-upright-index", type=int, default=None,
                        help="Write one candidate index to metadata.task_poses.upright")
    parser.add_argument("--apply-upright-indices", type=int, nargs="+", default=None,
                        help="Write multiple valid upright candidate indices to metadata.task_poses")
    parser.add_argument("--sample-upright-candidates", type=int, default=0,
                        help="Render 1..N random valid upright candidates per asset")
    parser.add_argument("--include-franka", action="store_true",
                        help="Add a Franka Panda at home pose on the table")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    assets_root = Path(args.assets_root) if args.assets_root else REPO_ROOT / "assets"
    library = AssetLibrary(assets_root)
    if args.apply_upright_index is not None or args.apply_upright_indices is not None:
        if not args.asset:
            raise ValueError("--apply-upright-index/--apply-upright-indices requires --asset")
        indices = (
            args.apply_upright_indices
            if args.apply_upright_indices is not None
            else [args.apply_upright_index]
        )
        apply_upright_candidates(library, args.asset, indices)
        return

    ensure_display()

    import genesis as gs  # type: ignore[import-not-found]

    gs.init(backend=(gs.cpu if args.cpu else gs.gpu), logging_level="warning")

    if args.asset:
        asset = library.get(args.asset)
        if asset is None:
            raise ValueError(f"Unknown asset: {args.asset}")
        assets = [asset]
    else:
        assets = object_assets(library)
    if not assets:
        raise RuntimeError(f"No object assets found under {assets_root}")

    if args.out:
        out_dir = Path(args.out)
    elif args.asset:
        if args.candidate_poses:
            mode_name = "candidate_poses"
        elif args.upright_candidates:
            mode_name = "upright_candidates"
        elif args.sample_upright_candidates:
            mode_name = "sample_upright_candidates"
        else:
            mode_name = "overview"
        out_dir = assets[0].root_dir / "qa" / "render_asset_table" / mode_name
    else:
        out_dir = REPO_ROOT / "output" / "asset_table_overview"
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 30, substeps=4),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_joint_limit=True,
            box_box_detection=False,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())

    table = library.get("table_simple")
    if table is None:
        raise RuntimeError("table_simple asset is required for the overview render")
    scene.add_entity(
        gs.morphs.URDF(file=str(table.urdf_path), pos=(0.0, 0.0, 0.0), fixed=True)
    )
    franka = None
    if args.include_franka:
        franka = scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
                pos=(-0.48, 0.0, TABLE_SURFACE_Z),
            )
        )

    rng = __import__("random").Random(args.seed)
    manifest = []
    if args.candidate_poses:
        if not args.asset:
            raise ValueError("--candidate-poses requires --asset")
        render_items = [
            (assets[0], i, quat)
            for i, quat in enumerate(orientation_candidates())
        ]
    elif args.upright_candidates:
        if not args.asset:
            raise ValueError("--upright-candidates requires --asset")
        task_poses = assets[0].metadata.task_poses or {}
        candidates_meta = task_poses.get("upright_candidates") or []
        if not candidates_meta and isinstance(task_poses.get("upright"), dict):
            candidates_meta = [task_poses["upright"]]
        render_items = [
            (
                assets[0],
                candidate.get("candidate_index"),
                list(candidate.get("quat_object") or candidate.get("quat") or candidate.get("quaternion")),
            )
            for candidate in candidates_meta
            if isinstance(candidate, dict)
            and (candidate.get("quat_object") or candidate.get("quat") or candidate.get("quaternion")) is not None
        ]
        if not render_items:
            raise ValueError(f"{args.asset} has no upright candidates in metadata")
    elif args.sample_upright_candidates:
        render_items = []
        for asset in assets:
            task_poses = asset.metadata.task_poses or {}
            candidates_meta = task_poses.get("upright_candidates") or []
            if not candidates_meta and isinstance(task_poses.get("upright"), dict):
                candidates_meta = [task_poses["upright"]]
            valid = [
                (
                    candidate.get("candidate_index"),
                    candidate.get("quat_object") or candidate.get("quat") or candidate.get("quaternion"),
                )
                for candidate in candidates_meta
                if isinstance(candidate, dict)
            ]
            valid = [(index, quat) for index, quat in valid if quat is not None]
            if not valid:
                valid = [(None, [1.0, 0.0, 0.0, 0.0])]
            sample_count = rng.randint(1, min(args.sample_upright_candidates, len(valid)))
            for index, quat in rng.sample(valid, sample_count):
                render_items.append((asset, index, list(quat)))
    else:
        render_items = [(asset, None, None) for asset in assets]

    for (asset, candidate_index, candidate_quat), xy in zip(
        render_items,
        grid_positions(
            len(render_items),
            CANDIDATE_COLS,
            (0.18 if args.include_franka else 0.0, 0.0),
            GRID_SPACING_M,
        ),
    ):
        scale = float(asset.metadata.metric_scale)
        if candidate_quat is None:
            base_z, quat = _resolve_upright_pose(asset, scale)
            pose_source = "upright"
        else:
            quat = candidate_quat
            base_z = _height_for_quat(asset, scale, quat) / 2.0
            pose_source = "upright_candidate"
        pos = [xy[0], xy[1], TABLE_SURFACE_Z + base_z]
        urdf_kwargs = {
            "file": str(asset.urdf_path),
            "pos": tuple(pos),
            "quat": _quat_wxyz_to_xyzw(quat),
            "default_armature": 0.0,
        }
        if scale != 1.0:
            urdf_kwargs["scale"] = scale
        scene.add_entity(
            morph=gs.morphs.URDF(**urdf_kwargs),
            material=gs.materials.Rigid(friction=asset.metadata.friction),
        )
        manifest.append({
            "asset": asset.name,
            "position": [round(v, 5) for v in pos],
            "quat_wxyz": [round(float(v), 6) for v in quat],
            "metric_scale": scale,
            "pose_source": pose_source,
        })
        if candidate_index is not None:
            manifest[-1]["candidate_index"] = candidate_index
            if args.candidate_poses:
                row, col = divmod(candidate_index, CANDIDATE_COLS)
                manifest[-1]["row"] = row
                manifest[-1]["col"] = col

    rows = math.ceil(len(render_items) / CANDIDATE_COLS)
    grid_width = max(1, CANDIDATE_COLS - 1) * GRID_SPACING_M
    cam_target = (0.05 if args.include_franka else 0.0, 0.0, TABLE_SURFACE_Z + (0.16 if args.include_franka else 0.05))
    cam_pos = (
        0.95 + grid_width * 0.4 if args.include_franka else 0.65 + grid_width * 0.4,
        -1.15 if args.include_franka else -0.85,
        1.75 if args.include_franka else 1.35,
    )
    cam_fov = 55 if args.include_franka else 45
    cam_res = (1280, 900)
    cam_oblique = scene.add_camera(
        res=cam_res,
        pos=cam_pos,
        lookat=cam_target,
        fov=cam_fov,
        GUI=False,
    )

    scene.build()
    if franka is not None:
        from robotsmith.sim.franka import HOME_QPOS, JOINT_NAMES

        motors_dof = [franka.get_joint(joint).dofs_idx_local[0] for joint in JOINT_NAMES]
        franka.set_dofs_position(HOME_QPOS, motors_dof)
    for _ in range(10):
        scene.step()

    img_oblique = render_cam(cam_oblique)
    if any(item.get("candidate_index") is not None for item in manifest):
        label_points = [
            [item["position"][0], item["position"][1], TABLE_SURFACE_Z + 0.18]
            for item in manifest
        ]
        label_pixels = project_points(
            label_points,
            cam_pos=cam_pos,
            cam_target=cam_target,
            fov_deg=cam_fov,
            image_size=cam_res,
        )
        for item, pixel in zip(manifest, label_pixels):
            item["label_pixel"] = list(pixel)
        img_oblique = draw_candidate_indices(
            img_oblique,
            label_pixels,
            [str(item["candidate_index"]) for item in manifest],
        )
    save_png(out_dir / "asset_table_oblique.png", img_oblique)
    (out_dir / "asset_table_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved {out_dir / 'asset_table_manifest.json'}")


if __name__ == "__main__":
    main()
