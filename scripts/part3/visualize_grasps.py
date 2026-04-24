"""Visualize GraspGen grasp candidates on an asset mesh.

Renders: object mesh point cloud + grasp coordinate frames in viser.
Used for verifying coordinate system alignment before Genesis integration.

Usage:
  # With real GraspGen model (requires GPU + graspgen installed):
  python scripts/part3/visualize_grasps.py \
    --asset bowl_02 --scale 0.35 \
    --gripper-config /path/to/graspgen_robotiq_2f_140.yml

  # With mock model (no GPU needed, random grasps for pipeline testing):
  python scripts/part3/visualize_grasps.py \
    --asset bowl_02 --scale 0.35 --mock
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from robotsmith.assets.library import AssetLibrary
from robotsmith.grasp.pointcloud_utils import asset_to_pointcloud
from robotsmith.grasp.transforms import rotmat_to_quat_wxyz


def parse_args():
    ap = argparse.ArgumentParser(description="Visualize grasp candidates on an asset")
    ap.add_argument("--asset", required=True, help="Asset name (e.g. bowl_02)")
    ap.add_argument("--assets-root", default=None, help="Assets directory")
    ap.add_argument("--scale", type=float, default=1.0, help="Mesh scale")
    ap.add_argument("--n-points", type=int, default=4096, help="Point cloud samples")
    ap.add_argument("--object-pos", type=float, nargs=3, default=[0.5, 0.0, 0.02],
                    help="Object world position (x y z)")
    ap.add_argument("--gripper-config", default=None,
                    help="GraspGen gripper config YAML (omit for --mock)")
    ap.add_argument("--mock", action="store_true",
                    help="Use mock model (random grasps, no GPU)")
    ap.add_argument("--top-k", type=int, default=10, help="Number of grasps to show")
    ap.add_argument("--port", type=int, default=8080, help="Viser server port")
    return ap.parse_args()


class MockModel:
    """Generate random top-down grasps around the point cloud centroid."""

    def predict(self, pc: np.ndarray):
        centroid = pc.mean(axis=0)
        n = 20
        poses = np.zeros((n, 4, 4), dtype=np.float32)
        scores = np.linspace(0.95, 0.5, n).astype(np.float32)
        rng = np.random.default_rng(42)
        for i in range(n):
            poses[i] = np.eye(4)
            poses[i, :3, 3] = centroid + rng.normal(0, 0.005, size=3)
            poses[i, 2, 3] = centroid[2] + 0.01 * (i + 1)
        return poses, scores


def main():
    args = parse_args()

    if args.assets_root is None:
        assets_root = Path(__file__).resolve().parent.parent.parent / "assets"
    else:
        assets_root = Path(args.assets_root)

    library = AssetLibrary(str(assets_root))
    asset = library.get(args.asset)
    if asset is None:
        print(f"[error] Asset '{args.asset}' not found in {assets_root}")
        print(f"[info] Available: {library.list_names()}")
        return

    object_pos = np.array(args.object_pos, dtype=np.float32)

    print(f"[asset] {asset.name} @ scale={args.scale}")
    print(f"[asset] visual_mesh={asset.visual_mesh}")
    print(f"[asset] collision_mesh={asset.collision_mesh}")

    pc_local = asset_to_pointcloud(asset, args.n_points, scale=args.scale)
    print(f"[pc] local frame: shape={pc_local.shape}, "
          f"extent={pc_local.max(0) - pc_local.min(0)}")

    pc_world = asset_to_pointcloud(
        asset, args.n_points,
        object_pos=object_pos, scale=args.scale,
    )
    print(f"[pc] world frame: center={pc_world.mean(0)}")

    # --- Run model ---
    if args.mock:
        print("[model] Using mock model (random grasps)")
        model = MockModel()
    else:
        if args.gripper_config is None:
            print("[error] --gripper-config required when not using --mock")
            return
        from robotsmith.grasp.graspgen_wrapper import GraspGenModel
        model = GraspGenModel(args.gripper_config)
        print(f"[model] Loading GraspGen from {args.gripper_config}")

    grasp_poses, grasp_scores = model.predict(pc_local)
    print(f"[grasps] {len(grasp_poses)} candidates, "
          f"score range: [{grasp_scores.min():.3f}, {grasp_scores.max():.3f}]")

    # Transform to world frame
    from robotsmith.grasp.transforms import pose_matrix
    T_world = pose_matrix(object_pos)
    world_poses = np.array([T_world @ g for g in grasp_poses])

    # --- Visualize with viser ---
    try:
        import viser
    except ImportError:
        print("[error] viser not installed. Run: pip install -e '.[viz]'")
        return

    server = viser.ViserServer(port=args.port)
    print(f"[viser] Open http://localhost:{args.port}")

    server.scene.add_point_cloud(
        "/object/pointcloud",
        points=pc_world,
        colors=np.full((len(pc_world), 3), 180, dtype=np.uint8),
        point_size=0.003,
    )

    top_k = min(args.top_k, len(world_poses))
    for i in range(top_k):
        pose = world_poses[i]
        score = float(grasp_scores[i])
        pos = pose[:3, 3]
        quat_wxyz = rotmat_to_quat_wxyz(pose[:3, :3])
        # viser uses wxyz
        r = min(255, int(255 * (1 - score)))
        g = min(255, int(255 * score))
        server.scene.add_frame(
            f"/grasps/{i:03d}",
            wxyz=quat_wxyz,
            position=pos,
            axes_length=0.03,
            axes_radius=0.001,
        )
        server.scene.add_label(
            f"/grasps/{i:03d}/label",
            text=f"#{i} s={score:.2f}",
            wxyz=quat_wxyz,
            position=pos + np.array([0, 0, 0.02]),
        )

    # Table plane reference
    server.scene.add_frame(
        "/world",
        wxyz=np.array([1, 0, 0, 0]),
        position=np.array([0, 0, 0]),
        axes_length=0.1,
        axes_radius=0.002,
    )

    print(f"[viser] Showing {top_k} grasps. Press Ctrl+C to exit.")
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
