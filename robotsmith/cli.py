"""CLI entry point for RoboSmith asset library."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _get_assets_root() -> Path:
    """Resolve the assets root directory."""
    candidates = [
        Path.cwd() / "assets",
        Path(__file__).parent.parent / "assets",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def cmd_bootstrap(args):
    """Bootstrap built-in assets."""
    from robotsmith.assets.builtin import bootstrap_builtin_assets

    root = Path(args.root)
    assets = bootstrap_builtin_assets(root)
    print(f"Bootstrapped {len(assets)} assets in {root}")
    for a in assets:
        print(f"  {a.name}")


def cmd_list(args):
    """List all assets."""
    from robotsmith.assets.library import AssetLibrary

    lib = AssetLibrary(args.root)
    if len(lib) == 0:
        print("No assets found. Run 'robotsmith bootstrap' first.")
        return
    for a in lib.list_all():
        tags = ", ".join(a.tags[:5])
        print(f"  {a.name:20s}  mass={a.metadata.mass_kg:.2f}kg  tags=[{tags}]")
    print(f"\n  Total: {len(lib)} assets")


def cmd_search(args):
    """Search for assets."""
    from robotsmith.assets.library import AssetLibrary

    lib = AssetLibrary(args.root)
    results = lib.search(args.query, top_k=args.top_k)
    if not results:
        print(f"No match for {args.query!r}.")
        if args.auto_gen:
            print("Auto-generating...")
            try:
                asset = lib.generate(args.query)
                print(f"Generated: {asset.name} -> {asset.urdf_path}")
            except Exception as e:
                print(f"Generation failed: {e}")
        return
    for a in results:
        print(f"  {a.name:20s}  URDF={a.urdf_path}")


def cmd_generate(args):
    """Generate a new asset via 3D gen."""
    if getattr(args, "list_backends", False):
        from robotsmith.gen.backend import list_backend_info
        infos = list_backend_info()
        print("Available 3D generation backends:\n")
        for info in infos:
            avail = "ready" if info.rocm_status == "verified" else info.rocm_status
            pbr = "PBR" if info.has_pbr else "no PBR"
            print(f"  {info.name:12s}  {info.model_name:30s}  {pbr:8s}  VRAM≥{info.min_vram_gb:.0f}GB  ROCm: {avail}")
        return

    from robotsmith.assets.library import AssetLibrary

    lib = AssetLibrary(args.root)

    gen_kwargs = {}
    if args.image:
        gen_kwargs["image_path"] = args.image

    prompt = args.prompt or "generated_object"

    try:
        asset = lib.generate(
            prompt,
            backend=args.backend,
            target_size_m=args.size,
            **gen_kwargs,
        )
        print(f"Generated: {asset.name}")
        print(f"  URDF: {asset.urdf_path}")
        print(f"  Tags: {asset.tags}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _load_scene_presets() -> dict:
    try:
        from robotsmith.scenes.presets import SCENE_PRESETS
        return dict(SCENE_PRESETS)
    except ImportError:
        return {}


def cmd_scene(args):
    """Resolve a scene preset."""
    from robotsmith.assets.library import AssetLibrary
    from robotsmith.scenes.backend import ProgrammaticSceneBackend

    lib = AssetLibrary(args.root)
    presets = _load_scene_presets()

    if args.name not in presets:
        print(f"Unknown scene: {args.name!r}. Available: {list(presets.keys())}")
        sys.exit(1)

    config = presets[args.name]
    backend = ProgrammaticSceneBackend(seed=args.seed)
    resolved = backend.resolve(config, lib)
    print(resolved.summary())

    if args.json:
        print(json.dumps(config.to_dict(), indent=2))


def cmd_validate(args):
    """Validate all assets with PyBullet."""
    try:
        from robotsmith.validate.pybullet_check import validate_all_assets, print_validation_report
    except ImportError:
        print("PyBullet not installed. Install with: pip install pybullet")
        sys.exit(1)

    results = validate_all_assets(Path(args.root))
    print_validation_report(results)


def cmd_view(args):
    """Launch viser-based 3D scene viewer."""
    try:
        from robotsmith.viz.scene_viewer import SceneViewer
    except ImportError:
        print("viser not installed. Install with: pip install 'robotsmith[viz]'")
        sys.exit(1)

    from robotsmith.assets.library import AssetLibrary

    lib = AssetLibrary(args.root)
    if len(lib) == 0:
        print("No assets found. Run 'robotsmith bootstrap' first.")
        sys.exit(1)

    viewer = SceneViewer(port=args.port)

    if args.asset:
        asset = lib.get(args.asset)
        if asset is None:
            print(f"Asset not found: {args.asset!r}")
            sys.exit(1)
        viewer.add_asset(asset, position=(0.0, 0.0, 0.0))
        print(f"Viewing asset: {args.asset}")
    else:
        presets = _load_scene_presets()
        scene_name = args.scene or "tabletop_simple"
        if scene_name not in presets:
            print(f"Unknown scene: {scene_name!r}. Available: {list(presets.keys())}")
            sys.exit(1)

        from robotsmith.scenes.backend import ProgrammaticSceneBackend
        config = presets[scene_name]
        backend = ProgrammaticSceneBackend(seed=args.seed)
        resolved = backend.resolve(config, lib)
        viewer.show_resolved_scene(resolved)
        print(f"Viewing scene: {scene_name} ({len(resolved.placed_objects)} objects)")

    if args.robot:
        robot_path = Path(args.robot)
        if robot_path.exists():
            viewer.add_robot_urdf(robot_path)
            print(f"Robot loaded: {robot_path}")
        else:
            print(f"Robot URDF not found: {robot_path}")

    viewer.run()


def cmd_browse(args):
    """Launch web-based asset gallery browser."""
    try:
        from robotsmith.viz.asset_browser import AssetBrowser
    except ImportError:
        print("viser not installed. Install with: pip install 'robotsmith[viz]'")
        sys.exit(1)

    from robotsmith.assets.library import AssetLibrary

    lib = AssetLibrary(args.root)
    if len(lib) == 0:
        print("No assets found. Run 'robotsmith bootstrap' first.")
        sys.exit(1)

    browser = AssetBrowser(lib, port=args.port, source_filter=args.filter)
    browser.run()


def main():
    parser = argparse.ArgumentParser(prog="robotsmith", description="Sim-ready digital asset library")
    parser.add_argument("--root", default=str(_get_assets_root()), help="Assets root directory")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("bootstrap", help="Generate built-in assets")

    sub.add_parser("list", help="List all assets")

    p_search = sub.add_parser("search", help="Search assets")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--auto-gen", action="store_true", help="Auto-generate if no match")

    p_gen = sub.add_parser("generate", help="Generate new asset via 3D gen")
    p_gen.add_argument("prompt", nargs="?", default="", help="Text prompt for generation / cataloging")
    p_gen.add_argument("--image", default=None, help="Input image for image-to-3D (required for hunyuan3d)")
    p_gen.add_argument("--backend", default="hunyuan3d",
                       choices=["hunyuan3d", "trellis2", "triposg"],
                       help="3D generation backend (default: hunyuan3d)")
    p_gen.add_argument("--size", type=float, default=0.1, help="Target size in meters")
    p_gen.add_argument("--list-backends", action="store_true", help="List available backends")

    p_scene = sub.add_parser("scene", help="Resolve a scene preset")
    p_scene.add_argument("name", help="Scene preset name")
    p_scene.add_argument("--seed", type=int, default=42)
    p_scene.add_argument("--json", action="store_true", help="Print config as JSON")

    sub.add_parser("validate", help="Validate all assets with PyBullet")

    p_view = sub.add_parser("view", help="Launch 3D scene viewer (viser)")
    p_view.add_argument("scene", nargs="?", default=None, help="Scene preset name")
    p_view.add_argument("--asset", default=None, help="View a single asset by name")
    p_view.add_argument("--robot", default=None, help="Path to robot URDF")
    p_view.add_argument("--port", type=int, default=8080, help="Viewer port (default: 8080)")
    p_view.add_argument("--seed", type=int, default=42)

    p_browse = sub.add_parser("browse", help="Browse all assets in web gallery (viser)")
    p_browse.add_argument("--filter", choices=["all", "builtin", "generated"], default="all",
                          help="Filter by source type")
    p_browse.add_argument("--port", type=int, default=8080, help="Browser port (default: 8080)")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    cmd_map = {
        "bootstrap": cmd_bootstrap,
        "list": cmd_list,
        "search": cmd_search,
        "generate": cmd_generate,
        "scene": cmd_scene,
        "validate": cmd_validate,
        "view": cmd_view,
        "browse": cmd_browse,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
