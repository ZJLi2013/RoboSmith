"""Sync generated assets from remote GPU node to local workspace.

Usage:
    python scripts/sync_assets.py <ssh-host> [--remote-dir /path/to/assets/generated]
    python scripts/sync_assets.py banff-sc-cs41-29.amd.com --remote-dir /data/robotsmith/assets/generated
    python scripts/sync_assets.py banff-sc-cs41-29.amd.com --docker rocm_dev  # if assets are inside a container

After sync, updates assets/catalog.json so the AssetLibrary sees newly pulled assets
without having to re-run generation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_GENERATED = REPO_ROOT / "assets" / "generated"
ASSET_FILES = ("model.urdf", "metadata.json", "visual.obj", "collision.obj",
               "visual.glb", "collision.glb", "reference.png")


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def list_remote_assets(host: str, remote_dir: str, docker: str | None) -> list[str]:
    """List asset subdirectories on the remote."""
    ls_cmd = f"ls -1 {remote_dir} 2>/dev/null || true"
    if docker:
        ls_cmd = f"docker exec {docker} bash -c '{ls_cmd}'"
    r = run(["ssh", host, ls_cmd], check=False)
    if r.returncode != 0 or not r.stdout.strip():
        print(f"  No assets found at {host}:{remote_dir}")
        return []
    return [d.strip() for d in r.stdout.strip().split("\n") if d.strip()]


def sync_asset(host: str, remote_dir: str, asset_name: str,
               docker: str | None) -> bool:
    """Copy one asset directory from remote to local. Returns True if new files."""
    local_dir = LOCAL_GENERATED / asset_name
    local_dir.mkdir(parents=True, exist_ok=True)

    pulled = False
    for fname in ASSET_FILES:
        local_file = local_dir / fname
        if local_file.exists():
            continue

        remote_path = f"{remote_dir}/{asset_name}/{fname}"

        if docker:
            # docker cp from running container to /tmp, then scp
            tmp_path = f"/tmp/_sync_{fname}"
            cp_cmd = f"docker cp {docker}:{remote_path} {tmp_path} 2>/dev/null"
            r = run(["ssh", host, cp_cmd], check=False)
            if r.returncode != 0:
                continue
            r = run(["scp", f"{host}:{tmp_path}", str(local_file)], check=False)
            run(["ssh", host, f"rm -f {tmp_path}"], check=False)
        else:
            r = run(["scp", f"{host}:{remote_path}", str(local_file)], check=False)

        if r.returncode == 0 and local_file.exists():
            pulled = True
            print(f"    + {fname}")

    return pulled


def rebuild_catalog() -> None:
    """Rebuild catalog.json from local filesystem."""
    sys.path.insert(0, str(REPO_ROOT))
    from robotsmith.assets.library import AssetLibrary
    lib = AssetLibrary(REPO_ROOT / "assets")
    catalog_path = lib.save_catalog()
    print(f"\n  catalog.json updated: {len(lib)} assets ({len(lib.list_generated())} generated)")
    print(f"  -> {catalog_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync generated assets from remote GPU node")
    parser.add_argument("host", help="SSH host (e.g. banff-sc-cs41-29.amd.com)")
    parser.add_argument("--remote-dir", default="/data/robotsmith/assets/generated",
                        help="Remote path to generated/ directory")
    parser.add_argument("--docker", default=None,
                        help="Docker container name if assets are inside a container")
    parser.add_argument("--dry-run", action="store_true",
                        help="List remote assets without downloading")
    args = parser.parse_args()

    print(f"=== Asset Sync: {args.host} -> local ===")
    print(f"  Remote: {args.remote_dir}")
    print(f"  Local:  {LOCAL_GENERATED}\n")

    assets = list_remote_assets(args.host, args.remote_dir, args.docker)
    if not assets:
        print("Nothing to sync.")
        return

    print(f"Found {len(assets)} remote asset(s): {', '.join(assets)}\n")

    if args.dry_run:
        return

    new_count = 0
    for name in assets:
        print(f"[{name}]")
        if sync_asset(args.host, args.remote_dir, name, args.docker):
            new_count += 1
        else:
            print("    (up to date)")

    print(f"\nSynced {new_count} new/updated asset(s).")
    rebuild_catalog()


if __name__ == "__main__":
    main()
