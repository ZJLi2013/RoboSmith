"""AssetLibrary: central catalog for sim-ready assets.

Each asset is a directory ``root/<id>/`` holding ``model.urdf``, its meshes,
``asset.json`` and ``checks.jsonl`` (see ``robotsmith.assets.store``). Drafts
load too when their build files are present; scenarios refuse them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from robotsmith.assets import store
from robotsmith.assets.schema import Asset, parse_urdf_joints

class AssetLibrary:
    """Manages a directory-based catalog of sim-ready assets."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._assets: dict[str, Asset] = {}
        self._load_catalog()

    def _load_catalog(self) -> None:
        """Scan ``root/<id>/`` for assets with model.urdf + asset.json."""
        buckets = store.load_buckets(self.root)
        if not self.root.exists():
            return
        for asset_dir in sorted(self.root.iterdir()):
            if asset_dir.is_dir():
                self._load_asset_dir(asset_dir, buckets)

    def _load_asset_dir(self, asset_dir: Path, buckets: dict) -> Optional[Asset]:
        """Load a single asset directory into the in-memory catalog."""
        urdf_path = asset_dir / "model.urdf"
        if not urdf_path.exists() or not (asset_dir / store.ASSET_FILE).exists():
            return None
        asset = Asset(
            name=asset_dir.name,
            root_dir=asset_dir,
            urdf_path=urdf_path,
            metadata=store.load_metadata(asset_dir, buckets),
            visual_mesh=self._find_mesh(asset_dir, "visual"),
            collision_mesh=self._find_mesh(asset_dir, "collision"),
            joints=parse_urdf_joints(urdf_path),
        )
        self._assets[asset.name] = asset
        return asset

    @staticmethod
    def _find_mesh(asset_dir: Path, prefix: str) -> Optional[Path]:
        for ext in (".obj", ".stl", ".ply", ".glb"):
            p = asset_dir / f"{prefix}{ext}"
            if p.exists():
                return p
        return None

    # ------ access ------

    def get(self, name: str) -> Optional[Asset]:
        """Get asset by exact name."""
        return self._assets.get(name)

    def add(self, asset: Asset) -> None:
        """Register an asset in memory."""
        self._assets[asset.name] = asset

    def list_all(self) -> list[Asset]:
        return list(self._assets.values())

    def list_names(self) -> list[str]:
        return list(self._assets.keys())

    def list_generated(self) -> list[Asset]:
        """List only pipeline-generated assets."""
        return [a for a in self._assets.values() if a.metadata.source == "generated"]

    def __len__(self) -> int:
        return len(self._assets)

    def __repr__(self) -> str:
        n_gen = len(self.list_generated())
        n_builtin = len(self) - n_gen
        return f"AssetLibrary({self.root}, {n_builtin} builtin + {n_gen} generated)"
