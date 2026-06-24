"""AssetLibrary: central catalog for sim-ready assets.

Assets live under these directories:
  - ``root/objects/``   — curated / built-in assets (always scanned)
  - ``root/generated/`` — pipeline-generated assets (also scanned)
  - ``root/articraft/`` — articulated assets (URDF + joints) from Articraft

A lightweight ``root/catalog.json`` persists the full index so that assets
generated on remote GPU nodes and synced back are immediately visible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from robotsmith.assets.schema import Asset, AssetMetadata, parse_urdf_joints

_CATALOG_FILE = "catalog.json"
_SCAN_DIRS = ("objects", "generated", "articraft")


class AssetLibrary:
    """Manages a directory-based catalog of sim-ready assets."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._assets: dict[str, Asset] = {}
        self._load_catalog()

    def _load_catalog(self) -> None:
        """Scan objects/ and generated/ for assets with model.urdf."""
        for subdir_name in _SCAN_DIRS:
            scan_dir = self.root / subdir_name
            if not scan_dir.exists():
                continue
            for asset_dir in sorted(scan_dir.iterdir()):
                if not asset_dir.is_dir():
                    continue
                self._load_asset_dir(asset_dir)

    def _load_asset_dir(self, asset_dir: Path) -> Optional[Asset]:
        """Load a single asset directory into the in-memory catalog."""
        urdf_path = asset_dir / "model.urdf"
        if not urdf_path.exists():
            return None
        meta_path = asset_dir / "metadata.json"
        metadata = (
            AssetMetadata.load(meta_path)
            if meta_path.exists()
            else AssetMetadata()
        )
        asset = Asset(
            name=asset_dir.name,
            root_dir=asset_dir,
            urdf_path=urdf_path,
            metadata=metadata,
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

    # ------ persistent catalog.json ------

    def save_catalog(self) -> Path:
        """Write a lightweight catalog.json listing every known asset.

        Only stores metadata + relative paths (no mesh data).
        Safe to commit to git as a small JSON index.
        """
        entries = []
        for asset in self._assets.values():
            try:
                rel = asset.root_dir.relative_to(self.root)
            except ValueError:
                rel = asset.root_dir
            entries.append({
                "name": asset.name,
                "dir": str(rel),
                "source": asset.metadata.source,
                "mass_kg": asset.metadata.mass_kg,
                "size_cm": asset.metadata.size_cm,
                "metric_scale": asset.metadata.metric_scale,
                "description": asset.metadata.description,
            })
        catalog_path = self.root / _CATALOG_FILE
        catalog_path.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return catalog_path

    # ------ access ------

    def get(self, name: str) -> Optional[Asset]:
        """Get asset by exact name."""
        return self._assets.get(name)

    def add(self, asset: Asset) -> None:
        """Register an asset and persist the catalog index."""
        self._assets[asset.name] = asset
        self.save_catalog()

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
