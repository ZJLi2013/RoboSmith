"""AssetLibrary: central catalog for sim-ready assets.

Each asset is a directory ``root/<id>/`` holding ``model.urdf``, its meshes,
``asset.json`` and ``checks.jsonl`` (see ``robotsmith.assets.store``). Drafts
load too when their build files are present; scenarios refuse them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from robotsmith.assets import store
from robotsmith.assets.schema import Asset, parse_urdf_joints

_STOPWORDS = {"a", "an", "and", "the", "with", "of", "for", "in", "on", "to", "by", "one", "is"}


def _words(text: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", text.lower()) if w and w not in _STOPWORDS}


def _size_diff(size: list[float], target: list[float]) -> float:
    """Mean relative difference over sorted horizontal edges and height."""
    a = sorted(size[:2]) + [size[2]]
    b = sorted(target[:2]) + [target[2]]
    return sum(abs(p - q) / max(p, q, 1e-9) for p, q in zip(a, b)) / 3

class AssetLibrary:
    """Manages a directory-based catalog of sim-ready assets."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._assets: dict[str, Asset] = {}
        self._load_catalog()

    def _load_catalog(self) -> None:
        """Scan ``root/<id>/`` for assets with model.urdf + asset.json."""
        if not self.root.exists():
            return
        for asset_dir in sorted(self.root.iterdir()):
            if asset_dir.is_dir():
                self._load_asset_dir(asset_dir)

    def _load_asset_dir(self, asset_dir: Path) -> Optional[Asset]:
        """Load a single asset directory into the in-memory catalog."""
        urdf_path = asset_dir / "model.urdf"
        if not urdf_path.exists() or not (asset_dir / store.ASSET_FILE).exists():
            return None
        asset = Asset(
            name=asset_dir.name,
            root_dir=asset_dir,
            urdf_path=urdf_path,
            metadata=store.load_metadata(asset_dir),
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

    def search_assets(self, query: str = "", category: Optional[str] = None,
                      size_cm: Optional[list[float]] = None, include_drafts: bool = False) -> list[dict]:
        """Admitted assets matching ``category`` exactly (or sharing a query word when no category is
        given), nearest ``size_cm`` first, then most shared query words, then id."""
        words = _words(query)
        hits = []
        for asset in self._assets.values():
            data = store.read_asset(asset.root_dir)
            reasons = store.draft_reasons(asset.root_dir, data)
            if reasons and not include_drafts:
                continue
            cat = data.get("category", "")
            if category is not None and cat != category:
                continue
            shared = sorted(words & _words(" ".join([asset.metadata.description, cat, data["id"]])))
            if category is None and not shared:
                continue
            size = asset.metadata.size_cm
            hits.append({
                "id": data["id"], "status": "draft" if reasons else "admitted", "category": cat,
                "build_hash": data["build"]["hash"], "size_cm": size, "shared_words": shared,
                "size_diff": round(_size_diff(size, size_cm), 3) if size_cm else None,
                "draft_reasons": reasons,
            })
        return sorted(hits, key=lambda h: (h["size_diff"] or 0.0, -len(h["shared_words"]), h["id"]))

    def list_generated(self) -> list[Asset]:
        """List only pipeline-generated assets."""
        return [a for a in self._assets.values() if a.metadata.source == "generated"]

    def __len__(self) -> int:
        return len(self._assets)

    def __repr__(self) -> str:
        n_gen = len(self.list_generated())
        n_builtin = len(self) - n_gen
        return f"AssetLibrary({self.root}, {n_builtin} builtin + {n_gen} generated)"
