"""Print the asset catalog a scenario author picks from: id, kind, size, place targets,
movable joints, description.

Usage:
  python scripts/assets/list_assets.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robotsmith.assets.library import AssetLibrary  # noqa: E402


def kind(asset) -> str:
    if asset.name == "table_simple":
        return "support"
    return "fixture" if asset.is_fixture else "object"


def main() -> None:
    library = AssetLibrary(REPO_ROOT / "assets")
    print(f"{'id':<22}{'kind':<9}{'size (cm)':<22}{'place targets':<28}{'joints':<16}description")
    for asset in sorted(library.list_all(), key=lambda a: a.name):
        if asset.name == "plane":
            continue
        m = asset.metadata
        size = " x ".join(f"{v:g}" for v in m.size_cm)
        targets = ", ".join(t["name"] for t in m.place_targets) or "-"
        joints = ", ".join(j.name for j in asset.movable_joints) or "-"
        desc = re.sub(r"\s*\[uid=[^\]]*\]|\s*\(Objaverse\)", "", m.description or "")
        desc = desc.split(". ")[0].rstrip(".")
        print(f"{asset.name:<22}{kind(asset):<9}{size:<22}{targets:<28}{joints:<16}{desc}")


if __name__ == "__main__":
    main()
