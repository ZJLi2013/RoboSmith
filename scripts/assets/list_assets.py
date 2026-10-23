"""Print the asset catalog a scenario author picks from: id, status, kind, size, place targets,
movable joints, description. Scenarios load only ``admitted`` assets.

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

from robotsmith.assets import package, store  # noqa: E402
from robotsmith.assets.library import AssetLibrary  # noqa: E402


def kind(asset) -> str:
    if asset.name == "table_simple":
        return "support"
    return "fixture" if asset.is_fixture else "object"


def main() -> None:
    library = AssetLibrary(REPO_ROOT / "assets")
    print(f"{'id':<24}{'status':<10}{'kind':<9}{'size (cm)':<22}{'place targets':<28}{'joints':<16}description")
    for d in package.asset_dirs(REPO_ROOT / "assets"):
        if d.name == "plane":
            continue
        reasons = store.draft_reasons(d)
        status = "draft" if reasons else "admitted"
        asset = library.get(d.name)
        if asset is None:
            print(f"{d.name:<24}{status:<10}{'-':<9}{'-':<22}{'-':<28}{'-':<16}({'; '.join(reasons)})")
            continue
        m = asset.metadata
        size = " x ".join(f"{v:g}" for v in m.size_cm)
        targets = ", ".join(t["name"] for t in m.place_targets) or "-"
        joints = ", ".join(j.name for j in asset.movable_joints) or "-"
        desc = re.sub(r"\s*\[uid=[^\]]*\]|\s*\(Objaverse\)", "", m.description or "")
        desc = desc.split(". ")[0].rstrip(".")
        if reasons:
            desc = f"({'; '.join(reasons)}) {desc}"
        print(f"{asset.name:<24}{status:<10}{kind(asset):<9}{size:<22}{targets:<28}{joints:<16}{desc}")


if __name__ == "__main__":
    main()
