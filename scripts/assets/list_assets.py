"""Print the asset catalog a scenario author picks from: id, status, category, kind, size, place
targets, movable joints, description. Scenarios load only ``admitted`` assets. With a search flag,
print ``AssetLibrary.search_assets`` results instead, nearest first.

Usage:
  python scripts/assets/list_assets.py
  python scripts/assets/list_assets.py --category drawer_cabinet --size 37 42 25 --query "single-drawer cabinet"
"""

from __future__ import annotations

import argparse
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


def search(library: AssetLibrary, args) -> None:
    hits = library.search_assets(args.query, category=args.category, size_cm=args.size, include_drafts=args.drafts)
    print(f"{'id':<24}{'status':<10}{'category':<18}{'size (cm)':<22}{'size diff':<11}shared words")
    for h in hits:
        size = " x ".join(f"{v:g}" for v in h["size_cm"])
        diff = "-" if h["size_diff"] is None else f"{h['size_diff']:.3f}"
        print(f"{h['id']:<24}{h['status']:<10}{h['category']:<18}{size:<22}{diff:<11}{', '.join(h['shared_words']) or '-'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--category")
    ap.add_argument("--size", type=float, nargs=3, metavar=("X", "Y", "Z"), help="target size in cm")
    ap.add_argument("--query", default="")
    ap.add_argument("--drafts", action="store_true", help="include drafts in search results")
    args = ap.parse_args()
    library = AssetLibrary(REPO_ROOT / "assets")
    if args.category or args.size or args.query:
        search(library, args)
        return
    print(f"{'id':<24}{'status':<10}{'category':<18}{'kind':<9}{'size (cm)':<22}{'place targets':<28}{'joints':<16}description")
    for d in package.asset_dirs(REPO_ROOT / "assets"):
        if d.name == "plane":
            continue
        reasons = store.draft_reasons(d)
        status = "draft" if reasons else "admitted"
        category = store.read_asset(d).get("category") or "-"
        asset = library.get(d.name)
        if asset is None:
            print(f"{d.name:<24}{status:<10}{category:<18}{'-':<9}{'-':<22}{'-':<28}{'-':<16}({'; '.join(reasons)})")
            continue
        m = asset.metadata
        size = " x ".join(f"{v:g}" for v in m.size_cm)
        targets = ", ".join(t["name"] for t in m.place_targets) or "-"
        joints = ", ".join(j.name for j in asset.movable_joints) or "-"
        desc = re.sub(r"\s*\[uid=[^\]]*\]|\s*\(Objaverse\)", "", m.description or "")
        desc = desc.split(". ")[0].rstrip(".")
        if reasons:
            desc = f"({'; '.join(reasons)}) {desc}"
        print(f"{asset.name:<24}{status:<10}{category:<18}{kind(asset):<9}{size:<22}{targets:<28}{joints:<16}{desc}")


if __name__ == "__main__":
    main()
