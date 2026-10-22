"""Publish build files to, and fetch them from, a package addressed by build hash.

A store is a local directory or ``hf://<owner>/<dataset>`` (a public Hugging Face
dataset). Build files live at ``<id>/<build_hash>/<file>``, so paths never change
and the ``asset.json`` files in git are the lock.

Usage:
  python -m robotsmith.assets.fetch
  python -m robotsmith.assets.package publish [--store hf://owner/dataset]
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from robotsmith.assets import store
from robotsmith.assets.library import _SCAN_DIRS

DEFAULT_STORE = "hf://lidavidsh/robosmith-assets"
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "assets"
REDISTRIBUTABLE = frozenset({"CC0-1.0", "CC-BY-4.0", "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"})


class PublishRefused(ValueError):
    pass


def asset_dirs(assets_root: Path) -> list[Path]:
    """The assets a package carries: every asset directory the library scans."""
    return sorted(p.parent for sub in _SCAN_DIRS for p in (assets_root / sub).glob(f"*/{store.ASSET_FILE}"))


def redistributable(license: str | None) -> bool:
    return bool(license) and all(part.strip() in REDISTRIBUTABLE for part in license.split(" AND "))


def _hf_repo(spec: str) -> str | None:
    return spec.removeprefix("hf://") if spec.startswith("hf://") else None


def _download(spec: str, rel: str, dst: Path) -> None:
    repo = _hf_repo(spec)
    if repo is None:
        shutil.copyfile(Path(spec) / rel, dst)
        return
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{rel}"
    with urllib.request.urlopen(url, timeout=120) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)


def fetch(assets_root: Path = DEFAULT_ROOT, spec: str = DEFAULT_STORE) -> list[str]:
    """Download the build files of every asset whose local copy is missing or stale; return those ids."""
    fetched = []
    for d in asset_dirs(assets_root):
        data = store.read_asset(d)
        files, digest = data["build"]["files"], data["build"]["hash"]
        if all((d / f).exists() for f in files) and store.build_hash(d, files) == digest:
            continue
        for f in files:
            _download(spec, f"{data['id']}/{digest}/{f}", d / f)
        if store.build_hash(d, files) != digest:
            raise ValueError(f"{data['id']}: fetched build files do not match build hash {digest}")
        fetched.append(data["id"])
    return fetched


def _card(assets: list[dict]) -> str:
    rows = "\n".join(
        f"| `{a['id']}` | {a['source']['license']} | {a['source'].get('attribution', '')} |" for a in assets)
    return (
        "---\nlicense: other\n---\n\n# RoboSmith assets\n\n"
        "Build files of the RoboSmith asset library at `<id>/<build_hash>/`. "
        "Each asset keeps the license below.\n\n"
        f"| asset | license | attribution |\n|---|---|---|\n{rows}\n"
    )


def publish(assets_root: Path = DEFAULT_ROOT, spec: str = DEFAULT_STORE) -> list[str]:
    """Upload the build files the store lacks; refuse everything if any asset may not be redistributed."""
    assets = [(d, store.read_asset(d)) for d in asset_dirs(assets_root)]
    refused = [a["id"] for _, a in assets if not redistributable(a["source"].get("license"))]
    if refused:
        raise PublishRefused(f"license missing or not redistributable: {', '.join(refused)}")
    stale = [a["id"] for d, a in assets if store.build_hash(d, a["build"]["files"]) != a["build"]["hash"]]
    if stale:
        raise ValueError(f"local build files do not match build hash: {', '.join(stale)}")

    files = {f"{a['id']}/{a['build']['hash']}/{f}": d / f for d, a in assets for f in a["build"]["files"]}
    card = _card([a for _, a in assets])
    repo = _hf_repo(spec)
    if repo is None:
        root = Path(spec)
        new = [rel for rel in files if not (root / rel).exists()]
        for rel in new:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(files[rel], root / rel)
        (root / "README.md").write_text(card, encoding="utf-8", newline="\n")
        return new

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    existing = set(api.list_repo_files(repo, repo_type="dataset"))
    new = [rel for rel in files if rel not in existing]
    ops = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(files[rel])) for rel in new]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode("utf-8")))
    api.create_commit(repo, ops, commit_message=f"publish {len(new)} build files", repo_type="dataset")
    return new


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("fetch", "publish"))
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--assets", type=Path, default=DEFAULT_ROOT)
    a = ap.parse_args(argv)
    n = len(asset_dirs(a.assets))
    if a.command == "fetch":
        got = fetch(a.assets, a.store)
        print(f"fetched {len(got)} of {n} assets; {n}/{n} build hash match")
    else:
        new = publish(a.assets, a.store)
        print(f"published {len(new)} new build files for {n} assets to {a.store}")


if __name__ == "__main__":
    main()
