"""Policy bucket onboarding decisions, stored in the grasp sidecar
``assets/grasp_policy_buckets.json`` (not in the asset model)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robotsmith.assets import store
from robotsmith.assets.library import AssetLibrary
from robotsmith.assets.schema import AssetMetadata

POLICY_BUCKETS = (
    "top_down_axis_horizontal",
    "oblique_axis_horizontal",
    "side_axis_horizontal",
    "top_down_axis_tilted",
    "oblique_axis_tilted",
    "side_axis_tilted",
    "top_down_axis_vertical",
    "oblique_axis_vertical",
    "side_axis_vertical",
)


def load_metadata(asset_name: str, assets_root: Path) -> AssetMetadata:
    asset = AssetLibrary(assets_root).get(asset_name)
    if asset is None:
        raise KeyError(f"unknown asset {asset_name!r} under {assets_root}")
    return asset.metadata


def save_policy_bucket(
    asset_name: str,
    assets_root: Path,
    bucket: str | list[str],
) -> None:
    store.save_bucket(assets_root, asset_name, bucket)


def bucket_hard_ok_counts(result: dict[str, Any]) -> dict[str, int]:
    diagnostics = result.get("episode_diagnostics", [])
    if not diagnostics:
        return {}
    counts = diagnostics[0].get("p0_bucket_hard_ok_counts", {}) or {}
    return {str(bucket): int(count) for bucket, count in counts.items()}


def eligible_probe_buckets(
    result: dict[str, Any],
    requested_buckets: list[str] | tuple[str, ...],
) -> list[str]:
    counts = bucket_hard_ok_counts(result)
    return [bucket for bucket in requested_buckets if counts.get(bucket, 0) > 0]


def select_positive_winners(
    probe_results: list[dict[str, Any]],
) -> list[str]:
    if not probe_results:
        return []
    best_successes = max(int(r.get("success_count", 0)) for r in probe_results)
    if best_successes <= 0:
        return []
    return [
        str(r["bucket"])
        for r in probe_results
        if int(r.get("success_count", 0)) == best_successes
    ]
