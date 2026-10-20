from robotsmith.grasp.policy_onboarding import (
    eligible_probe_buckets,
    metadata_path,
    save_policy_bucket,
    select_positive_winners,
)
from robotsmith.assets.schema import AssetMetadata


def _write_asset(tmp_path, subdir: str, name: str, metadata: AssetMetadata | None = None):
    asset_dir = tmp_path / "assets" / subdir / name
    asset_dir.mkdir(parents=True)
    (asset_dir / "model.urdf").write_text("<robot name='test'></robot>", encoding="utf-8")
    (metadata or AssetMetadata()).save(asset_dir / "metadata.json")
    return asset_dir


def test_select_positive_winners_requires_positive_top():
    winners = select_positive_winners([
        {"bucket": "top_down_axis_horizontal", "success_count": 2},
        {"bucket": "side_axis_vertical", "success_count": 0},
    ])

    assert winners == ["top_down_axis_horizontal"]


def test_select_positive_winners_rejects_all_zero():
    winners = select_positive_winners([
        {"bucket": "top_down_axis_horizontal", "success_count": 0},
        {"bucket": "side_axis_vertical", "success_count": 0},
    ])

    assert winners == []


def test_select_positive_winners_keeps_tied_positive_top():
    winners = select_positive_winners([
        {"bucket": "top_down_axis_horizontal", "success_count": 1},
        {"bucket": "side_axis_vertical", "success_count": 1},
        {"bucket": "oblique_axis_tilted", "success_count": 0},
    ])

    assert winners == ["top_down_axis_horizontal", "side_axis_vertical"]


def test_eligible_probe_buckets_come_from_p0_hard_ok_counts():
    scan_result = {
        "episode_diagnostics": [{
            "p0_bucket_hard_ok_counts": {
                "side_axis_vertical": 14,
                "oblique_axis_vertical": 11,
                "top_down_axis_horizontal": 3,
            },
        }],
    }

    buckets = eligible_probe_buckets(
        scan_result,
        [
            "top_down_axis_horizontal",
            "side_axis_horizontal",
            "side_axis_vertical",
        ],
    )

    assert buckets == ["top_down_axis_horizontal", "side_axis_vertical"]


def test_eligible_probe_buckets_empty_without_p0_hard_ok_counts():
    assert eligible_probe_buckets({"episode_diagnostics": []}, ["side_axis_vertical"]) == []


def test_metadata_path_finds_generated_asset(tmp_path):
    asset_dir = _write_asset(tmp_path, "generated", "battery_1")

    assert metadata_path("battery_1", tmp_path / "assets") == asset_dir / "metadata.json"


def test_save_policy_bucket_updates_generated_asset_metadata(tmp_path):
    asset_dir = _write_asset(tmp_path, "generated", "battery_1")

    save_policy_bucket("battery_1", tmp_path / "assets", ["side_axis_horizontal"])

    meta = AssetMetadata.load(asset_dir / "metadata.json")
    assert meta.grasp_policy_bucket == ["side_axis_horizontal"]
