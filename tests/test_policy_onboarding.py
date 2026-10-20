from robotsmith.assets import store
from robotsmith.grasp.policy_onboarding import (
    eligible_probe_buckets,
    load_metadata,
    save_policy_bucket,
    select_positive_winners,
)


def _write_asset(tmp_path, subdir: str, name: str):
    asset_dir = tmp_path / "assets" / subdir / name
    asset_dir.mkdir(parents=True)
    (asset_dir / "model.urdf").write_text(
        "<robot name='test'><link name='l'><visual><geometry><box size='0.05 0.05 0.05'/>"
        "</geometry></visual></link></robot>",
        encoding="utf-8",
    )
    store.new_asset(asset_dir, "rigid", "generated", {})
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


def test_save_policy_bucket_writes_sidecar_not_asset_json(tmp_path):
    asset_dir = _write_asset(tmp_path, "generated", "battery_1")
    before = (asset_dir / "asset.json").read_text(encoding="utf-8")

    save_policy_bucket("battery_1", tmp_path / "assets", ["side_axis_horizontal"])

    assert load_metadata("battery_1", tmp_path / "assets").grasp_policy_bucket == ["side_axis_horizontal"]
    assert (asset_dir / "asset.json").read_text(encoding="utf-8") == before
