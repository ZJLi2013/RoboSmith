from pathlib import Path
from types import SimpleNamespace

from robotsmith.assets.schema import NO_POSITIVE_GRASP_POLICY_BUCKET
from robotsmith.rollout import rollout_orchestrator as orchestrator


def _config(tmp_path: Path, **kwargs) -> orchestrator.RolloutOrchestratorConfig:
    defaults = {
        "assets": ["asset_a"],
        "assets_root": tmp_path / "assets",
        "output_dir": tmp_path / "output",
        "probe_buckets": ["bucket_a", "bucket_b"],
        "stage1_only": True,
    }
    defaults.update(kwargs)
    return orchestrator.RolloutOrchestratorConfig(**defaults)


def test_run_asset_skips_existing_no_positive(monkeypatch, tmp_path):
    monkeypatch.setattr(
        orchestrator,
        "load_metadata",
        lambda asset_name, assets_root: SimpleNamespace(
            grasp_policy_bucket=NO_POSITIVE_GRASP_POLICY_BUCKET,
        ),
    )

    record = orchestrator.run_asset(_config(tmp_path), "asset_a")

    assert record["stage1"] == "skipped_existing_no_positive"
    assert record["status"] == "no_positive"
    assert record["final_grasp_policy_bucket"] == NO_POSITIVE_GRASP_POLICY_BUCKET


def test_run_asset_uses_existing_metadata_bucket_for_stage2(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        orchestrator,
        "load_metadata",
        lambda asset_name, assets_root: SimpleNamespace(
            grasp_policy_bucket="bucket_a",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_run_validation",
        lambda **kwargs: calls.append(kwargs) or {
            "asset": kwargs["asset_name"],
            "success_episode_ids": [0],
            "success_rate": 1.0,
        },
    )

    record = orchestrator.run_asset(_config(tmp_path, stage1_only=False), "asset_a")

    assert record["stage1"] == "skipped_existing_metadata"
    assert record["status"] == "done"
    assert record["final_grasp_policy_bucket"] == "bucket_a"
    assert calls == [{
        "asset_name": "asset_a",
        "output_root": tmp_path / "output" / "asset_a" / "stage2",
        "n_episodes": 5,
        "seed": 42,
        "grasp_planner": "learned",
        "no_videos": False,
        "clean": True,
    }]


def test_run_asset_stops_when_p0_scan_has_no_eligible_buckets(monkeypatch, tmp_path):
    monkeypatch.setattr(
        orchestrator,
        "load_metadata",
        lambda asset_name, assets_root: SimpleNamespace(grasp_policy_bucket=None),
    )
    monkeypatch.setattr(
        orchestrator,
        "_scan_asset",
        lambda config, asset_name: {
            "episode_diagnostics": [{
                "p0_bucket_hard_ok_counts": {"other_bucket": 3},
            }],
        },
    )

    record = orchestrator.run_asset(_config(tmp_path), "asset_a")

    assert record["status"] == "no_eligible_probe_buckets"
    assert record["stage1"]["eligible_probe_buckets"] == []
    assert record["stage1"]["probe_results"] == []


def test_run_asset_writes_tied_positive_buckets(monkeypatch, tmp_path):
    written = []
    monkeypatch.setattr(
        orchestrator,
        "load_metadata",
        lambda asset_name, assets_root: SimpleNamespace(grasp_policy_bucket=None),
    )
    monkeypatch.setattr(
        orchestrator,
        "_scan_asset",
        lambda config, asset_name: {
            "episode_diagnostics": [{
                "p0_bucket_hard_ok_counts": {"bucket_a": 1, "bucket_b": 2},
            }],
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "_run_validation",
        lambda **kwargs: {
            "asset": kwargs["asset_name"],
            "success_episode_ids": [0],
            "success_rate": 1.0,
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "save_policy_bucket",
        lambda asset_name, assets_root, bucket: written.append(
            (asset_name, assets_root, bucket)
        ),
    )

    config = _config(tmp_path, stage1_only=True)
    record = orchestrator.run_asset(config, "asset_a")

    assert record["status"] == "stage1_done"
    assert record["tied_buckets"] == ["bucket_a", "bucket_b"]
    assert record["written_grasp_policy_bucket"] == ["bucket_a", "bucket_b"]
    assert written == [("asset_a", tmp_path / "assets", ["bucket_a", "bucket_b"])]


def test_run_rollout_orchestrator_writes_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(
        orchestrator,
        "run_asset",
        lambda config, asset_name: {"asset": asset_name, "status": "done"},
    )

    config = _config(tmp_path, assets=["asset_a", "asset_b"])
    summary = orchestrator.run_rollout_orchestrator(config)

    assert summary["assets"] == ["asset_a", "asset_b"]
    assert summary["results"] == [
        {"asset": "asset_a", "status": "done"},
        {"asset": "asset_b", "status": "done"},
    ]
    assert (tmp_path / "output" / "metadata_guided_rollout_summary.json").exists()
