import trimesh

from robotsmith.assets import store
from robotsmith.assets.audit import AuditConfig, audit_and_update, audit_mesh
from robotsmith.assets.schema import Asset, AssetMetadata


def _write_asset(tmp_path, visual_mesh, collision_mesh, size_cm) -> Asset:
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()
    visual_path = asset_dir / "visual.obj"
    collision_path = asset_dir / "collision.obj"
    visual_mesh.export(str(visual_path))
    collision_mesh.export(str(collision_path))
    (asset_dir / "model.urdf").write_text(
        '<robot name="asset"><link name="l"><visual><geometry>'
        '<mesh filename="visual.obj"/></geometry></visual></link></robot>'
    )
    metadata = AssetMetadata(
        size_cm=size_cm,
        source="objaverse",
        metric_scale=1.0,
    )
    store.new_asset(asset_dir, "rigid", "objaverse", {
        "metric_scale": store.annotation(1.0, "human"),
    })
    return Asset(
        name="asset",
        root_dir=asset_dir,
        urdf_path=asset_dir / "model.urdf",
        metadata=metadata,
        visual_mesh=visual_path,
        collision_mesh=collision_path,
    )


def test_audit_mesh_accepts_matching_visual_collision(tmp_path):
    visual = trimesh.creation.box(extents=[0.08, 0.05, 0.1])
    collision = visual.copy()
    asset = _write_asset(tmp_path, visual, collision, [8.0, 5.0, 10.0])

    mesh = audit_mesh(asset, AuditConfig(consistency_sample_points=1200, consistency_grid=12))

    assert "visual_collision_mismatch" not in mesh["issues"]
    assert mesh["visual_collision_consistency"]["ok"]


def test_audit_mesh_rejects_missing_visual_protrusion(tmp_path):
    body = trimesh.creation.box(extents=[0.06, 0.05, 0.1])
    handle = trimesh.creation.box(extents=[0.05, 0.02, 0.05])
    handle.apply_translation([0.055, 0.0, 0.0])
    visual = trimesh.util.concatenate([body, handle])
    collision = body.copy()
    asset = _write_asset(tmp_path, visual, collision, [10.5, 5.0, 10.0])

    mesh = audit_mesh(
        asset,
        AuditConfig(
            consistency_sample_points=1200,
            consistency_grid=12,
            consistency_min_coverage=0.9,
        ),
    )

    assert "visual_collision_mismatch" in mesh["issues"]
    assert not mesh["visual_collision_consistency"]["ok"]


def test_audit_and_update_records_checks_bound_to_build_files(tmp_path):
    visual = trimesh.creation.box(extents=[0.08, 0.05, 0.1])
    asset = _write_asset(tmp_path, visual, visual.copy(), [8.0, 5.0, 10.0])

    audit_and_update(asset, AuditConfig(consistency_sample_points=1200, consistency_grid=12))

    meta = store.load_metadata(asset.root_dir)
    assert meta.mesh["issues"] == []
    assert meta.metric_scale == asset.metadata.metric_scale
    data = store.read_asset(asset.root_dir)
    assert data["annotations"]["metric_scale"]["by"] == "check:gripper_fit@1"
    assert set(store.current_checks(asset.root_dir, data)) == {"mesh_consistency", "gripper_fit"}

    trimesh.creation.box(extents=[0.02, 0.02, 0.02]).export(str(asset.collision_mesh))
    assert store.current_checks(asset.root_dir, data) == {}
    assert store.load_metadata(asset.root_dir).mesh == {}
