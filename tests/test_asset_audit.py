import json

import trimesh

from robotsmith.assets.audit import AuditConfig, audit_mesh
from robotsmith.assets.schema import Asset, AssetMetadata


def _write_asset(tmp_path, visual_mesh, collision_mesh, size_cm) -> Asset:
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()
    visual_path = asset_dir / "visual.obj"
    collision_path = asset_dir / "collision.obj"
    visual_mesh.export(str(visual_path))
    collision_mesh.export(str(collision_path))
    metadata = AssetMetadata(
        size_cm=size_cm,
        source="objaverse",
        metric_scale=1.0,
    )
    (asset_dir / "metadata.json").write_text(json.dumps({}), encoding="utf-8")
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
