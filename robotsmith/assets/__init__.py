"""assets — the asset facts source consumed by every downstream layer.

``AssetLibrary`` onboards RocRecon output into a catalog; ``Asset`` /
``AssetMetadata`` carry the facts (``metric_scale``, ``task_poses.upright``,
physical params, provenance) that scene / grasp / motion layers read. This layer
states *what an object is*; it does not decide scene placement, task, or grasp.
"""

from robotsmith.assets.schema import Asset, AssetMetadata
from robotsmith.assets.library import AssetLibrary
from robotsmith.assets.audit import (
    AuditConfig,
    SUPPORT_ASSETS,
    audit_and_update,
    audit_asset,
    audit_assets,
    audit_mesh,
)

__all__ = [
    "Asset",
    "AssetMetadata",
    "AssetLibrary",
    "AuditConfig",
    "SUPPORT_ASSETS",
    "audit_and_update",
    "audit_asset",
    "audit_assets",
    "audit_mesh",
]
