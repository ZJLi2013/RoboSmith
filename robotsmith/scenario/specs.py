"""Self-describing target frames that the runtime resolves against the live world."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameRef:
    """How a target position anchors to the world.

    A spatial reference is self-describing about its parent frame, so the runtime
    can resolve every reference against the live world each segment instead of
    each primitive deciding anchoring ad-hoc (the gap that left ``place`` reading
    a static world point while ``open``/``close`` re-anchored to the live joint).

    kind="world": ``xyz`` is a fixed world point (the legacy behaviour).
    kind="articulated": the point is attached to articulated asset ``parent``;
        its world position is ``parent_pose ∘ local_offset`` carried along task
        ``joint`` by the joint's live travel — so e.g. a drawer-tray center
        tracks the drawer as it opens/closes.
    kind="articulated_opening": the point is the live midpoint of asset
        ``parent``'s opening (a named ``place_targets`` entry in its metadata):
        ``lip + open_dir * (live_slide * 0.5)`` at tray-floor height.
        Geometry lives in the asset metadata, so the scene carries no numbers.
    kind="placement": a named static placement affordance (``place_targets``
        entry with ``surface_local``) on any asset — e.g. a shelf surface on a
        fixture. Resolves to ``parent_pose ∘ surface_local`` with no joint, so the
        drop point tracks the asset's pose (incl. yaw) without per-scene numbers.

    ``approach`` is the placement/grasp **insert direction** (unit, the EE moves
    along ``+approach`` from standoff to the point; default ``(0,0,-1)`` = top-down,
    consistent with grasp's frame Z). For asset-anchored kinds it is declared in the
    asset frame and rotated to world by ``resolve_approach``; for a placement
    affordance it may instead live on the ``place_targets`` entry. ``None`` → default.
    """

    kind: str  # "world" | "articulated" | "articulated_opening" | "placement"
    xyz: tuple[float, float, float] | None = None
    parent: str | None = None
    joint: str | None = None
    local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    opening: str | None = None  # articulated_opening / placement: place_targets name
    approach: tuple[float, float, float] | None = None
