"""Small authoring helpers for scenario scene specs."""

from __future__ import annotations

from robotsmith.scenario.specs import (
    FrameRef,
    LayoutSpec,
    ObjectSpec,
    RegionSpec,
    SceneSpec,
    TargetPositionSpec,
)


def on_placement(asset: str, name: str) -> FrameRef:
    """Anchor a target to a named static placement affordance on any asset.

    The named ``place_targets`` entry (e.g. a shelf surface on a fixture) carries its own
    local drop point — and optionally its approach direction — in the asset frame.
    The runtime resolves both against the asset's live pose each segment, so the
    drop point and approach track the asset (including yaw) with no per-scene
    numbers. Use this instead of a hand-computed ``fixed_position`` world marker.

    Example::

        target_position("die_target", anchor=on_placement("supporter", "shelf_lower"))
    """

    return FrameRef(kind="placement", parent=asset, opening=name)


def region(
    name: str,
    xy_bounds: tuple[tuple[float, float], tuple[float, float]],
    *,
    z: float | None = None,
) -> RegionSpec:
    """Create a named tabletop region for agent-friendly layout authoring.

    Example:
        region("left_reachable", ((0.42, -0.14), (0.50, -0.06)))
    """

    return RegionSpec(name=name, xy_bounds=xy_bounds, z=z)


def layout(
    *,
    workspace: str = "franka_tabletop",
    regions: list[RegionSpec] | tuple[RegionSpec, ...] = (),
) -> LayoutSpec:
    """Create a layout from named regions.

    Example:
        layout(regions=[region("left_reachable", ((0.42, -0.14), (0.50, -0.06)))])
    """

    return LayoutSpec(workspace=workspace, regions={r.name: r for r in regions})


def obj(
    name: str,
    *,
    asset: str,
    region: str | None = None,
    fixed_position: tuple[float, float, float] | None = None,
    joint_init: dict[str, float] | None = None,
    yaw_deg: float = 0.0,
) -> ObjectSpec:
    """Create a physical object spec.

    ``joint_init`` (articulated assets) sets per-scenario initial joint state,
    e.g. ``obj("drawer", asset="drawer_cabinet", joint_init={"drawer_slide": 0.35})``
    for a frozen-open drawer; it overrides the asset metadata default on reset.

    ``yaw_deg`` rotates the placed object about world +Z on top of its upright
    pose (positive = CCW from above, negative = clockwise) — e.g. turn a shelf
    fixture so its open side faces the arm without editing shared asset metadata.

    Example:
        obj("die", asset="die_01", region="left_reachable")
    """

    return ObjectSpec(
        name=name,
        asset=asset,
        region=region,
        fixed_position=fixed_position,
        joint_init=dict(joint_init) if joint_init else {},
        yaw_deg=yaw_deg,
    )


def target_position(
    name: str,
    *,
    region: str | None = None,
    fixed_position: tuple[float, float, float] | None = None,
    anchor: FrameRef | None = None,
) -> TargetPositionSpec:
    """Create a non-physical target position referenced by skills/predicates.

    A point is either a static world value (``fixed_position``/``region``) or
    anchored to a moving part via ``anchor`` (a ``FrameRef``); a moving
    part requires an explicit anchor, so a constant can't silently ride on it.

    Example:
        target_position("goal", fixed_position=(0.60, 0.10, 0.809))
    """

    return TargetPositionSpec(
        name=name,
        region=region,
        fixed_position=fixed_position,
        anchor=anchor,
    )


def scene(
    name: str,
    *,
    layout: LayoutSpec,
    objects: list[ObjectSpec] | tuple[ObjectSpec, ...],
    target_positions: list[TargetPositionSpec] | tuple[TargetPositionSpec, ...] = (),
    camera_position: tuple[float, float, float] | None = None,
    camera_target: tuple[float, float, float] | None = None,
    table_size: tuple[float, float, float] | None = None,
    table_height: float | None = None,
) -> SceneSpec:
    """Create a scenario scene authoring artifact.

    Optional ``camera_position`` / ``camera_target`` override the overview camera
    pose for this scene (e.g. to frame a tall articulated cabinet); when omitted
    the simulator-agnostic SceneConfig default is used.

    Optional ``table_size`` ([x, y, z] m) / ``table_height`` override the support
    table extent (e.g. to extend it over an off-to-the-side anchored cabinet);
    when omitted the SceneConfig default (1.2 x 0.8, height 0.75) is used.

    Example:
        scene(
            "cap_pick_die",
            layout=layout(regions=[
                region("left_reachable", ((0.42, -0.14), (0.50, -0.06))),
            ]),
            objects=[obj("die", asset="die_01", region="left_reachable")],
        )
    """

    return SceneSpec(
        name=name,
        layout=layout,
        objects=tuple(objects),
        target_positions=tuple(target_positions),
        camera_position=camera_position,
        camera_target=camera_target,
        table_size=table_size,
        table_height=table_height,
    )
