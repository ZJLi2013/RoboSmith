"""Small authoring helpers for CAP scene specs."""

from __future__ import annotations

from robotsmith.cap.specs import (
    FrameRef,
    LayoutSpec,
    ObjectSpec,
    RegionSpec,
    SceneSpec,
    TargetPositionSpec,
)


def on_articulated(
    asset: str,
    joint: str,
    *,
    local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> FrameRef:
    """Anchor a target to an articulated asset's joint (e.g. a drawer tray).

    The runtime resolves the world point each segment from the asset's live pose
    plus the joint's live travel, so the target tracks the part as it moves.

    Example:
        target_position(
            "drawer_open_slot",
            anchor=on_articulated("drawer", "drawer_slide",
                                  local_offset=(0.035, 0.0, 0.136)),
        )
    """

    return FrameRef(
        kind="articulated",
        parent=asset,
        joint=joint,
        local_offset=local_offset,
    )


def on_articulated_opening(asset: str, opening: str = "drawer_opening") -> FrameRef:
    """Anchor a place target to the live midpoint of an articulated opening.

    Unlike ``on_articulated`` (a fixed point that rides the full joint travel and
    retracts under the carcass when the part recoils), this resolves to the
    *currently exposed* opening's midpoint — ``lip + open_dir*(live_slide*frac)``
    at tray-floor height. All geometry (lip, tray floor, travel fraction) comes
    from the asset's ``place_targets`` metadata, so the scene carries no numbers.

    Example::

        target_position("drawer_open_slot",
                        anchor=on_articulated_opening("drawer", "drawer_opening"))
    """

    return FrameRef(kind="articulated_opening", parent=asset, opening=opening)


def region(
    name: str,
    xy_bounds: tuple[tuple[float, float], tuple[float, float]],
    *,
    z: float | None = None,
    min_distance: float | None = None,
    keep_out_radius: float | None = None,
) -> RegionSpec:
    """Create a named tabletop region for agent-friendly layout authoring.

    Example:
        region("left_reachable", ((0.42, -0.14), (0.50, -0.06)))
    """

    return RegionSpec(
        name=name,
        xy_bounds=xy_bounds,
        z=z,
        min_distance=min_distance,
        keep_out_radius=keep_out_radius,
    )


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
    pose: str = "upright",
    region: str | None = None,
    fixed_position: tuple[float, float, float] | None = None,
    joint_init: dict[str, float] | None = None,
) -> ObjectSpec:
    """Create a physical object spec.

    ``joint_init`` (articulated assets) sets per-scenario initial joint state,
    e.g. ``obj("drawer", asset="drawer_cabinet", joint_init={"drawer_slide": 0.35})``
    for a frozen-open drawer; it overrides the asset metadata default on reset.

    Example:
        obj("die", asset="die_01", region="left_reachable")
    """

    return ObjectSpec(
        name=name,
        asset=asset,
        pose=pose,
        region=region,
        fixed_position=fixed_position,
        joint_init=dict(joint_init) if joint_init else {},
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
    anchored to a moving part via ``anchor`` (see ``on_articulated``); a moving
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
    """Create a CAP scene authoring artifact.

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
