"""Typed CAP authoring specs for scene/task definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

RegionName = str


@dataclass(frozen=True)
class RegionSpec:
    """Named tabletop region for object or target placement.

    Args:
        name: Logical region name, e.g. "left_reachable".
        xy_bounds: XY min/max corners in world/table coordinates.
        z: Optional placement Z used when lowering to legacy ranges.
        min_distance: Optional semantic spacing hint between placements.
        keep_out_radius: Optional semantic keep-out radius for goal/container
            areas; not a raw Genesis/runtime knob.
    """

    name: str
    xy_bounds: tuple[tuple[float, float], tuple[float, float]]
    z: float | None = None
    min_distance: float | None = None
    keep_out_radius: float | None = None


@dataclass(frozen=True)
class LayoutSpec:
    """Workspace and named regions for a scene.

    Args:
        workspace: Workspace preset name, currently "franka_tabletop" by
            default.
        regions: Mapping from region name to RegionSpec. Objects and targets
            reference these names instead of repeating raw placement bounds.
    """

    workspace: str = "franka_tabletop"
    regions: dict[str, RegionSpec] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectSpec:
    """Physical object in a CAP scene.

    Args:
        name: Logical object name used by instructions, predicates, and later
            skills.
        asset: Asset query or asset id in the RoboSmith asset catalog.
        pose: Asset metadata pose key. Current authoring helpers expect
            "upright".
        region: Optional named region reference for semantic placement.
        fixed_position: Optional exact XYZ world position for fixture/debug
            scenes. region and fixed_position are layout hints, not task
            semantics.
        joint_init: Articulated assets only. Per-scenario initial joint state
            {joint: qpos} (e.g. {"drawer_slide": 0.35} for a frozen-open
            drawer), overriding the asset metadata default on reset. Empty =
            use the asset default.
        yaw_deg: Per-scenario rotation (degrees) about world +Z applied on top
            of the asset's canonical upright pose. Positive = counter-clockwise
            seen from above; negative = clockwise. Lets a scene orient a fixture
            (e.g. face an open shelf toward the arm) without editing the asset's
            shared upright metadata.
    """

    name: str
    asset: str
    pose: str = "upright"
    region: RegionName | None = None
    fixed_position: tuple[float, float, float] | None = None
    joint_init: dict[str, float] = field(default_factory=dict)
    yaw_deg: float = 0.0


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
        ``lip + open_dir * (live_slide * travel_fraction)`` at tray-floor height.
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


@dataclass(frozen=True)
class TargetPositionSpec:
    """Named non-physical target position used by success predicates.

    Args:
        name: Logical target name, e.g. "goal".
        region: Optional named region reference used to resolve a target point.
        fixed_position: Optional exact XYZ target position.
        anchor: Optional self-describing FrameRef. When set (e.g. attached to an
            articulated container), the runtime resolves the point against the
            live world each segment instead of trusting a static world value.

    Target positions are not simulator objects. Legacy predicates may need the
    adapter to materialize them into predicate-readable env_state.
    """

    name: str
    region: RegionName | None = None
    fixed_position: tuple[float, float, float] | None = None
    anchor: FrameRef | None = None


@dataclass(frozen=True)
class SceneSpec:
    """CAP scene authoring artifact.

    Args:
        name: Scene identifier used by adapters and generated legacy configs.
        layout: Workspace and named region definitions.
        objects: Physical objects that should appear in the simulator.
        target_positions: Optional non-physical targets referenced by success
            predicates.

    SceneSpec does not contain language instructions, success conditions, or
    robot action policy.
    """

    name: str
    layout: LayoutSpec
    objects: tuple[ObjectSpec, ...]
    target_positions: tuple[TargetPositionSpec, ...] = ()
    camera_position: tuple[float, float, float] | None = None
    camera_target: tuple[float, float, float] | None = None
    table_size: tuple[float, float, float] | None = None
    table_height: float | None = None


@dataclass(frozen=True)
class TaskSuccessCfg:
    """Typed success predicate leaf for authoring.

    Args:
        predicate: Authoring-level predicate name.
        params: Structured predicate parameters using logical object/target
            names.

    A single leaf in the success tree. Combine leaves with AllOf / AnyOf / Not.
    Lowers to a runtime SuccessNode(op="leaf", ...).

    refs are the logical object/target names this leaf references (including
    fixed references like "table"). The helper that builds the leaf declares
    them, so ref extraction no longer needs a per-predicate switch.
    """

    predicate: str
    params: dict[str, Any]
    refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class AllOf:
    """Success when all terms hold (AND)."""

    terms: tuple["SuccessSpec", ...]


@dataclass(frozen=True)
class AnyOf:
    """Success when any term holds (OR)."""

    terms: tuple["SuccessSpec", ...]


@dataclass(frozen=True)
class Not:
    """Success when the inner term does not hold (NOT)."""

    term: "SuccessSpec"


SuccessSpec = Union[TaskSuccessCfg, AllOf, AnyOf, Not]


@dataclass(frozen=True)
class TaskSpec:
    """CAP task authoring artifact.

    Args:
        name: Task identifier.
        scene: SceneSpec bound to this task.
        instruction: Language instruction for the task.
        success: Success condition; a single predicate leaf (TaskSuccessCfg) or
            a composite tree built from AllOf / AnyOf / Not.

    TaskSpec intentionally excludes expert action sequences such as pick/place
    steps; those belong to a separate policy/action layer.
    """

    name: str
    scene: SceneSpec
    instruction: str
    success: "SuccessSpec"
