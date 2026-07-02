"""Asset and metadata schema for sim-ready objects."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


NO_POSITIVE_GRASP_POLICY_BUCKET = "no_positive"
"""Sentinel for assets whose full Stage 1 bucket probe found no positive bucket."""

# URDF joint types that carry a controllable / movable degree of freedom.
_MOVABLE_JOINT_TYPES = frozenset({"prismatic", "revolute", "continuous"})


@dataclass(frozen=True)
class UrdfJoint:
    """One kinematic joint parsed directly from an asset's ``model.urdf``.

    The URDF is the single source of truth for joint kinematics; this struct is
    read at load time and never persisted into ``metadata.json``.
    """

    name: str
    type: str
    parent: str
    child: str
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    lower: float | None = None
    upper: float | None = None

    @property
    def is_movable(self) -> bool:
        return self.type in _MOVABLE_JOINT_TYPES


def parse_urdf_joints(urdf_path: Path) -> list[UrdfJoint]:
    """Parse all ``<joint>`` elements (type/axis/limits/parent/child) from a URDF."""
    root = ET.parse(urdf_path).getroot()
    joints: list[UrdfJoint] = []
    for j in root.findall("joint"):
        axis_el = j.find("axis")
        axis = (1.0, 0.0, 0.0)
        if axis_el is not None and axis_el.get("xyz"):
            parts = [float(v) for v in axis_el.get("xyz").split()]
            if len(parts) == 3:
                axis = (parts[0], parts[1], parts[2])
        limit_el = j.find("limit")
        lower = upper = None
        if limit_el is not None:
            lower = float(limit_el.get("lower")) if limit_el.get("lower") is not None else None
            upper = float(limit_el.get("upper")) if limit_el.get("upper") is not None else None
        parent_el = j.find("parent")
        child_el = j.find("child")
        joints.append(
            UrdfJoint(
                name=j.get("name", ""),
                type=j.get("type", ""),
                parent=parent_el.get("link", "") if parent_el is not None else "",
                child=child_el.get("link", "") if child_el is not None else "",
                axis=axis,
                lower=lower,
                upper=upper,
            )
        )
    return joints


@dataclass(frozen=True)
class UrdfCollisionBox:
    """One axis-aligned collision box parsed from a URDF link, in the link frame.

    Spheres/cylinders are approximated by their bounding box so a box-only
    collision world (rocRobo ``world`` supports box + halfspace) can consume
    them conservatively. ``size`` is the full box size (not half-extent).
    """

    link: str
    size: tuple[float, float, float]
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # ``<collision name="...">`` of the source element ("" if unnamed). Lets the
    # collision world exempt a single box (e.g. a place target ``shelf_lower``)
    # on a multi-box single-link fixture, which per-link exemption cannot isolate.
    name: str = ""


def _origin_of(el) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    origin = el.find("origin")
    xyz = (0.0, 0.0, 0.0)
    rpy = (0.0, 0.0, 0.0)
    if origin is not None:
        if origin.get("xyz"):
            p = [float(v) for v in origin.get("xyz").split()]
            if len(p) == 3:
                xyz = (p[0], p[1], p[2])
        if origin.get("rpy"):
            r = [float(v) for v in origin.get("rpy").split()]
            if len(r) == 3:
                rpy = (r[0], r[1], r[2])
    return xyz, rpy


def parse_urdf_collision_boxes(urdf_path: Path) -> list[UrdfCollisionBox]:
    """Parse every link's ``<collision>`` geometry into per-link boxes.

    Boxes keep their size; spheres become a cube of side ``2*radius``; cylinders
    become a box of ``(2r, 2r, length)``. Geometry types without a clear box
    bound are skipped.
    """
    root = ET.parse(urdf_path).getroot()
    boxes: list[UrdfCollisionBox] = []
    for link in root.findall("link"):
        link_name = link.get("name", "")
        for col in link.findall("collision"):
            geom = col.find("geometry")
            if geom is None:
                continue
            xyz, rpy = _origin_of(col)
            box_name = col.get("name", "") or ""
            box_el = geom.find("box")
            sph_el = geom.find("sphere")
            cyl_el = geom.find("cylinder")
            if box_el is not None and box_el.get("size"):
                s = [float(v) for v in box_el.get("size").split()]
                if len(s) == 3:
                    boxes.append(
                        UrdfCollisionBox(link_name, (s[0], s[1], s[2]), xyz, rpy, box_name)
                    )
            elif sph_el is not None and sph_el.get("radius") is not None:
                r = float(sph_el.get("radius"))
                boxes.append(
                    UrdfCollisionBox(link_name, (2 * r, 2 * r, 2 * r), xyz, rpy, box_name)
                )
            elif cyl_el is not None:
                r = float(cyl_el.get("radius", "0") or 0.0)
                length = float(cyl_el.get("length", "0") or 0.0)
                boxes.append(
                    UrdfCollisionBox(link_name, (2 * r, 2 * r, length), xyz, rpy, box_name)
                )
    return boxes


@dataclass
class AssetMetadata:
    """Physics and catalog metadata for a sim-ready asset."""

    mass_kg: float = 0.1
    friction: float = 0.5
    restitution: float = 0.1
    density_kg_m3: float = 1000.0
    size_cm: list[float] = field(default_factory=lambda: [5.0, 5.0, 5.0])
    metric_scale: float = 1.0
    """Fixed asset-level scale applied when loading this asset into metric scenes.

    This calibrates raw/generated meshes to the Franka/table world scale. It is
    intentionally asset-owned: every scene sees the same asset at the same
    physical size.
    """
    source: str = "builtin"
    role: str = "object"
    """Asset role: ``"object"`` (robot-manipulated, dynamic, grasp-planned) or
    ``"fixture"`` (static environment prop, loaded with a fixed base, not
    grasp-planned). Orthogonal to geometry (mesh vs primitive). Articulated
    assets are fixtures regardless of this field (see ``Asset.is_fixture``)."""
    description: str = ""
    stable_poses: list[dict] = field(default_factory=list)
    """Pre-computed stable resting poses on a flat surface.
    Each entry: {"z": float, "quat": [w, x, y, z]}.
    Empty list means not yet computed (fallback: upright with z = half-height)."""
    canonical_frame: dict = field(default_factory=dict)
    """How the raw mesh frame maps into RoboSmith's semantic object frame.

    Reserved for import-time canonicalization metadata such as unit, up axis,
    origin convention, and T_object_mesh. Existing assets may leave this empty.
    """
    task_poses: dict = field(default_factory=dict)
    """Named semantic poses in the canonical object frame.

    Example:
      {"upright": {"quat_object": [w, x, y, z], "description": "..."}}
    Scene placement consumes these instead of guessing from mesh identity.
    """
    grasp_policy_bucket: str | list[str] | None = None
    """Validated Stage 1 bucket(s), or ``no_positive`` after a full failed probe."""
    grasp_strategy: str | None = None
    """Optional per-asset override of how a *pickable* grasp pose is acquired:
    ``"learned"`` (GraspGen) or ``"none"`` (skip grasp-planning this object).
    ``None`` means use the run-level default (learned). Articulated assets ignore
    this — they are moved by joint primitives, not grasp-planned. Reserved: leave
    unset until a scene genuinely needs a non-default strategy for one object."""
    mesh: dict = field(default_factory=dict)
    """Compact asset mesh audit cached by ``robotsmith.assets.audit``."""
    task_joints: list[str] = field(default_factory=list)
    """Articulated assets only: URDF joint names that are task-relevant (e.g. the
    drawer slide). Empty for rigid assets. Joint kinematics stay in the URDF; this
    only marks *which* joints carry task semantics."""
    handles: list[dict] = field(default_factory=list)
    """Articulated assets only: graspable parts for open/close, each
    ``{"name": str, "link": str}``. Empty for rigid assets."""
    place_targets: list[dict] = field(default_factory=list)
    """Named placement affordances on **any** asset (fixtures included, not just
    articulated): where a ``place`` drops into/onto the asset, derived from asset
    geometry (no per-scene magic numbers). Two shapes:

    - **Joint opening** (articulated, e.g. a drawer tray)::

        {"name": "drawer_opening", "joint": "drawer_slide",
         "lip_local": [x, y, z], "tray_floor_z": z, "travel_fraction": 0.5}

      ``lip_local`` is the front opening edge in the asset (URDF) frame at tray-floor
      height; ``tray_floor_z`` the cavity floor top. Live drop point =
      ``lip + open_dir * (live_slide * travel_fraction)`` — the exposed-opening
      midpoint — so it tracks the part as it recoils. (kind ``articulated_opening``.)

    - **Static surface** (e.g. a fixture shelf)::

        {"name": "shelf_lower", "surface_local": [x, y, z],
         "extent_xy": [dx, dy], "approach": [ax, ay, az]}

      ``surface_local`` is the support-surface point in the asset frame; the drop
      point = ``parent_pose ∘ surface_local`` (tracks pose incl. yaw, no joint).
      (kind ``placement``.)

    All offsets are in the asset frame and scaled by ``metric_scale`` at resolve
    time. Optional ``approach`` (unit insert direction in the asset frame, default
    top-down ``-Z``) is rotated to world by ``resolve_approach`` — e.g. ``[-1,0,0]``
    for a shelf whose open side faces local ``+X`` (side tuck-under)."""
    joint_init: dict = field(default_factory=dict)
    """Articulated assets only: default initial joint state ``{joint: qpos}``
    applied on reset (e.g. ``{"drawer_slide": 0.0}`` = closed). Empty for rigid."""

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> AssetMetadata:
        data = json.loads(path.read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Asset:
    """A sim-ready asset with URDF, meshes, and metadata."""

    name: str
    root_dir: Path
    urdf_path: Path
    metadata: AssetMetadata
    visual_mesh: Optional[Path] = None
    collision_mesh: Optional[Path] = None
    joints: list[UrdfJoint] = field(default_factory=list)
    """Joints parsed from ``model.urdf`` at load. Empty for single-link rigid assets."""

    @property
    def is_articulated(self) -> bool:
        """True when the URDF declares at least one movable joint."""
        return any(j.is_movable for j in self.joints)

    @property
    def uses_mesh_geometry(self) -> bool:
        """True when the asset carries visual/collision mesh files (mesh-based
        geometry the mesh audit applies to). Primitive-only URDFs have neither."""
        return self.visual_mesh is not None or self.collision_mesh is not None

    @property
    def is_fixture(self) -> bool:
        """True for static environment props: loaded with a fixed base and never
        grasp-planned. Articulated assets are always fixtures (anchored base)."""
        return self.metadata.role == "fixture" or self.is_articulated

    @property
    def movable_joints(self) -> list[UrdfJoint]:
        return [j for j in self.joints if j.is_movable]

    @property
    def primary_moving_link(self) -> Optional[str]:
        """Child link of the first movable joint (the part that rides the joint),
        or ``None`` for a rigid asset. Single source of truth for "which link
        moves" — currently first-joint only (multi-DOF is a future item)."""
        movable = self.movable_joints
        return movable[0].child if movable else None

    def get_joint(self, name: str) -> Optional[UrdfJoint]:
        for j in self.joints:
            if j.name == name:
                return j
        return None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "root_dir": str(self.root_dir),
            "urdf_path": str(self.urdf_path),
            "visual_mesh": str(self.visual_mesh) if self.visual_mesh else None,
            "collision_mesh": str(self.collision_mesh) if self.collision_mesh else None,
            "metadata": asdict(self.metadata),
        }

    def __repr__(self) -> str:
        kind = "articulated" if self.is_articulated else "rigid"
        return f"Asset({self.name!r}, {kind}, mass={self.metadata.mass_kg}kg)"
