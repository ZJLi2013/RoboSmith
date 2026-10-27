"""Geometry derived from an asset's build files (URDF + meshes) at load time.

Sizes, handle points and placement surfaces are read from the URDF at zero joint
configuration instead of being stored in ``asset.json``; annotations only name
which link / collision element carries the semantics.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from robotsmith.assets.schema import (
    UrdfCollisionBox,
    UrdfJoint,
    _origin_of,
    parse_urdf_collision_boxes,
)


def _transform(xyz, rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    t = np.eye(4)
    t[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    t[:3, 3] = xyz
    return t


def link_transforms(urdf_path: Path) -> dict[str, np.ndarray]:
    """Each link's pose in the asset (root link) frame at zero joint configuration."""
    root = ET.parse(urdf_path).getroot()
    parent_of = {
        j.find("child").get("link"): (j.find("parent").get("link"), _transform(*_origin_of(j)))
        for j in root.findall("joint")
    }
    out: dict[str, np.ndarray] = {}

    def resolve(link: str) -> np.ndarray:
        if link not in out:
            parent, t = parent_of.get(link, (None, None))
            out[link] = np.eye(4) if parent is None else resolve(parent) @ t
        return out[link]

    for link in root.findall("link"):
        resolve(link.get("name", ""))
    return out


def inertial_origin(urdf_path: Path) -> np.ndarray:
    """Root link's inertial origin in the asset frame; Genesis reports a free object's position here."""
    root = ET.parse(urdf_path).getroot()
    children = {j.find("child").get("link") for j in root.findall("joint")}
    base = next(link for link in root.findall("link") if link.get("name", "") not in children)
    inertial = base.find("inertial")
    if inertial is None:
        return np.zeros(3)
    xyz, rpy = _origin_of(inertial)
    if any(rpy):
        raise ValueError(f"{urdf_path}: rotated inertial frame (rpy={rpy}) is not supported")
    return np.asarray(xyz, dtype=np.float64)


def _bounds(points: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    world = _apply(t, points)
    return world.min(axis=0), world.max(axis=0)


def box_bounds(box: UrdfCollisionBox, link_tf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half = np.asarray(box.size) / 2.0
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
    return _bounds(corners, link_tf @ _transform(box.origin_xyz, box.origin_rpy))


def visual_mesh(urdf_path: Path):
    """All visual meshes of the asset, in the asset frame, as one ``trimesh.Trimesh``."""
    import trimesh

    tfs = link_transforms(urdf_path)
    meshes = []
    for link in ET.parse(urdf_path).getroot().findall("link"):
        for vis in link.findall("visual"):
            mesh_el = vis.find("geometry/mesh")
            if mesh_el is not None:
                mesh = trimesh.load(str(urdf_path.parent / mesh_el.get("filename")), force="mesh")
                meshes.append(mesh.apply_transform(tfs[link.get("name", "")] @ _transform(*_origin_of(vis))))
    return trimesh.util.concatenate(meshes)


def visual_points(urdf_path: Path) -> np.ndarray:
    """Box corners and mesh hull vertices of all visual geometry in the asset frame, in metres."""
    import trimesh

    tfs = link_transforms(urdf_path)
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    parts = []
    for box in parse_urdf_collision_boxes(urdf_path, element="visual"):
        t = tfs[box.link] @ _transform(box.origin_xyz, box.origin_rpy)
        parts.append(_apply(t, signs * np.asarray(box.size) / 2.0))
    root = ET.parse(urdf_path).getroot()
    for link in root.findall("link"):
        for vis in link.findall("visual"):
            mesh_el = vis.find("geometry/mesh")
            if mesh_el is None:
                continue
            mesh = trimesh.load(str(urdf_path.parent / mesh_el.get("filename")), force="mesh")
            t = tfs[link.get("name", "")] @ _transform(*_origin_of(vis))
            parts.append(_apply(t, np.asarray(mesh.convex_hull.vertices)))
    if not parts:
        raise ValueError(f"{urdf_path} has no visual geometry")
    return np.concatenate(parts)


def _apply(t: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (t @ np.c_[points, np.ones(len(points))].T).T[:, :3]


def visual_bounds(urdf_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds of all visual geometry in the asset frame, in metres."""
    points = visual_points(urdf_path)
    return points.min(axis=0), points.max(axis=0)


def size_cm(urdf_path: Path) -> list[float]:
    """Axis-aligned extent of all visual geometry, in cm."""
    lo, hi = visual_bounds(urdf_path)
    return [round(float(v) * 100.0, 1) for v in hi - lo]


def _named_boxes(urdf_path: Path) -> dict[str, UrdfCollisionBox]:
    return {b.name: b for b in parse_urdf_collision_boxes(urdf_path) if b.name}


def handles(urdf_path: Path, specs: list[dict]) -> list[dict]:
    """``{link, collision?}`` -> ``{link, local_pos}``: the centre of the named
    collision element (default: the link's first one) in the asset frame."""
    tfs = link_transforms(urdf_path)
    boxes = parse_urdf_collision_boxes(urdf_path)
    out = []
    for spec in specs:
        name = spec.get("collision")
        box = next(b for b in boxes if b.link == spec["link"] and (name is None or b.name == name))
        lo, hi = box_bounds(box, tfs[box.link])
        out.append({"link": spec["link"], "local_pos": ((lo + hi) / 2.0).tolist()})
    return out


def place_targets(urdf_path: Path, specs: list[dict], joints: list[UrdfJoint]) -> list[dict]:
    """``{name, surface?, approach?}`` -> the top of collision box ``surface``
    (default ``name``) in the asset frame.

    A box on a link driven by a movable joint is an opening: ``joint`` and
    ``lip_local`` (the box's top edge furthest along the joint axis) are added.
    Otherwise it is a static surface: ``surface_local`` is the top centre.
    ``box`` is the box's ``[lo, hi]`` at zero joint configuration.
    """
    tfs = link_transforms(urdf_path)
    boxes = _named_boxes(urdf_path)
    joint_of = {j.child: j for j in joints if j.is_movable}
    out = []
    for spec in specs:
        box = boxes[spec.get("surface", spec["name"])]
        lo, hi = box_bounds(box, tfs[box.link])
        center, half = (lo + hi) / 2.0, (hi - lo) / 2.0
        target = {k: v for k, v in spec.items() if k != "surface"}
        target["box"] = [lo.tolist(), hi.tolist()]
        joint = joint_of.get(box.link)
        if joint is not None:
            axis = np.asarray(joint.axis, dtype=np.float64)
            axis = axis / np.linalg.norm(axis)
            lip = center + axis * float(np.abs(axis) @ half)
            lip[2] = hi[2]
            target["joint"] = joint.name
            target["lip_local"] = lip.tolist()
        else:
            target["surface_local"] = [float(center[0]), float(center[1]), float(hi[2])]
        out.append(target)
    return out
