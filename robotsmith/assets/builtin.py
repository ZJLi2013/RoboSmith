"""Bootstrap built-in sim-ready primitive assets.

Generates URDF files using simple geometric primitives (box, cylinder, sphere)
so they work in both PyBullet and Genesis without external mesh files.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from robotsmith.assets.schema import Asset, AssetMetadata

PRIMITIVE_DEFS: list[dict] = [
    {
        "name": "mug_red",
        "geometry": "cylinder",
        "radius": 0.04,
        "length": 0.12,
        "color": [0.8, 0.15, 0.1, 1.0],
        "mass_kg": 0.25,
        "friction": 0.6,
        "size_cm": [8, 8, 12],
        "tags": ["mug", "cup", "red", "container", "grasp"],
        "description": "Red ceramic mug",
    },
    {
        "name": "bowl_white",
        "geometry": "sphere_half",
        "radius": 0.075,
        "color": [0.95, 0.95, 0.92, 1.0],
        "mass_kg": 0.3,
        "friction": 0.5,
        "size_cm": [15, 15, 7],
        "tags": ["bowl", "white", "container", "grasp"],
        "description": "White ceramic bowl",
    },
    {
        "name": "plate_round",
        "geometry": "cylinder",
        "radius": 0.11,
        "length": 0.02,
        "color": [0.92, 0.92, 0.88, 1.0],
        "mass_kg": 0.35,
        "friction": 0.5,
        "size_cm": [22, 22, 2],
        "tags": ["plate", "dish", "round", "flat", "grasp"],
        "description": "Round ceramic plate",
    },
    {
        "name": "block_red",
        "geometry": "box",
        "size": [0.05, 0.05, 0.05],
        "color": [0.9, 0.2, 0.15, 1.0],
        "mass_kg": 0.05,
        "friction": 0.6,
        "size_cm": [5, 5, 5],
        "tags": ["block", "cube", "red", "stackable", "grasp"],
        "description": "Red wooden block",
    },
    {
        "name": "block_blue",
        "geometry": "box",
        "size": [0.05, 0.05, 0.05],
        "color": [0.15, 0.3, 0.85, 1.0],
        "mass_kg": 0.05,
        "friction": 0.6,
        "size_cm": [5, 5, 5],
        "tags": ["block", "cube", "blue", "stackable", "grasp"],
        "description": "Blue wooden block",
    },
    {
        "name": "block_green",
        "geometry": "box",
        "size": [0.05, 0.05, 0.05],
        "color": [0.2, 0.75, 0.25, 1.0],
        "mass_kg": 0.05,
        "friction": 0.6,
        "size_cm": [5, 5, 5],
        "tags": ["block", "cube", "green", "stackable", "grasp"],
        "description": "Green wooden block",
    },
    {
        "name": "bottle_tall",
        "geometry": "cylinder",
        "radius": 0.035,
        "length": 0.22,
        "color": [0.3, 0.6, 0.35, 0.8],
        "mass_kg": 0.4,
        "friction": 0.4,
        "size_cm": [7, 7, 22],
        "tags": ["bottle", "tall", "green", "container", "grasp", "pour"],
        "description": "Tall glass bottle",
    },
    {
        "name": "can_soda",
        "geometry": "cylinder",
        "radius": 0.033,
        "length": 0.12,
        "color": [0.85, 0.1, 0.1, 1.0],
        "mass_kg": 0.35,
        "friction": 0.45,
        "size_cm": [6.6, 6.6, 12],
        "tags": ["can", "soda", "red", "cylinder", "grasp"],
        "description": "Red soda can",
    },
    {
        "name": "fork_silver",
        "geometry": "box",
        "size": [0.01, 0.005, 0.19],
        "color": [0.75, 0.75, 0.78, 1.0],
        "mass_kg": 0.04,
        "friction": 0.35,
        "size_cm": [1, 0.5, 19],
        "tags": ["fork", "silver", "utensil", "thin", "grasp"],
        "description": "Silver fork",
    },
    {
        "name": "spoon_silver",
        "geometry": "box",
        "size": [0.015, 0.005, 0.18],
        "color": [0.75, 0.75, 0.78, 1.0],
        "mass_kg": 0.035,
        "friction": 0.35,
        "size_cm": [1.5, 0.5, 18],
        "tags": ["spoon", "silver", "utensil", "thin", "grasp"],
        "description": "Silver spoon",
    },
]


def _box_inertia(mass: float, sx: float, sy: float, sz: float) -> tuple[float, float, float]:
    ixx = mass / 12.0 * (sy * sy + sz * sz)
    iyy = mass / 12.0 * (sx * sx + sz * sz)
    izz = mass / 12.0 * (sx * sx + sy * sy)
    return ixx, iyy, izz


def _cylinder_inertia(mass: float, radius: float, length: float) -> tuple[float, float, float]:
    ixx = mass / 12.0 * (3 * radius * radius + length * length)
    iyy = ixx
    izz = mass / 2.0 * radius * radius
    return ixx, iyy, izz


def _sphere_inertia(mass: float, radius: float) -> tuple[float, float, float]:
    i = 2.0 / 5.0 * mass * radius * radius
    return i, i, i


def _generate_urdf(defn: dict) -> str:
    """Generate a URDF string for a primitive geometry definition."""
    name = defn["name"]
    geom = defn["geometry"]
    mass = defn["mass_kg"]
    r, g, b, a = defn["color"]

    if geom == "box":
        sx, sy, sz = defn["size"]
        ixx, iyy, izz = _box_inertia(mass, sx, sy, sz)
        geom_xml = f'<box size="{sx} {sy} {sz}"/>'
        origin_z = sz / 2.0
    elif geom == "cylinder":
        radius = defn["radius"]
        length = defn["length"]
        ixx, iyy, izz = _cylinder_inertia(mass, radius, length)
        geom_xml = f'<cylinder radius="{radius}" length="{length}"/>'
        origin_z = length / 2.0
    elif geom == "sphere_half":
        radius = defn["radius"]
        ixx, iyy, izz = _sphere_inertia(mass, radius)
        geom_xml = f'<sphere radius="{radius}"/>'
        origin_z = radius * 0.3
    else:
        raise ValueError(f"Unknown geometry: {geom}")

    return f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 {origin_z:.6f}" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="{ixx:.8f}" ixy="0" ixz="0" iyy="{iyy:.8f}" iyz="0" izz="{izz:.8f}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 {origin_z:.6f}" rpy="0 0 0"/>
      <geometry>{geom_xml}</geometry>
      <material name="{name}_mat">
        <color rgba="{r} {g} {b} {a}"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 {origin_z:.6f}" rpy="0 0 0"/>
      <geometry>{geom_xml}</geometry>
    </collision>
  </link>
</robot>
"""


def _generate_table_urdf() -> str:
    """A simple table: a box top on 4 legs, all as a single fixed link."""
    return """<?xml version="1.0"?>
<robot name="table_simple">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0.375" rpy="0 0 0"/>
      <mass value="15.0"/>
      <inertia ixx="0.5" ixy="0" ixz="0" iyy="0.3" iyz="0" izz="0.6"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.375" rpy="0 0 0"/>
      <geometry><box size="0.8 0.6 0.75"/></geometry>
      <material name="table_mat"><color rgba="0.55 0.35 0.2 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0.75" rpy="0 0 0"/>
      <geometry><box size="0.8 0.6 0.05"/></geometry>
    </collision>
    <collision>
      <origin xyz="0.35 0.25 0.375" rpy="0 0 0"/>
      <geometry><cylinder radius="0.025" length="0.75"/></geometry>
    </collision>
    <collision>
      <origin xyz="-0.35 0.25 0.375" rpy="0 0 0"/>
      <geometry><cylinder radius="0.025" length="0.75"/></geometry>
    </collision>
    <collision>
      <origin xyz="0.35 -0.25 0.375" rpy="0 0 0"/>
      <geometry><cylinder radius="0.025" length="0.75"/></geometry>
    </collision>
    <collision>
      <origin xyz="-0.35 -0.25 0.375" rpy="0 0 0"/>
      <geometry><cylinder radius="0.025" length="0.75"/></geometry>
    </collision>
  </link>
</robot>
"""


def _generate_plane_urdf() -> str:
    """Ground plane."""
    return """<?xml version="1.0"?>
<robot name="plane">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="10 10 0.001"/></geometry>
      <material name="plane_mat"><color rgba="0.85 0.85 0.82 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="10 10 0.001"/></geometry>
    </collision>
  </link>
</robot>
"""


def bootstrap_builtin_assets(assets_root: Path) -> list[Asset]:
    """Generate all built-in primitive assets on disk and return Asset objects."""
    objects_dir = assets_root / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)
    created: list[Asset] = []

    for defn in PRIMITIVE_DEFS:
        asset_dir = objects_dir / defn["name"]
        asset_dir.mkdir(exist_ok=True)

        urdf_path = asset_dir / "model.urdf"
        urdf_path.write_text(_generate_urdf(defn), encoding="utf-8")

        meta = AssetMetadata(
            mass_kg=defn["mass_kg"],
            friction=defn["friction"],
            size_cm=defn["size_cm"],
            tags=defn["tags"],
            source="builtin_primitive",
            description=defn["description"],
        )
        meta.save(asset_dir / "metadata.json")

        created.append(Asset(
            name=defn["name"],
            root_dir=asset_dir,
            urdf_path=urdf_path,
            metadata=meta,
        ))

    # Table
    table_dir = objects_dir / "table_simple"
    table_dir.mkdir(exist_ok=True)
    urdf = table_dir / "model.urdf"
    urdf.write_text(_generate_table_urdf(), encoding="utf-8")
    table_meta = AssetMetadata(
        mass_kg=15.0, friction=0.5, size_cm=[80, 60, 75],
        tags=["table", "furniture", "surface"], source="builtin_primitive",
        description="Simple wooden table (80x60cm, 75cm tall)",
    )
    table_meta.save(table_dir / "metadata.json")
    created.append(Asset(name="table_simple", root_dir=table_dir, urdf_path=urdf, metadata=table_meta))

    # Ground plane
    plane_dir = objects_dir / "plane"
    plane_dir.mkdir(exist_ok=True)
    urdf = plane_dir / "model.urdf"
    urdf.write_text(_generate_plane_urdf(), encoding="utf-8")
    plane_meta = AssetMetadata(
        mass_kg=0.0, friction=0.5, size_cm=[1000, 1000, 0.1],
        tags=["plane", "ground", "floor", "static"], source="builtin_primitive",
        description="Ground plane (10x10m)",
    )
    plane_meta.save(plane_dir / "metadata.json")
    created.append(Asset(name="plane", root_dir=plane_dir, urdf_path=urdf, metadata=plane_meta))

    return created
