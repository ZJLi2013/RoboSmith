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
    # --- blocks (3 color variants) ---
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
    # --- box (2 size variants, flat rectangular) ---
    {
        "name": "box_small",
        "geometry": "box",
        "size": [0.08, 0.06, 0.03],
        "color": [0.85, 0.75, 0.55, 1.0],
        "mass_kg": 0.08,
        "friction": 0.5,
        "size_cm": [8, 6, 3],
        "tags": ["box", "rectangular", "flat", "grasp"],
        "description": "Small cardboard box",
    },
    {
        "name": "box_large",
        "geometry": "box",
        "size": [0.12, 0.08, 0.04],
        "color": [0.7, 0.55, 0.4, 1.0],
        "mass_kg": 0.15,
        "friction": 0.5,
        "size_cm": [12, 8, 4],
        "tags": ["box", "rectangular", "flat", "grasp"],
        "description": "Large cardboard box",
    },
]

# L-block requires a composite URDF (two joined boxes), handled separately.
L_BLOCK_DEFS: list[dict] = [
    {
        "name": "lblock_yellow",
        "color": [0.95, 0.85, 0.2, 1.0],
        "mass_kg": 0.08,
        "friction": 0.6,
        "size_cm": [8, 4, 4],
        "tags": ["lblock", "L-shape", "yellow", "non-convex", "grasp"],
        "description": "Yellow L-shaped block",
        "arm_a": [0.08, 0.04, 0.04],
        "arm_b": [0.04, 0.04, 0.04],
        "offset_b": [0.02, 0.04, 0.0],
    },
    {
        "name": "lblock_purple",
        "color": [0.6, 0.3, 0.75, 1.0],
        "mass_kg": 0.10,
        "friction": 0.6,
        "size_cm": [10, 4, 4],
        "tags": ["lblock", "L-shape", "purple", "non-convex", "grasp"],
        "description": "Purple L-shaped block",
        "arm_a": [0.10, 0.04, 0.04],
        "arm_b": [0.04, 0.04, 0.04],
        "offset_b": [0.03, 0.04, 0.0],
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


def _generate_lblock_urdf(defn: dict) -> str:
    """Generate a URDF for an L-shaped block (two box links joined by fixed joint)."""
    name = defn["name"]
    mass = defn["mass_kg"]
    r, g, b, a = defn["color"]
    ax, ay, az = defn["arm_a"]
    bx, by, bz = defn["arm_b"]
    ox, oy, oz = defn["offset_b"]

    mass_a = mass * 0.6
    mass_b = mass * 0.4
    ixx_a, iyy_a, izz_a = _box_inertia(mass_a, ax, ay, az)
    ixx_b, iyy_b, izz_b = _box_inertia(mass_b, bx, by, bz)

    return f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="arm_a">
    <inertial>
      <origin xyz="0 0 {az/2:.6f}" rpy="0 0 0"/>
      <mass value="{mass_a}"/>
      <inertia ixx="{ixx_a:.8f}" ixy="0" ixz="0" iyy="{iyy_a:.8f}" iyz="0" izz="{izz_a:.8f}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 {az/2:.6f}" rpy="0 0 0"/>
      <geometry><box size="{ax} {ay} {az}"/></geometry>
      <material name="{name}_mat"><color rgba="{r} {g} {b} {a}"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 {az/2:.6f}" rpy="0 0 0"/>
      <geometry><box size="{ax} {ay} {az}"/></geometry>
    </collision>
  </link>
  <link name="arm_b">
    <inertial>
      <origin xyz="0 0 {bz/2:.6f}" rpy="0 0 0"/>
      <mass value="{mass_b}"/>
      <inertia ixx="{ixx_b:.8f}" ixy="0" ixz="0" iyy="{iyy_b:.8f}" iyz="0" izz="{izz_b:.8f}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 {bz/2:.6f}" rpy="0 0 0"/>
      <geometry><box size="{bx} {by} {bz}"/></geometry>
      <material name="{name}_mat"><color rgba="{r} {g} {b} {a}"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 {bz/2:.6f}" rpy="0 0 0"/>
      <geometry><box size="{bx} {by} {bz}"/></geometry>
    </collision>
  </link>
  <joint name="ab_joint" type="fixed">
    <parent link="arm_a"/>
    <child link="arm_b"/>
    <origin xyz="{ox} {oy} {oz}" rpy="0 0 0"/>
  </joint>
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

    # L-blocks
    for defn in L_BLOCK_DEFS:
        asset_dir = objects_dir / defn["name"]
        asset_dir.mkdir(exist_ok=True)

        urdf_path = asset_dir / "model.urdf"
        urdf_path.write_text(_generate_lblock_urdf(defn), encoding="utf-8")

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
