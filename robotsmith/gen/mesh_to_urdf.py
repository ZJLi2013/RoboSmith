"""Convert a trimesh mesh to a sim-ready URDF with collision geometry and inertia.

Handles: scaling to physical size, centering, collision mesh generation,
mass/inertia estimation, and URDF packaging.

Supports two visual mesh formats:
  - GLB (default): preserves PBR textures → textured rendering
  - OBJ (fallback): geometry only → white mesh

Collision mesh is a decimated copy of the original geometry (not a convex
hull), so concave shapes like bowls retain their interior surfaces.  The
sim engine (Genesis) applies CoACD convex decomposition at load time when
``convexify=True`` (default).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

URDF_TEMPLATE = """\
<?xml version="1.0"?>
<robot name="{name}">
  <link name="base_link">
    <inertial>
      <origin xyz="{com_x:.6f} {com_y:.6f} {com_z:.6f}" rpy="0 0 0"/>
      <mass value="{mass:.6f}"/>
      <inertia ixx="{ixx:.8f}" ixy="{ixy:.8f}" ixz="{ixz:.8f}"
               iyy="{iyy:.8f}" iyz="{iyz:.8f}" izz="{izz:.8f}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{visual_mesh}"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{collision_mesh}"/></geometry>
    </collision>
  </link>
</robot>
"""


def _bbox_inertia(mass: float, extents: np.ndarray) -> np.ndarray:
    """Fallback: approximate inertia from bounding box extents."""
    ex, ey, ez = extents
    ixx = mass / 12.0 * (ey ** 2 + ez ** 2)
    iyy = mass / 12.0 * (ex ** 2 + ez ** 2)
    izz = mass / 12.0 * (ex ** 2 + ey ** 2)
    return np.diag([ixx, iyy, izz])


def _has_texture(mesh) -> bool:
    """Check if a trimesh object has non-trivial texture data."""
    import trimesh
    if hasattr(mesh.visual, "material") and mesh.visual.material is not None:
        mat = mesh.visual.material
        if hasattr(mat, "image") and mat.image is not None:
            return True
        if hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
            return True
    if isinstance(mesh.visual, trimesh.visual.TextureVisuals):
        if mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
            return True
    return False


def _decimate(mesh, max_faces: int = 5000):
    """Decimate a mesh to *max_faces* while preserving shape (concavity).

    Uses ``fast_simplification`` when available; falls back to keeping the
    original mesh unchanged (still concave, just higher poly).
    """
    import trimesh as _trimesh

    if len(mesh.faces) <= max_faces:
        return mesh
    try:
        import fast_simplification
        ratio = max_faces / len(mesh.faces)
        pts, faces = fast_simplification.simplify(
            mesh.vertices.astype(np.float32),
            mesh.faces.astype(np.int32),
            target_reduction=1.0 - ratio,
        )
        return _trimesh.Trimesh(vertices=pts, faces=faces)
    except ImportError:
        return mesh


def mesh_to_urdf(
    mesh,
    output_dir: str | Path,
    name: str = "generated_object",
    target_size_m: Optional[float] = None,
    mass_kg: Optional[float] = None,
    density_kg_m3: float = 800.0,
    visual_format: str = "auto",
) -> Path:
    """Convert a trimesh.Trimesh to URDF with visual and collision meshes.

    Args:
        mesh: a trimesh.Trimesh object
        output_dir: directory to write URDF + meshes
        name: asset name (used in URDF robot name and filenames)
        target_size_m: if set, scale mesh so its longest bbox edge equals this
        mass_kg: override mass; if None, computed from volume * density
        density_kg_m3: density for mass estimation (default: wood-like 800)
        visual_format: "auto" (GLB if textured, else OBJ), "glb", or "obj"

    Returns:
        Path to the generated model.urdf
    """
    import trimesh

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if target_size_m is not None:
        extents = mesh.bounding_box.extents
        max_extent = max(extents)
        if max_extent > 0:
            scale = target_size_m / max_extent
            mesh.apply_scale(scale)

    centroid = mesh.centroid
    mesh.apply_translation(-centroid)

    use_glb = (visual_format == "glb") or (
        visual_format == "auto" and _has_texture(mesh)
    )

    if use_glb:
        visual_filename = "visual.glb"
        visual_path = output_dir / visual_filename
        mesh.export(str(visual_path), file_type="glb")
    else:
        visual_filename = "visual.obj"
        visual_path = output_dir / visual_filename
        mesh.export(str(visual_path), file_type="obj")

    # Also export OBJ as fallback for viewers that don't support GLB
    obj_path = output_dir / "visual.obj"
    if use_glb and not obj_path.exists():
        mesh.export(str(obj_path), file_type="obj")

    collision = _decimate(mesh, max_faces=5000)
    collision_path = output_dir / "collision.obj"
    collision.export(str(collision_path), file_type="obj")

    if mesh.is_watertight and mesh.volume > 0:
        volume = mesh.volume
        mass = mass_kg if mass_kg is not None else volume * density_kg_m3
        try:
            inertia = mesh.moment_inertia
            if np.any(np.isnan(inertia)) or np.all(inertia == 0):
                raise ValueError("bad inertia")
        except Exception:
            inertia = _bbox_inertia(mass, mesh.bounding_box.extents)
        com = mesh.center_mass
    else:
        extents = mesh.bounding_box.extents
        volume = float(np.prod(extents))
        mass = mass_kg if mass_kg is not None else max(volume * density_kg_m3, 0.01)
        inertia = _bbox_inertia(mass, extents)
        com = np.zeros(3)

    urdf_content = URDF_TEMPLATE.format(
        name=name,
        com_x=com[0], com_y=com[1], com_z=com[2],
        mass=mass,
        ixx=inertia[0, 0], ixy=inertia[0, 1], ixz=inertia[0, 2],
        iyy=inertia[1, 1], iyz=inertia[1, 2],
        izz=inertia[2, 2],
        visual_mesh=visual_filename,
        collision_mesh="collision.obj",
    )

    urdf_path = output_dir / "model.urdf"
    urdf_path.write_text(urdf_content, encoding="utf-8")

    return urdf_path
