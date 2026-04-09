"""Mesh post-processing: remove base plane artifact, re-center, etc.

Image-to-3D models (Hunyuan3D, TRELLIS, TripoSG) commonly produce a flat
base plane due to training data bias. This module detects and removes it.

The trimmed mesh is NOT guaranteed to be watertight. For tabletop grasping
this is acceptable — collision uses convex hull, and mass/inertia falls
back to bounding-box estimation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def remove_base_plane(
    mesh,
    normal_thresh: float = 0.85,
    height_percentile: float = 5.0,
    min_face_ratio: float = 0.005,
) -> tuple:
    """Remove flat downward-facing faces near the bottom of the mesh.

    Detection criteria (both must be met):
      1. Face normal Z-component < -normal_thresh  (facing downward)
      2. Face centroid Z < height_percentile% of the mesh height range

    Args:
        mesh: trimesh.Trimesh object.
        normal_thresh: cosine threshold for "facing down" (0.85 ≈ 32°).
        height_percentile: bottom percentage of height range to consider.
        min_face_ratio: skip removal if fewer than this fraction of faces
            would be removed (avoids false positives on non-flat meshes).

    Returns:
        (cleaned_mesh, n_removed) — the processed mesh and count of faces removed.
        If no base plane is detected, returns the original mesh unchanged.
    """
    import trimesh

    face_normals = mesh.face_normals
    faces_down = face_normals[:, 2] < -normal_thresh

    vertices = mesh.vertices
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    z_range = z_max - z_min
    if z_range < 1e-8:
        return mesh, 0

    z_cutoff = z_min + z_range * (height_percentile / 100.0)

    face_centroids = mesh.triangles_center
    faces_low = face_centroids[:, 2] < z_cutoff

    base_mask = faces_down & faces_low
    n_base = int(base_mask.sum())

    if n_base == 0:
        return mesh, 0

    if n_base / len(mesh.faces) < min_face_ratio:
        return mesh, 0

    keep_mask = ~base_mask
    cleaned = mesh.submesh([np.where(keep_mask)[0]], append=True)

    return cleaned, n_base


def cleanup_mesh(
    mesh,
    remove_base: bool = True,
    recenter: bool = True,
    remove_degenerate: bool = True,
) -> tuple:
    """Full cleanup pipeline for imported or generated meshes.

    Args:
        mesh: trimesh.Trimesh object.
        remove_base: attempt base plane removal.
        recenter: translate mesh so bounding box center is at origin.
        remove_degenerate: remove zero-area faces and unreferenced vertices.

    Returns:
        (cleaned_mesh, stats_dict) with cleanup statistics.
    """
    stats = {"base_faces_removed": 0, "degenerate_faces_removed": 0}

    if remove_degenerate:
        n_before = len(mesh.faces)
        if hasattr(mesh, 'remove_degenerate_faces'):
            mesh.remove_degenerate_faces()
        else:
            mesh.update_faces(mesh.nondegenerate)
        if hasattr(mesh, 'remove_unreferenced_vertices'):
            mesh.remove_unreferenced_vertices()
        stats["degenerate_faces_removed"] = n_before - len(mesh.faces)

    if remove_base:
        mesh, n_removed = remove_base_plane(mesh)
        stats["base_faces_removed"] = n_removed

    if recenter:
        centroid = mesh.bounding_box.centroid
        mesh.apply_translation(-centroid)

    return mesh, stats
