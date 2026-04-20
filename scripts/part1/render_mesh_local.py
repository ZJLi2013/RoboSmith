"""Render a 3D mesh (OBJ/GLB) to PNG using trimesh + pyrender.

Usage:
    pip install pyrender trimesh
    python scripts/part1/render_mesh_local.py path/to/visual.obj -o render.png
"""
import argparse
import os
import sys

import numpy as np
import trimesh


def render_mesh_to_png(mesh_path: str, output_path: str, resolution: tuple = (800, 600)):
    """Render a mesh to PNG using pyrender offscreen renderer."""
    try:
        import pyrender
    except ImportError:
        print("pyrender not installed. Trying trimesh scene export...")
        return render_with_trimesh(mesh_path, output_path, resolution)

    mesh = trimesh.load(mesh_path, force="mesh")
    print(f"Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    current_size = mesh.bounding_box.extents.max()
    if current_size > 0:
        scale = 0.12 / current_size
        mesh.apply_scale(scale)
    mesh.apply_translation(-mesh.centroid)

    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.7, 0.3, 0.2, 1.0],
        metallicFactor=0.2,
        roughnessFactor=0.6,
    )
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene.add(pyrender_mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4)
    camera_pose = np.eye(4)
    camera_pose[2, 3] = 0.4
    camera_pose[1, 3] = 0.05
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(*resolution)
    color, _ = renderer.render(scene)
    renderer.delete()

    from PIL import Image
    img = Image.fromarray(color)
    img.save(output_path)
    print(f"Rendered: {output_path} ({resolution[0]}x{resolution[1]})")


def render_with_trimesh(mesh_path: str, output_path: str, resolution: tuple = (800, 600)):
    """Fallback: render using trimesh's built-in scene viewer."""
    mesh = trimesh.load(mesh_path, force="mesh")
    print(f"Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    current_size = mesh.bounding_box.extents.max()
    if current_size > 0:
        scale = 0.12 / current_size
        mesh.apply_scale(scale)
    mesh.apply_translation(-mesh.centroid)

    scene = trimesh.Scene(mesh)

    try:
        png_data = scene.save_image(resolution=resolution, visible=False)
        with open(output_path, "wb") as f:
            f.write(png_data)
        print(f"Rendered: {output_path} ({resolution[0]}x{resolution[1]})")
    except Exception as e:
        print(f"trimesh save_image failed: {e}")
        print("Install pyrender for offscreen rendering: pip install pyrender")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", help="Path to OBJ/GLB mesh file")
    parser.add_argument("-o", "--output", default="render.png")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    args = parser.parse_args()

    render_mesh_to_png(args.mesh, args.output, (args.width, args.height))
