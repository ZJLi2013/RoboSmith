"""Viser-based 3D scene viewer for RoboSmith assets and scenes.

Supports:
- Built-in URDF primitives (box, cylinder, sphere with color)
- OBJ mesh assets (from Hunyuan3D generation)
- Robot URDFs (Franka, etc.) via viser.extras.ViserUrdf
- Interactive GUI controls for object position adjustment
"""

from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import trimesh

if TYPE_CHECKING:
    import viser
    from robotsmith.assets.schema import Asset
    from robotsmith.scenes.backend import ResolvedScene


def _parse_urdf_visual(urdf_path: Path) -> list[dict]:
    """Extract visual geometry info from a single-link URDF.

    Returns a list of dicts with keys: type, params, color, origin_xyz, origin_rpy.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    visuals = []

    materials = {}
    for mat in root.iter("material"):
        name = mat.get("name", "")
        color_el = mat.find("color")
        if color_el is not None and name:
            rgba = [float(v) for v in color_el.get("rgba", "0.5 0.5 0.5 1").split()]
            materials[name] = rgba

    for link in root.iter("link"):
        for vis in link.iter("visual"):
            origin = vis.find("origin")
            xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()] if origin is not None else [0, 0, 0]
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else [0, 0, 0]

            color = [0.5, 0.5, 0.5, 1.0]
            mat_el = vis.find("material")
            if mat_el is not None:
                color_el = mat_el.find("color")
                if color_el is not None:
                    color = [float(v) for v in color_el.get("rgba", "0.5 0.5 0.5 1").split()]
                elif mat_el.get("name", "") in materials:
                    color = materials[mat_el.get("name")]

            geom = vis.find("geometry")
            if geom is None:
                continue

            info = {"color": color, "origin_xyz": xyz, "origin_rpy": rpy}

            box = geom.find("box")
            if box is not None:
                size = [float(v) for v in box.get("size", "0.1 0.1 0.1").split()]
                info["type"] = "box"
                info["size"] = size
                visuals.append(info)
                continue

            cyl = geom.find("cylinder")
            if cyl is not None:
                info["type"] = "cylinder"
                info["radius"] = float(cyl.get("radius", "0.05"))
                info["length"] = float(cyl.get("length", "0.1"))
                visuals.append(info)
                continue

            sph = geom.find("sphere")
            if sph is not None:
                info["type"] = "sphere"
                info["radius"] = float(sph.get("radius", "0.05"))
                visuals.append(info)
                continue

            mesh_el = geom.find("mesh")
            if mesh_el is not None:
                info["type"] = "mesh"
                info["filename"] = mesh_el.get("filename", "")
                visuals.append(info)
                continue

    return visuals


def _visual_to_trimesh(vis_info: dict, urdf_dir: Path) -> Optional[trimesh.Trimesh]:
    """Convert parsed URDF visual info to a positioned trimesh object."""
    geom_type = vis_info["type"]
    ox, oy, oz = vis_info["origin_xyz"]

    if geom_type == "box":
        sx, sy, sz = vis_info["size"]
        mesh = trimesh.creation.box(extents=[sx, sy, sz])
    elif geom_type == "cylinder":
        mesh = trimesh.creation.cylinder(
            radius=vis_info["radius"],
            height=vis_info["length"],
        )
    elif geom_type == "sphere":
        mesh = trimesh.creation.uv_sphere(radius=vis_info["radius"])
    elif geom_type == "mesh":
        mesh_path = urdf_dir / vis_info["filename"]
        if not mesh_path.exists():
            return None
        loaded = trimesh.load(str(mesh_path), force="mesh")
        if not isinstance(loaded, trimesh.Trimesh):
            return None
        mesh = loaded
    else:
        return None

    mesh.apply_translation([ox, oy, oz])

    rgba = vis_info.get("color", [0.5, 0.5, 0.5, 1.0])
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        face_colors=np.array([int(c * 255) for c in rgba[:4]], dtype=np.uint8),
    )

    return mesh


def _euler_to_wxyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Convert RPY (extrinsic XYZ) to quaternion (w, x, y, z)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)


class SceneViewer:
    """Web-based 3D scene viewer using viser.

    Usage::

        viewer = SceneViewer(port=8080)
        viewer.show_resolved_scene(resolved_scene)
        viewer.add_robot_urdf("franka.urdf")
        viewer.run()
    """

    def __init__(self, port: int = 8080, label: str = "RoboSmith Scene Viewer"):
        try:
            import viser
        except ImportError:
            raise ImportError(
                "viser is required for visualization. Install with: pip install 'robotsmith[viz]'"
            )
        self._viser = viser
        self._server: viser.ViserServer = viser.ViserServer(port=port, label=label)
        self._node_counter = 0
        self._position_controls: dict[str, dict] = {}

        self._server.scene.set_up_direction("+z")
        self._add_ground_grid()

    def _next_name(self, prefix: str) -> str:
        self._node_counter += 1
        return f"/scene/{prefix}_{self._node_counter}"

    def _add_ground_grid(self) -> None:
        self._server.scene.add_grid(
            "/ground_grid",
            width=4.0,
            height=4.0,
            cell_size=0.25,
            position=(0.0, 0.0, 0.0),
        )

    def add_asset(
        self,
        asset: Asset,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
        label: Optional[str] = None,
        interactive: bool = True,
    ) -> str:
        """Add a single asset to the viewer.

        Args:
            asset: RoboSmith Asset object.
            position: (x, y, z) in meters.
            rotation_rpy: (roll, pitch, yaw) in radians.
            label: Display label; defaults to asset.name.
            interactive: If True, add GUI sliders for position adjustment.

        Returns:
            The scene node name.
        """
        name = label or asset.name
        node_name = self._next_name(name)

        visuals = _parse_urdf_visual(asset.urdf_path)
        urdf_dir = asset.urdf_path.parent

        wxyz = _euler_to_wxyz(*rotation_rpy)

        for i, vis_info in enumerate(visuals):
            mesh = _visual_to_trimesh(vis_info, urdf_dir)
            if mesh is None:
                continue
            sub_name = f"{node_name}/visual_{i}" if len(visuals) > 1 else node_name
            self._server.scene.add_mesh_trimesh(
                sub_name,
                mesh=mesh,
                position=position,
                wxyz=wxyz,
            )

        self._server.scene.add_label(
            f"{node_name}/label",
            text=name,
            position=(position[0], position[1], position[2] + 0.08),
        )

        if interactive:
            self._add_position_controls(name, node_name, position, visuals, urdf_dir)

        return node_name

    def _add_position_controls(
        self,
        name: str,
        node_name: str,
        position: tuple[float, float, float],
        visuals: list[dict],
        urdf_dir: Path,
    ) -> None:
        folder = self._server.gui.add_folder(f"Pos: {name}")
        with folder:
            sx = self._server.gui.add_slider(f"{name}_x", min=-2.0, max=2.0, step=0.01, initial_value=position[0])
            sy = self._server.gui.add_slider(f"{name}_y", min=-2.0, max=2.0, step=0.01, initial_value=position[1])
            sz = self._server.gui.add_slider(f"{name}_z", min=-0.5, max=2.0, step=0.01, initial_value=position[2])

        def _on_update(_) -> None:
            new_pos = (sx.value, sy.value, sz.value)
            for i, vis_info in enumerate(visuals):
                mesh = _visual_to_trimesh(vis_info, urdf_dir)
                if mesh is None:
                    continue
                sub = f"{node_name}/visual_{i}" if len(visuals) > 1 else node_name
                self._server.scene.add_mesh_trimesh(sub, mesh=mesh, position=new_pos)
            self._server.scene.add_label(
                f"{node_name}/label",
                text=name,
                position=(new_pos[0], new_pos[1], new_pos[2] + 0.08),
            )

        sx.on_update(_on_update)
        sy.on_update(_on_update)
        sz.on_update(_on_update)

    def show_resolved_scene(
        self,
        scene: ResolvedScene,
        show_table: bool = True,
        show_plane: bool = True,
        interactive: bool = True,
    ) -> None:
        """Load an entire ResolvedScene into the viewer.

        Args:
            scene: Output from SceneBackend.resolve().
            show_table: Show the table asset.
            show_plane: Show the ground plane.
            interactive: Add GUI sliders per object.
        """
        if show_plane and scene.plane_asset:
            self.add_asset(
                scene.plane_asset,
                position=(0.0, 0.0, 0.0),
                label="ground",
                interactive=False,
            )

        if show_table and scene.table_asset:
            self.add_asset(
                scene.table_asset,
                position=(0.4, 0.0, 0.0),
                label="table",
                interactive=interactive,
            )

        for po in scene.placed_objects:
            self.add_asset(
                po.asset,
                position=tuple(po.position),
                rotation_rpy=tuple(po.rotation),
                interactive=interactive,
            )

        self._add_scene_info_panel(scene)

    def _add_scene_info_panel(self, scene: ResolvedScene) -> None:
        md = f"**Scene:** {scene.config.name}\n\n"
        md += f"**Objects:** {len(scene.placed_objects)}\n\n"
        for po in scene.placed_objects:
            pos = [round(v, 3) for v in po.position]
            md += f"- {po.asset.name} @ {pos}\n"
        self._server.gui.add_markdown(md)

    def add_robot_urdf(
        self,
        urdf_path: str | Path,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        joint_angles: Optional[dict[str, float]] = None,
        interactive_joints: bool = True,
    ) -> None:
        """Add a robot URDF (e.g. Franka Panda) via viser.extras.ViserUrdf.

        Requires yourdfpy: ``pip install yourdfpy``

        Args:
            urdf_path: Path to robot URDF file.
            position: Base position of the robot.
            joint_angles: Optional dict of {joint_name: angle_rad}.
            interactive_joints: If True, add joint angle sliders.
        """
        try:
            from viser.extras import ViserUrdf
        except ImportError:
            raise ImportError(
                "viser.extras.ViserUrdf requires yourdfpy. "
                "Install with: pip install yourdfpy"
            )

        urdf_path = Path(urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"Robot URDF not found: {urdf_path}")

        viser_urdf = ViserUrdf(
            self._server,
            urdf_or_path=str(urdf_path),
            root_node_name="/robot",
        )

        self._server.scene.add_frame(
            "/robot_base",
            position=position,
            show_axes=True,
            axes_length=0.15,
            axes_radius=0.005,
        )

        if interactive_joints:
            actuated = [
                j for j in viser_urdf._urdf.joint_names
                if viser_urdf._urdf.joint_map[j].type in ("revolute", "prismatic", "continuous")
            ]
            if actuated:
                folder = self._server.gui.add_folder("Robot Joints")
                with folder:
                    sliders = {}
                    for jname in actuated:
                        joint = viser_urdf._urdf.joint_map[jname]
                        lo = joint.limit.lower if joint.limit else -math.pi
                        hi = joint.limit.upper if joint.limit else math.pi
                        init = 0.0
                        if joint_angles and jname in joint_angles:
                            init = joint_angles[jname]
                        s = self._server.gui.add_slider(
                            jname, min=lo, max=hi, step=0.01, initial_value=init,
                        )
                        sliders[jname] = s

                    def _on_joint_update(_) -> None:
                        cfg = {n: s.value for n, s in sliders.items()}
                        viser_urdf.update_cfg(cfg)

                    for s in sliders.values():
                        s.on_update(_on_joint_update)

                if joint_angles:
                    viser_urdf.update_cfg(joint_angles)

    def add_frame(
        self,
        name: str,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        axes_length: float = 0.1,
    ) -> None:
        """Add a coordinate frame marker (useful for debugging positions)."""
        self._server.scene.add_frame(
            self._next_name(f"frame_{name}"),
            position=position,
            show_axes=True,
            axes_length=axes_length,
            axes_radius=0.003,
        )

    def run(self, blocking: bool = True) -> None:
        """Start the viewer. Open browser at the printed URL.

        Args:
            blocking: If True, block until Ctrl-C.
        """
        url = f"http://localhost:{self._server._port}"
        print(f"\n  RoboSmith Scene Viewer running at: {url}")
        print("  Press Ctrl-C to stop.\n")

        if blocking:
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\nViewer stopped.")
