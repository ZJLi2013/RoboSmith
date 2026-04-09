"""Web-based asset gallery browser using viser.

Displays all assets from the AssetLibrary in a grid layout.
Each asset is rendered as a 3D mesh with a label and metadata panel.
Supports filtering by source (builtin / generated / all).

Usage::

    from robotsmith.assets.library import AssetLibrary
    from robotsmith.viz.asset_browser import AssetBrowser

    lib = AssetLibrary("./assets")
    browser = AssetBrowser(lib, port=8080)
    browser.run()
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import viser
    from robotsmith.assets.library import AssetLibrary
    from robotsmith.assets.schema import Asset


def _load_asset_mesh(asset: Asset):
    """Load a trimesh object for an asset, handling both OBJ meshes and URDF primitives."""
    import trimesh
    import xml.etree.ElementTree as ET

    if asset.visual_mesh and asset.visual_mesh.exists():
        mesh = trimesh.load(str(asset.visual_mesh), force="mesh")
        if isinstance(mesh, trimesh.Trimesh):
            return mesh

    urdf_path = asset.urdf_path
    if not urdf_path.exists():
        return None

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    materials = {}
    for mat in root.iter("material"):
        name = mat.get("name", "")
        color_el = mat.find("color")
        if color_el is not None and name:
            rgba = [float(v) for v in color_el.get("rgba", "0.5 0.5 0.5 1").split()]
            materials[name] = rgba

    for link in root.iter("link"):
        for vis in link.iter("visual"):
            geom = vis.find("geometry")
            if geom is None:
                continue

            color = [0.5, 0.5, 0.5, 1.0]
            mat_el = vis.find("material")
            if mat_el is not None:
                color_el = mat_el.find("color")
                if color_el is not None:
                    color = [float(v) for v in color_el.get("rgba", "0.5 0.5 0.5 1").split()]
                elif mat_el.get("name", "") in materials:
                    color = materials[mat_el.get("name")]

            mesh = None
            box = geom.find("box")
            if box is not None:
                size = [float(v) for v in box.get("size", "0.1 0.1 0.1").split()]
                mesh = trimesh.creation.box(extents=size)

            cyl = geom.find("cylinder")
            if cyl is not None:
                mesh = trimesh.creation.cylinder(
                    radius=float(cyl.get("radius", "0.05")),
                    height=float(cyl.get("length", "0.1")),
                )

            sph = geom.find("sphere")
            if sph is not None:
                mesh = trimesh.creation.uv_sphere(radius=float(sph.get("radius", "0.05")))

            mesh_el = geom.find("mesh")
            if mesh_el is not None:
                mesh_path = urdf_path.parent / mesh_el.get("filename", "")
                if mesh_path.exists():
                    mesh = trimesh.load(str(mesh_path), force="mesh")
                    if not isinstance(mesh, trimesh.Trimesh):
                        mesh = None

            if mesh is not None:
                mesh.visual = trimesh.visual.ColorVisuals(
                    mesh=mesh,
                    face_colors=np.array([int(c * 255) for c in color[:4]], dtype=np.uint8),
                )
                return mesh

    return None


class AssetBrowser:
    """Web-based asset gallery browser.

    Arranges all assets in a grid on a table surface with labels,
    plus a sidebar panel showing metadata for the selected / hovered asset.
    """

    CELL_SIZE = 0.25
    MAX_COLS = 6
    DISPLAY_SCALE = 0.12

    def __init__(
        self,
        library: AssetLibrary,
        port: int = 8080,
        source_filter: str = "all",
    ) -> None:
        try:
            import viser
        except ImportError:
            raise ImportError("viser is required. Install with: pip install viser")

        self._viser = viser
        self._library = library
        self._source_filter = source_filter
        self._server: viser.ViserServer = viser.ViserServer(
            port=port, label="RoboSmith Asset Browser"
        )
        self._server.scene.set_up_direction("+z")

        self._assets = self._filter_assets()
        self._setup_scene()
        self._setup_gui()

    def _filter_assets(self) -> list:
        all_assets = self._library.list_all()
        if self._source_filter == "generated":
            return [a for a in all_assets if a.metadata.source == "generated"]
        elif self._source_filter == "builtin":
            return [a for a in all_assets if a.metadata.source != "generated"]
        return all_assets

    def _setup_scene(self) -> None:
        import trimesh

        self._server.scene.add_grid(
            "/ground",
            width=self.MAX_COLS * self.CELL_SIZE + 0.5,
            height=max(1, math.ceil(len(self._assets) / self.MAX_COLS)) * self.CELL_SIZE + 0.5,
            cell_size=self.CELL_SIZE,
            position=(
                (self.MAX_COLS - 1) * self.CELL_SIZE / 2,
                -(max(1, math.ceil(len(self._assets) / self.MAX_COLS)) - 1) * self.CELL_SIZE / 2,
                -0.01,
            ),
        )

        for idx, asset in enumerate(self._assets):
            col = idx % self.MAX_COLS
            row = idx // self.MAX_COLS
            x = col * self.CELL_SIZE
            y = -row * self.CELL_SIZE
            z = 0.0

            mesh = _load_asset_mesh(asset)
            if mesh is not None:
                extents = mesh.bounding_box.extents
                max_ext = max(extents)
                if max_ext > 0:
                    scale = self.DISPLAY_SCALE / max_ext
                    mesh.apply_scale(scale)
                mesh.apply_translation(-mesh.centroid)

                self._server.scene.add_mesh_trimesh(
                    f"/assets/{asset.name}/mesh",
                    mesh=mesh,
                    position=(x, y, z + self.DISPLAY_SCALE / 2),
                )
            else:
                placeholder = trimesh.creation.box(extents=[0.04, 0.04, 0.04])
                placeholder.visual = trimesh.visual.ColorVisuals(
                    mesh=placeholder,
                    face_colors=np.array([180, 180, 180, 255], dtype=np.uint8),
                )
                self._server.scene.add_mesh_trimesh(
                    f"/assets/{asset.name}/mesh",
                    mesh=placeholder,
                    position=(x, y, z + 0.02),
                )

            source_icon = "★" if asset.metadata.source == "generated" else "●"
            self._server.scene.add_label(
                f"/assets/{asset.name}/label",
                text=f"{source_icon} {asset.name}",
                position=(x, y, z + self.DISPLAY_SCALE + 0.03),
            )

    def _setup_gui(self) -> None:
        n_builtin = len([a for a in self._assets if a.metadata.source != "generated"])
        n_gen = len([a for a in self._assets if a.metadata.source == "generated"])

        summary = (
            f"## RoboSmith Asset Browser\n\n"
            f"**Total:** {len(self._assets)} assets\n\n"
            f"- ● Builtin: {n_builtin}\n"
            f"- ★ Generated: {n_gen}\n\n"
            f"---\n\n"
        )
        for asset in self._assets:
            tags = ", ".join(asset.tags[:5])
            src = "generated" if asset.metadata.source == "generated" else "builtin"
            size = "×".join(f"{s}" for s in asset.metadata.size_cm)
            summary += (
                f"**{asset.name}** ({src})\n"
                f"- Tags: `{tags}`\n"
                f"- Mass: {asset.metadata.mass_kg} kg\n"
                f"- Size: {size} cm\n"
            )
            if asset.metadata.description:
                summary += f"- {asset.metadata.description}\n"
            summary += "\n"

        self._server.gui.add_markdown(summary)

    def run(self, blocking: bool = True) -> None:
        url = f"http://localhost:{self._server._port}"
        print(f"\n  RoboSmith Asset Browser running at: {url}")
        print(f"  Showing {len(self._assets)} assets")
        print("  Press Ctrl-C to stop.\n")
        if blocking:
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\nBrowser stopped.")
