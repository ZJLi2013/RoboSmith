"""Load a ResolvedScene into a Genesis simulation.

This module bridges the simulator-agnostic ResolvedScene to Genesis entities,
replacing the hardcoded scene setup in scripts/part2/collect_data.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class GenesisSceneHandle:
    """References to Genesis entities created from a ResolvedScene."""

    scene: object  # gs.Scene
    franka: object  # gs.Entity
    objects: list[object] = field(default_factory=list)
    object_names: list[str] = field(default_factory=list)
    """Parallel to ``objects`` — logical names from PlacedObject.name."""
    placed: list = field(default_factory=list)
    """The PlacedObject list from the ResolvedScene, for metadata access."""
    cameras: dict[str, object] = field(default_factory=dict)
    table: Optional[object] = None


def _quat_wxyz_to_xyzw(quat_wxyz: list[float]) -> tuple[float, ...]:
    """Convert [w,x,y,z] metadata format to [x,y,z,w] for Genesis morph/set_quat."""
    w, x, y, z = quat_wxyz
    return (x, y, z, w)


def load_resolved_scene(
    resolved,
    *,
    gs_module=None,
    show_viewer: bool = False,
    fps: int = 30,
    substeps: int = 4,
    box_box_detection: bool = True,
) -> GenesisSceneHandle:
    """Create a Genesis scene from a ResolvedScene.

    Parameters
    ----------
    resolved : ResolvedScene
        The fully-resolved scene with placed objects.
    gs_module : module, optional
        The ``genesis`` module. If None, imported at call time.
        Allows deferring the heavy import and testing without Genesis.
    show_viewer : bool
        Whether to show the Genesis GUI viewer.
    fps : int
        Simulation frames per second.
    substeps : int
        Physics substeps per frame.
    box_box_detection : bool
        Enable box-box collision detection (disable for AMD LLVM workaround).
    """
    if gs_module is None:
        import genesis as gs_module  # noqa: N811

    gs = gs_module
    config = resolved.config

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / fps, substeps=substeps),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_joint_limit=True,
            box_box_detection=box_box_detection,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())

    # Position table so it covers both the Franka base (at x=0) and the
    # workspace area.  The table URDF origin is at the bottom-center;
    # tabletop extends ±table_size[0]/2 in X, ±table_size[1]/2 in Y.
    ws = config.workspace_xy
    ws_center_x = (ws[0][0] + ws[1][0]) / 2.0
    # Center table between Franka base (x=0) and workspace center
    table_center_x = ws_center_x / 2.0
    table_center_y = (ws[0][1] + ws[1][1]) / 2.0

    table_entity = None
    if resolved.table_asset and resolved.table_asset.urdf_path.exists():
        table_entity = scene.add_entity(
            gs.morphs.URDF(
                file=str(resolved.table_asset.urdf_path),
                pos=(table_center_x, table_center_y, 0.0),
                fixed=True,
            ),
        )

    obj_entities = []
    obj_names = []
    for po in resolved.placed_objects:
        pos = tuple(po.position)
        quat = _quat_wxyz_to_xyzw(po.quaternion)

        friction = po.asset.metadata.friction
        urdf_kwargs = dict(
            file=str(po.asset.urdf_path),
            pos=pos,
            quat=quat,
            default_armature=0.0,
        )
        if po.scale != 1.0:
            urdf_kwargs["scale"] = po.scale
        entity = scene.add_entity(
            morph=gs.morphs.URDF(**urdf_kwargs),
            material=gs.materials.Rigid(friction=friction),
        )
        obj_entities.append(entity)
        obj_names.append(po.name)

    table_surface_z = config.table_height + config.table_size[2] / 2.0
    # Franka base sits on the table surface (not floating)
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0, 0, table_surface_z),
        ),
    )

    cameras = {}
    cam_cfg = config.camera_position, config.camera_target
    cameras["default"] = scene.add_camera(
        res=(640, 480),
        pos=tuple(cam_cfg[0]),
        lookat=tuple(cam_cfg[1]),
        fov=45,
        GUI=False,
    )

    return GenesisSceneHandle(
        scene=scene,
        franka=franka,
        objects=obj_entities,
        object_names=obj_names,
        placed=resolved.placed_objects,
        cameras=cameras,
        table=table_entity,
    )
