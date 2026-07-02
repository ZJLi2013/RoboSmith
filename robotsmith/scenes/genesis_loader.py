"""Load a ResolvedScene into a Genesis simulation.

`scenes.backend` decides what should exist in the scene: concrete assets,
poses, scales, and logical object names. This module is the Genesis adapter: it
turns that ResolvedScene into Genesis entities and returns handles needed by
SimEnv.
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
    table: Optional[object] = None


def load_resolved_scene(
    resolved,
    *,
    gs_module=None,
    show_viewer: bool = False,
    fps: int = 30,
    substeps: int = 8,
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
        Physics substeps per frame. 8 keeps the contact solver stiff enough that
        a dropped object settles into the drawer tray instead of sinking and
        popping back out (soft contact at low substeps caused a visible bounce).
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

    scene.add_entity(
        morph=gs.morphs.Box(
            size=(20.0, 20.0, 0.02),
            pos=(0.0, 0.0, -0.01),
            fixed=True,
        ),
        surface=gs.surfaces.Default(color=(0.64, 0.68, 0.72, 1.0)),
    )

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
        # Genesis morph/set_quat use [w,x,y,z], the same convention as our
        # metadata quaternions, so pass them through unchanged.
        quat = tuple(po.quaternion)

        friction = po.asset.metadata.friction
        urdf_kwargs = dict(
            file=str(po.asset.urdf_path),
            pos=pos,
            quat=quat,
            default_armature=0.0,
        )
        # Fixtures (static props, and articulated furniture) are anchored at
        # their root link so they don't drift under gravity / contact.
        if getattr(po.asset, "is_fixture", False):
            urdf_kwargs["fixed"] = True
        if po.metric_scale != 1.0:
            urdf_kwargs["scale"] = po.metric_scale
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

    return GenesisSceneHandle(
        scene=scene,
        franka=franka,
        objects=obj_entities,
        object_names=obj_names,
        placed=resolved.placed_objects,
        table=table_entity,
    )
