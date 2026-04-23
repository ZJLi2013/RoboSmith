"""Scene preset for pick_cube: single block on tabletop."""

from robotsmith.scenes.config import SceneConfig, ObjectPlacement

pick_cube_scene = SceneConfig(
    name="pick_cube",
    description="Single red block for pick task",
    objects=[
        ObjectPlacement(
            asset_query="block_red",
            count=1,
            name_override="cube",
        ),
    ],
    workspace_xy=[[0.40, -0.20], [0.70, 0.20]],
)
