"""Scene preset for place_cube: single block + target marker on ground plane."""

from robotsmith.scenes.config import SceneConfig, ObjectPlacement

place_cube_scene = SceneConfig(
    name="place_cube",
    description="Single red block and a target marker for pick-and-place task",
    objects=[
        ObjectPlacement(
            asset_query="block_red",
            count=1,
            name_override="cube",
        ),
    ],
    table_height=0.0,
    table_size=[1.2, 0.8, 0.0],
    workspace_xy=[[0.40, -0.20], [0.70, 0.20]],
)
