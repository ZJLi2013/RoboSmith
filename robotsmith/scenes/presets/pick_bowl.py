"""Scene preset for pick_bowl: single bowl (scaled down) on ground plane."""

from robotsmith.scenes.config import SceneConfig, ObjectPlacement

pick_bowl_scene = SceneConfig(
    name="pick_bowl",
    description="Single bowl scaled to ~7 cm for pick task",
    objects=[
        ObjectPlacement(
            asset_query="bowl",
            count=1,
            scale=0.5,
            name_override="bowl",
        ),
    ],
    table_height=0.0,
    table_size=[1.2, 0.8, 0.0],
    workspace_xy=[[0.40, -0.20], [0.70, 0.20]],
)
