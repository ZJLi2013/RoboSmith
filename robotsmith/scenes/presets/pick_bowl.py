"""Scene preset for pick_bowl: single bowl (scaled down) on tabletop."""

from robotsmith.scenes.config import SceneConfig, ObjectPlacement

pick_bowl_scene = SceneConfig(
    name="pick_bowl",
    description="Single bowl (bowl_02, scaled to ~5 cm) for pick task",
    objects=[
        ObjectPlacement(
            asset_query="bowl_02",
            count=1,
            scale=0.35,
            name_override="bowl",
        ),
    ],
    workspace_xy=[[0.40, -0.20], [0.70, 0.20]],
)
