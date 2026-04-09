"""Simple tabletop scene: 1 table + 3-5 manipulable objects."""

from robotsmith.scenes.config import SceneConfig, ObjectPlacement

TABLE_Z = 0.775

tabletop_simple = SceneConfig(
    name="tabletop_simple",
    description="Simple tabletop with mug, bowl, and blocks for pick-place tasks",
    objects=[
        ObjectPlacement(
            asset_query="mug",
            count=1,
            position_range=[[0.35, -0.15, TABLE_Z], [0.55, 0.15, TABLE_Z]],
        ),
        ObjectPlacement(
            asset_query="bowl",
            count=1,
            position_range=[[0.25, -0.2, TABLE_Z], [0.45, -0.05, TABLE_Z]],
        ),
        ObjectPlacement(
            asset_query="block",
            count=3,
            position_range=[[0.3, -0.15, TABLE_Z], [0.6, 0.15, TABLE_Z]],
        ),
    ],
    table_size=[0.8, 0.6, 0.05],
    table_height=0.75,
)
