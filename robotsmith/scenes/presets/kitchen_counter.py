"""Kitchen counter scene: utensils + containers for diverse manipulation."""

from robotsmith.scenes.config import SceneConfig, ObjectPlacement

TABLE_Z = 0.775

kitchen_counter = SceneConfig(
    name="kitchen_counter",
    description="Kitchen counter with mug, plate, bottle, fork, and spoon",
    objects=[
        ObjectPlacement(
            asset_query="mug",
            count=1,
            position_range=[[0.35, -0.1, TABLE_Z], [0.5, 0.1, TABLE_Z]],
        ),
        ObjectPlacement(
            asset_query="plate",
            count=1,
            position_range=[[0.25, -0.15, TABLE_Z], [0.4, 0.0, TABLE_Z]],
        ),
        ObjectPlacement(
            asset_query="bottle",
            count=1,
            position_range=[[0.5, -0.15, TABLE_Z], [0.65, 0.15, TABLE_Z]],
        ),
        ObjectPlacement(
            asset_query="fork",
            count=1,
            position_range=[[0.3, 0.1, TABLE_Z], [0.45, 0.2, TABLE_Z]],
        ),
        ObjectPlacement(
            asset_query="spoon",
            count=1,
            position_range=[[0.3, -0.2, TABLE_Z], [0.45, -0.1, TABLE_Z]],
        ),
    ],
    table_size=[0.9, 0.6, 0.05],
    table_height=0.75,
)
