"""Sorting table scene: multi-color blocks for sorting tasks."""

from robotsmith.scenes.config import SceneConfig, ObjectPlacement

TABLE_Z = 0.775

sorting_table = SceneConfig(
    name="sorting_table",
    description="Table with colored blocks and a plate target for sorting tasks",
    objects=[
        ObjectPlacement(
            asset_query="red block",
            count=2,
            position_range=[[0.3, -0.15, TABLE_Z], [0.55, 0.0, TABLE_Z]],
        ),
        ObjectPlacement(
            asset_query="blue block",
            count=2,
            position_range=[[0.3, 0.0, TABLE_Z], [0.55, 0.15, TABLE_Z]],
        ),
        ObjectPlacement(
            asset_query="green block",
            count=2,
            position_range=[[0.35, -0.1, TABLE_Z], [0.6, 0.1, TABLE_Z]],
        ),
        ObjectPlacement(
            asset_query="plate",
            count=1,
            fixed_position=[0.5, 0.0, TABLE_Z],
        ),
    ],
    table_size=[0.8, 0.6, 0.05],
    table_height=0.75,
)
