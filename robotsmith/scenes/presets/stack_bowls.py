"""Scene preset for stack_bowls: three bowls on ground plane."""

from robotsmith.scenes.config import SceneConfig, ObjectPlacement

stack_bowls_scene = SceneConfig(
    name="stack_bowls",
    description="Three bowls (bowl_02, scaled to ~3.5 cm) for stacking task",
    objects=[
        ObjectPlacement(
            asset_query="bowl_02", count=1, scale=0.25, name_override="bowl_a",
        ),
        ObjectPlacement(
            asset_query="bowl_02", count=1, scale=0.25, name_override="bowl_b",
        ),
        ObjectPlacement(
            asset_query="bowl_02", count=1, scale=0.25, name_override="bowl_c",
        ),
    ],
    table_height=0.0,
    table_size=[1.2, 0.8, 0.0],
    workspace_xy=[[0.40, -0.20], [0.70, 0.20]],
)
