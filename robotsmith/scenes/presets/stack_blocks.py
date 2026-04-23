"""Scene preset for stack_blocks: three colored blocks on tabletop."""

from robotsmith.scenes.config import SceneConfig, ObjectPlacement

stack_blocks_scene = SceneConfig(
    name="stack_blocks",
    description="Three colored blocks (red, green, blue) for stacking task",
    objects=[
        ObjectPlacement(asset_query="block_red", count=1, name_override="block_red"),
        ObjectPlacement(asset_query="block_green", count=1, name_override="block_green"),
        ObjectPlacement(asset_query="block_blue", count=1, name_override="block_blue"),
    ],
    workspace_xy=[[0.40, -0.20], [0.70, 0.20]],
)
