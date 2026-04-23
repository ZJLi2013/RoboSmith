"""Scene preset for line_bowls: three bowls to be arranged in a line."""

from robotsmith.scenes.config import SceneConfig, ObjectPlacement

line_bowls_scene = SceneConfig(
    name="line_bowls",
    description="Three bowls (bowl_02, scaled to ~5 cm) for pick-and-place line arrangement",
    objects=[
        ObjectPlacement(
            asset_query="bowl_02", count=1, scale=0.5, name_override="bowl_a",
        ),
        ObjectPlacement(
            asset_query="bowl_02", count=1, scale=0.5, name_override="bowl_b",
        ),
        ObjectPlacement(
            asset_query="bowl_02", count=1, scale=0.5, name_override="bowl_c",
        ),
    ],
    workspace_xy=[[0.40, -0.20], [0.70, 0.20]],
)
