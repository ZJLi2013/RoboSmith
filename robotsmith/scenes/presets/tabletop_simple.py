"""Simple tabletop scene: 1 table + 3-5 manipulable objects.

Objects are placed on the table surface using collision-aware placement.
Z height is automatically computed from table_height + stable_pose.z.
"""

from robotsmith.scenes.config import SceneConfig, ObjectPlacement

tabletop_simple = SceneConfig(
    name="tabletop_simple",
    description="Simple tabletop with mug, bowl, and blocks for pick-place tasks",
    objects=[
        ObjectPlacement(asset_query="mug", count=1),
        ObjectPlacement(asset_query="bowl", count=1),
        ObjectPlacement(asset_query="block", count=3),
    ],
    table_size=[1.2, 0.8, 0.05],
    table_height=0.75,
    workspace_xy=[[0.35, -0.20], [0.65, 0.20]],
)
