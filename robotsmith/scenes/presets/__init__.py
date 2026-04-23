"""Scene presets registry."""

from robotsmith.scenes.config import SceneConfig

SCENE_PRESETS: dict[str, SceneConfig] = {}


def _register(config: SceneConfig) -> SceneConfig:
    SCENE_PRESETS[config.name] = config
    return config


from robotsmith.scenes.presets.tabletop_simple import tabletop_simple  # noqa: E402
_register(tabletop_simple)

from robotsmith.scenes.presets.pick_cube import pick_cube_scene  # noqa: E402
_register(pick_cube_scene)

from robotsmith.scenes.presets.place_cube import place_cube_scene  # noqa: E402
_register(place_cube_scene)

from robotsmith.scenes.presets.stack_blocks import stack_blocks_scene  # noqa: E402
_register(stack_blocks_scene)

from robotsmith.scenes.presets.pick_bowl import pick_bowl_scene  # noqa: E402
_register(pick_bowl_scene)

from robotsmith.scenes.presets.stack_bowls import stack_bowls_scene  # noqa: E402
_register(stack_bowls_scene)
