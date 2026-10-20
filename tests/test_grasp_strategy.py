from robotsmith.grasp.planner import resolve_grasp_strategy


class _Asset:
    def __init__(self, *, articulated=False, fixture=False):
        self.is_articulated = articulated
        # Articulated assets are always fixtures (Asset.is_fixture); resolve_grasp_strategy
        # gates on is_fixture, so mirror that invariant here.
        self.is_fixture = articulated or fixture


def test_articulated_asset_is_never_grasp_planned():
    assert resolve_grasp_strategy(_Asset(articulated=True), default="learned") == "none"


def test_general_asset_uses_default():
    assert resolve_grasp_strategy(_Asset(), default="learned") == "learned"
