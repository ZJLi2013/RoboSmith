import pytest

from robotsmith.grasp.planner import resolve_grasp_strategy


class _Meta:
    def __init__(self, grasp_strategy=None):
        self.grasp_strategy = grasp_strategy


class _Asset:
    def __init__(self, *, articulated=False, grasp_strategy=None, fixture=False):
        self.is_articulated = articulated
        # Articulated assets are always fixtures (Asset.is_fixture); resolve_grasp_strategy
        # gates on is_fixture, so mirror that invariant here.
        self.is_fixture = articulated or fixture
        self.metadata = _Meta(grasp_strategy)


def test_articulated_asset_is_never_grasp_planned():
    asset = _Asset(articulated=True, grasp_strategy="learned")
    assert resolve_grasp_strategy(asset, default="learned") == "none"


def test_general_asset_uses_default():
    assert resolve_grasp_strategy(_Asset(), default="learned") == "learned"


def test_per_asset_override_wins_over_default():
    asset = _Asset(grasp_strategy="none")
    assert resolve_grasp_strategy(asset, default="learned") == "none"


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError, match="grasp_strategy"):
        resolve_grasp_strategy(_Asset(grasp_strategy="magic"))
