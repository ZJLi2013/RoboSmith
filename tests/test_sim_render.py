"""SimEnv.render: the backend ROBOSMITH_RENDERER names, a hipRT backend that fails without falling back, one wrist
pose for both backends, and a reading that names the backend that rendered."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from robotsmith.sim.sim_env import SimEnv, renderer_from_env  # noqa: E402

SCENE = SimpleNamespace(camera_position=(1.0, 0.0, 1.5), camera_target=(0.3, 0.0, 0.8))
HAND_POS = np.array([0.4, 0.1, 1.0])
# Hand turned 90 degrees about z: hand x is world y.
HAND_QUAT = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
OFFSET = np.array([[1.0, 0.0, 0.0, 0.05],
                   [0.0, -1.0, 0.0, 0.0],
                   [0.0, 0.0, -1.0, -0.08],
                   [0.0, 0.0, 0.0, 1.0]])


class _GenesisCamera:
    def render(self, **_):
        return np.full((480, 640, 3), 7, np.float32), None, None, None


class _Hand:
    def get_pos(self):
        return torch.tensor(HAND_POS)

    def get_quat(self):
        return torch.tensor(HAND_QUAT)


def _env(renderer):
    return SimEnv(scene=object(), franka=None, scene_config=SCENE, cam_up=_GenesisCamera(),
                  cam_wrist=_GenesisCamera(), wrist_offset_T=torch.tensor(OFFSET), renderer=renderer,
                  end_effector=_Hand())


@pytest.fixture
def fake_hiprt(monkeypatch):
    made = []

    class HiprtRenderer:
        def __init__(self, eye, lookat, up, fovy_deg, resolution, path_samples):
            self.fovy_deg, self.views = fovy_deg, []
            made.append(self)

        def bind(self, scene):
            return self

        def render_views(self, views):
            self.views.extend(views)
            return torch.full((len(views), 480, 640, 3), 0.5)

    module = ModuleType("instinct_render.genesis")
    module.HiprtRenderer = HiprtRenderer
    monkeypatch.setitem(sys.modules, "instinct_render", ModuleType("instinct_render"))
    monkeypatch.setitem(sys.modules, "instinct_render.genesis", module)
    return made


def test_unset_renderer_is_genesis_and_an_unknown_one_is_refused(monkeypatch):
    monkeypatch.delenv("ROBOSMITH_RENDERER", raising=False)
    assert renderer_from_env() == "genesis"
    monkeypatch.setenv("ROBOSMITH_RENDERER", "hip-raster")
    with pytest.raises(ValueError):
        renderer_from_env()


def test_genesis_backend_renders_through_the_genesis_camera():
    rgb = _env("genesis").render("up")
    assert rgb.dtype == np.uint8 and rgb.shape == (480, 640, 3) and int(rgb[0, 0, 0]) == 7


def test_hiprt_backend_without_instinct_render_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "instinct_render", None)
    env = _env("hiprt")
    with pytest.raises(ImportError):
        env.render("up")
    assert env.render_reading()["cameras"] == {}


def test_hiprt_wrist_view_is_the_hand_pose_times_the_one_offset(fake_hiprt):
    env = _env("hiprt")
    env.render("wrist")
    env.render("wrist")

    assert len(fake_hiprt) == 1 and fake_hiprt[0].fovy_deg == 65.0
    eye, lookat, up = (np.asarray(v) for v in fake_hiprt[0].views[-1])
    # Offset x (+0.05 along hand x) is world +y; the camera looks along minus the offset's z column, world +z here.
    np.testing.assert_allclose(eye, HAND_POS + [0.0, 0.05, -0.08], atol=1e-9)
    np.testing.assert_allclose(lookat - eye, [0.0, 0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(up, [1.0, 0.0, 0.0], atol=1e-9)


def test_reading_names_the_backend_that_rendered_and_only_cameras_with_frames(fake_hiprt):
    env = _env("hiprt")
    rgb = env.render("up")
    reading = env.render_reading()

    assert int(rgb[0, 0, 0]) == 128
    assert reading["renderer"] == "hiprt"
    assert set(reading["cameras"]) == {"up"} and reading["cameras"]["up"]["frames"] == 1
