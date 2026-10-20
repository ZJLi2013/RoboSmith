"""Unit tests for the rocRobo motion backend.

Node-independent: World builder (URDF parse + live-pose transform), trajectory
resampling, and the MotionPlanner fallback/padding logic with a fake serve
client. The actual collision-free solve runs in the GPU e2e, not here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robotsmith.assets.schema import (
    Asset,
    AssetMetadata,
    UrdfJoint,
    parse_urdf_collision_boxes,
)
from robotsmith.motion.planner import GenesisBackend, PlanResult, Waypoint
from robotsmith.motion.rocrobo_backend import RocRoboBackend, _resample
from robotsmith.motion.rocrobo_world import (
    build_collision_world,
    build_obstacle_world,
)

_REPO = Path(__file__).resolve().parents[1]
_DRAWER_URDF = _REPO / "assets" / "articraft" / "drawer_cabinet" / "model.urdf"


def _drawer_asset() -> Asset:
    return Asset(
        name="drawer_cabinet",
        root_dir=_DRAWER_URDF.parent,
        urdf_path=_DRAWER_URDF,
        metadata=AssetMetadata(),
        joints=[
            UrdfJoint(
                name="drawer_slide",
                type="prismatic",
                parent="base",
                child="drawer",
                axis=(1.0, 0.0, 0.0),
                lower=0.0,
                upper=0.35,
            ),
            # Handle on its own fixed-jointed link (mirrors the real URDF); the
            # world builder must carry it along the slide via _rigid_moving_links.
            UrdfJoint(
                name="handle_mount",
                type="fixed",
                parent="drawer",
                child="drawer_handle",
            ),
        ],
    )


# --- URDF collision box parsing -------------------------------------------

def test_parse_collision_boxes_has_base_and_drawer_links():
    boxes = parse_urdf_collision_boxes(_DRAWER_URDF)
    links = {b.link for b in boxes}
    assert "base" in links and "drawer" in links
    # Spherical handle (on the fixed-jointed drawer_handle link) becomes a 2r cube
    # (radius 0.025 -> 0.05 side).
    handle = [b for b in boxes if b.link == "drawer_handle" and np.allclose(b.size, (0.05, 0.05, 0.05))]
    assert handle, "spherical handle should be approximated by a 0.05 box"


# --- World builder ----------------------------------------------------------

def test_build_collision_world_drawer_slides_along_axis():
    asset = _drawer_asset()
    pos = (0.8, 0.0, 0.775)
    quat = (1.0, 0.0, 0.0, 0.0)  # identity
    closed = build_collision_world(asset, pos, quat, 1.0, joint_value=0.0, moving_link="drawer")
    opened = build_collision_world(asset, pos, quat, 1.0, joint_value=0.35, moving_link="drawer")
    # Build order matches parse_urdf_collision_boxes, so we can attribute each
    # delta to its source link and assert per-link (not just "something moved").
    box_links = [b.link for b in parse_urdf_collision_boxes(_DRAWER_URDF)]
    deltas = [np.array(o["center"]) - np.array(c["center"]) for c, o in zip(closed, opened)]
    assert len(box_links) == len(deltas)
    by_link: dict[str, list] = {}
    for link, d in zip(box_links, deltas):
        by_link.setdefault(link, []).append(d)
    # The drawer body AND the fixed-jointed handle ride the slide; base stays put.
    # The handle case is the regression guard: a separated handle link must follow
    # via _rigid_moving_links, not be stranded at the closed pose.
    for link in ("drawer", "drawer_handle"):
        assert by_link.get(link), f"expected collision boxes on {link!r}"
        for d in by_link[link]:
            assert np.allclose(d, [0.35, 0.0, 0.0], atol=1e-6), f"{link} box must slide"
    for d in by_link.get("base", []):
        assert np.linalg.norm(d) < 1e-9, "base boxes stay put"


def test_build_collision_world_scale_and_ground():
    asset = _drawer_asset()
    world = build_collision_world(
        asset, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), 0.5, joint_value=0.0,
        moving_link="drawer", ground_z=0.75,
    )
    assert any(p["type"] == "halfspace" and p["point"][2] == 0.75 for p in world)
    # extent = FULL size (serve halves it): 0.45-wide box at scale 0.5 -> 0.225
    bottom = next(b for b in world if np.isclose(b["extent"][0], 0.225))
    assert bottom


def test_build_collision_world_exempt_link_dropped():
    asset = _drawer_asset()
    full = build_collision_world(asset, (0, 0, 0), (1, 0, 0, 0), 1.0, moving_link="drawer")
    no_drawer = build_collision_world(
        asset, (0, 0, 0), (1, 0, 0, 0), 1.0, moving_link="drawer", exempt_links=["drawer"]
    )
    assert all("box" == p["type"] for p in no_drawer)
    assert len(no_drawer) < len(full)


def test_build_obstacle_world_from_scene_state():
    asset = _drawer_asset()
    scene_state = {
        "assets": {"drawer": asset, "die": None},
        "positions": {"drawer": np.array([0.8, 0.0, 0.775]), "die": np.array([0.45, 0.32, 0.8])},
        "object_quats": {"drawer": (1.0, 0.0, 0.0, 0.0)},
        "object_scales": {"drawer": 1.0},
        "joint_positions": {"drawer": {"drawer_slide": 0.3}},
    }
    world = build_obstacle_world(scene_state, ground_z=0.75)
    assert any(p["type"] == "halfspace" for p in world)
    assert any(p["type"] == "box" for p in world)
    # The rigid die (asset None) must not contribute obstacles.
    # Only the articulated cabinet + ground are present.
    n_cabinet = len(parse_urdf_collision_boxes(_DRAWER_URDF))
    assert sum(1 for p in world if p["type"] == "box") == n_cabinet


# --- Resampling -------------------------------------------------------------

def test_resample_time_based_preserves_endpoints():
    traj = np.array([[0.0] * 7, [1.0] * 7, [2.0] * 7])  # 3 pts @ src_dt=0.1 -> 0.2s
    out = _resample(traj, src_dt=0.1, dst_dt=0.05)
    assert len(out) == 5  # 0.2s / 0.05 + 1
    assert np.allclose(out[0], 0.0)
    assert np.allclose(out[-1], 2.0)
    assert np.allclose(out[2], 1.0)  # midpoint


def test_resample_no_src_dt_returns_points():
    traj = np.zeros((4, 7))
    assert len(_resample(traj, src_dt=None, dst_dt=0.033)) == 4


# --- Backend: fallback + padding -------------------------------------------

class _FakeClient:
    def __init__(self, response, available=True):
        self._response = response
        self.available = available
        self.requests = []

    def request(self, payload):
        self.requests.append(payload)
        return self._response


def _fake_genesis_solve_ik(pos, quat, finger, init_qpos=None):
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, finger, finger])


def test_genesis_backend_plan_motion_unsupported():
    gb = GenesisBackend(_fake_genesis_solve_ik)
    res = gb.plan_motion(np.zeros(9), [Waypoint(np.zeros(3), np.array([1, 0, 0, 0]), 0.0)])
    assert isinstance(res, PlanResult) and not res.success


@pytest.mark.parametrize("response, world", [
    ({}, None),
    ({"ok": False, "reason": "no-solution"}, [{"type": "box"}]),
])
def test_solve_ik_falls_back_to_genesis(response, world):
    backend = RocRoboBackend(_fake_genesis_solve_ik, client=_FakeClient(response))
    q = backend.solve_ik(np.zeros(3), np.array([1, 0, 0, 0]), 0.04, world=world)
    assert np.allclose(q, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.04, 0.04])


def test_solve_ik_uses_serve_and_pads_fingers():
    client = _FakeClient({"ok": True, "q": [0, 0, 0, 0, 0, 0, 0], "quality": {}})
    backend = RocRoboBackend(_fake_genesis_solve_ik, client=client)
    q = backend.solve_ik(np.zeros(3), np.array([1, 0, 0, 0]), 0.02, world=[{"type": "box"}])
    assert q.shape == (9,)
    assert np.isclose(q[7], 0.02) and np.isclose(q[8], 0.02)
    assert client.requests[0]["op"] == "solve_ik"


def test_plan_motion_success_pads_and_resamples():
    # 3 retimed pts over total_s=0.2 -> src_dt 0.1; resample @ 0.05 -> 5 pts.
    traj7 = [[0.0] * 7, [1.0] * 7, [2.0] * 7]
    client = _FakeClient(
        {"ok": True, "traj": traj7, "timed": {"q": traj7, "total_s": 0.2, "vel_ok": True}}
    )
    backend = RocRoboBackend(_fake_genesis_solve_ik, client=client)
    res = backend.plan_motion(
        np.zeros(9),
        [Waypoint(np.zeros(3), np.array([1, 0, 0, 0]), 0.03)],
        world=[{"type": "box"}],
        control_dt=0.05,
    )
    assert res.success
    assert all(q.shape == (9,) for q in res.trajectory)
    assert np.isclose(res.trajectory[0][7], 0.03)  # finger held
    assert len(res.trajectory) == 5  # resampled 0.2s @ 0.05


def test_plan_motion_uses_raw_traj_when_no_timing():
    traj7 = [[0.0] * 7, [1.0] * 7]
    client = _FakeClient({"ok": True, "traj": traj7, "timed": None})
    backend = RocRoboBackend(_fake_genesis_solve_ik, client=client)
    res = backend.plan_motion(
        np.zeros(9), [Waypoint(np.zeros(3), np.array([1, 0, 0, 0]), 0.0)],
        world=[{"type": "box"}],
    )
    assert res.success and len(res.trajectory) == 2


@pytest.mark.parametrize("response, world", [
    ({"ok": False, "reason": "infeasible"}, [{"type": "box"}]),
    ({}, None),
])
def test_plan_motion_unsuccessful(response, world):
    backend = RocRoboBackend(_fake_genesis_solve_ik, client=_FakeClient(response))
    res = backend.plan_motion(
        np.zeros(9), [Waypoint(np.zeros(3), np.array([1, 0, 0, 0]), 0.0)], world=world,
    )
    assert not res.success
    if response:
        assert "infeasible" in res.reason
