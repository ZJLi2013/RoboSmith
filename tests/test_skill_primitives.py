"""SkillPrimitive registry + open/close primitives.

Validates dispatch, the handle-drag resolution, and that the open/close drag
motion produces a valid joint trajectory (with a fake IK solver). Contact-based
opening needs Genesis and is covered by the GPU end-to-end run.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from robotsmith.assets.schema import UrdfJoint
from robotsmith.execution.executor import MotionExecutor
from robotsmith.motion.params import MotionParams
from robotsmith.motion.planner import GenesisBackend
from robotsmith.skills import Skill, run_skills
from robotsmith.assets.geometry import rotate_vec_wxyz
from robotsmith.skills.frames import (
    _resolve_handle_drag,
    resolve_frame,
)


def _drawer_asset():
    meta = SimpleNamespace(
        handles=[{"link": "drawer", "local_pos": [0.27, 0.0, 0.33]}],
        place_targets=[
            {"name": "drawer_opening", "joint": "drawer_slide", "lip_local": [0.2, 0.0, 0.1]}
        ],
    )
    joint = UrdfJoint("drawer_slide", "prismatic", "base", "drawer", (1.0, 0.0, 0.0), 0.0, 0.35)
    return SimpleNamespace(
        name="drawer",
        metadata=meta,
        movable_joints=[joint],
        get_joint=lambda n: joint if n == "drawer_slide" else None,
    )


def test_rotate_vec_yaw_90():
    # +90deg about Z (wxyz = [cos45, 0,0, sin45]) maps +X -> +Y.
    q = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    assert np.allclose(rotate_vec_wxyz(q, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-6)


class TestResolveHandleDrag:
    def _ctx(self, opening_quat=(1.0, 0.0, 0.0, 0.0)):
        scene_state = {
            "positions": {"drawer": np.array([0.55, 0.0, 0.0])},
            "assets": {"drawer": _drawer_asset()},
            "object_quats": {"drawer": list(opening_quat)},
        }
        return SimpleNamespace(skill=Skill("open", "drawer"), scene_state=scene_state, obj_pos=scene_state["positions"]["drawer"])

    def test_open_handle_and_move(self):
        ctx = self._ctx()
        handle, move, _ = _resolve_handle_drag(ctx, opening=True)
        assert np.allclose(handle, [0.55 + 0.27, 0.0, 0.33])
        assert np.allclose(move, [0.35, 0.0, 0.0])  # full travel along +X

    def test_close_starts_open_and_moves_back(self):
        ctx = self._ctx()
        ctx.skill = Skill("close", "drawer")
        handle, move, _ = _resolve_handle_drag(ctx, opening=False)
        # close starts from the open handle position and drags back toward closed.
        assert np.allclose(handle, [0.55 + 0.27 + 0.35, 0.0, 0.33])
        assert np.allclose(move, [-0.35, 0.0, 0.0])

    def test_distance_override(self):
        ctx = self._ctx()
        ctx.skill = Skill("open", "drawer", params={"distance": 0.2})
        _, move, _ = _resolve_handle_drag(ctx, opening=True)
        assert np.allclose(move, [0.2, 0.0, 0.0])


class TestResolveFrame:
    """Single seam: a frame self-describes its anchor; closed-loop is a property
    of resolution, not opt-in per primitive. The invariant locks this in."""

    def _state(self, slide):
        return {
            "positions": {"drawer": np.array([0.55, 0.0, 0.0])},
            "assets": {"drawer": _drawer_asset()},
            "object_quats": {"drawer": [1.0, 0.0, 0.0, 0.0]},
            "joint_positions": {"drawer": {"drawer_slide": slide}},
        }

    def test_world_frame_is_static_across_slide(self):
        frame = SimpleNamespace(kind="world", xyz=(0.5, 0.1, 0.8))
        assert np.allclose(resolve_frame(frame, self._state(0.0)), [0.5, 0.1, 0.8])
        assert np.allclose(resolve_frame(frame, self._state(0.3)), [0.5, 0.1, 0.8])

    def test_articulated_frame_tracks_live_slide(self):
        frame = SimpleNamespace(
            kind="articulated", parent="drawer", joint="drawer_slide",
            local_offset=(0.27, 0.0, 0.33),
        )
        at_closed = resolve_frame(frame, self._state(0.0))
        at_open = resolve_frame(frame, self._state(0.3))
        assert np.allclose(at_closed, [0.55 + 0.27, 0.0, 0.33])
        assert np.allclose(at_open, [0.55 + 0.27 + 0.30, 0.0, 0.33])
        assert not np.allclose(at_closed, at_open)

    def test_articulated_opening_tracks_exposed_midpoint(self):
        # Opening anchor rides only half the slide (exposed-opening midpoint) at
        # the tray-floor z, sourced from the asset's place_targets metadata.
        frame = SimpleNamespace(
            kind="articulated_opening", parent="drawer", opening="drawer_opening",
        )
        at_closed = resolve_frame(frame, self._state(0.0))
        at_open = resolve_frame(frame, self._state(0.3))
        # lip = parent(0.55) + lip_local.x(0.2) = 0.75; z = lip_local.z = 0.1.
        assert np.allclose(at_closed, [0.75, 0.0, 0.1])
        # half-slide: 0.75 + 0.3 * 0.5 = 0.90 (vs full-travel 1.05 for "articulated").
        assert np.allclose(at_open, [0.90, 0.0, 0.1])


class _FakePlan:
    def __init__(self):
        self.grasp_pos = np.array([0.5, 0.0, 0.12])
        self.pre_grasp_pos = None
        self.retreat_pos = None
        self.metadata = {}


class _FakePlanner:
    def plan(self, *a, **k):
        return [_FakePlan()]

    def plan_place(self, *a, **k):
        return _FakePlan()


class _FakeExecutor:
    def __init__(self):
        self.calls = []
        self.last_motion_trace = None

    def pick(self, *a, **k):
        self.calls.append("pick")
        return [np.full(9, 1.0)]

    def place(self, *a, **k):
        self.calls.append("place")
        return [np.full(9, 2.0)]

    def drag_handle(self, *a, **k):
        self.calls.append(("drag", k.get("action")))
        return [np.full(9, 3.0)]


def _scene_state():
    return {
        "home_qpos": np.zeros(9),
        "positions": {"die": np.array([0.5, 0.0, 0.0]), "drawer": np.array([0.55, 0.0, 0.0])},
        "assets": {"drawer": _drawer_asset()},
        "object_quats": {"drawer": [1.0, 0.0, 0.0, 0.0]},
        "table_surface_z": 0.0,
    }


class TestRegistryDispatch:
    def test_dispatch_pick_place_open_close(self):
        ex = _FakeExecutor()
        skills = [
            Skill("pick", "die"),
            Skill("place", "die"),
            Skill("open", "drawer"),
            Skill("close", "drawer"),
        ]
        traj = run_skills(skills, _FakePlanner(), ex, lambda *a, **k: np.zeros(9),
                          _scene_state(), MotionParams())
        assert ex.calls == ["pick", "place", ("drag", "open"), ("drag", "close")]
        # Each fake segment is tagged by a distinct constant qpos (pick=1, place=2,
        # drag=3). place may prepend a base-aim transit, so assert the markers and
        # their order rather than an exact length.
        markers = [t[-1] for t in traj if np.allclose(t, t[-1])]
        assert markers == [1.0, 2.0, 3.0, 3.0]

    def test_pick_with_every_candidate_filtered_out_records_counts_and_holds(self):
        class EmptyPlanner(_FakePlanner):
            last_filter_counts = {"raw": 8, "z_reject": 8, "accepted": 0}

            def plan(self, *a, **k):
                return []

        ex, state = _FakeExecutor(), _scene_state()
        traj = run_skills([Skill("pick", "die")], EmptyPlanner(), ex, lambda *a, **k: np.zeros(9),
                          state, MotionParams())
        assert ex.calls == [] and len(traj) > 0
        trace = state["skill_traces"][0]
        assert trace["no_candidates"] and trace["filter_counts"]["z_reject"] == 8

    def _trial_run(self, lifted, try_all=False):
        class ThreePlans(_FakePlanner):
            def plan(self, *a, **k):
                plans = [_FakePlan() for _ in range(3)]
                for i, p in enumerate(plans):
                    p.metadata = {"candidate_index": i, "kind": "top_down"}
                return plans

        calls = []

        def trial(traj, target):
            calls.append(target)
            return {"lifted": lifted[len(calls) - 1], "lift_m": 0.1, "seconds": 0.5}

        ex, state = _FakeExecutor(), _scene_state()
        state["grasp_trial"], state["grasp_trial_all"] = trial, try_all
        traj = run_skills([Skill("pick", "die")], ThreePlans(), ex, lambda *a, **k: np.zeros(9), state, MotionParams())
        return traj, state["skill_traces"][0], calls

    def test_trial_keeps_the_first_candidate_that_lifts(self):
        traj, trace, calls = self._trial_run([False, True, True])
        assert calls == ["die", "die"] and trace["selected_candidate_index"] == 1
        assert [t["lifted"] for t in trace["trials"]] == [False, True] and trace["trial_seconds"] == 1.0
        assert np.allclose(traj[-1], 1.0)

    def test_trial_all_tries_every_candidate_and_no_success_holds(self):
        _, trace, calls = self._trial_run([False, True, True], try_all=True)
        assert len(calls) == 3 and trace["selected_candidate_index"] == 1
        traj, trace, _ = self._trial_run([False, False, False])
        assert trace["no_trial_success"] and trace["selected_candidate_index"] is None
        assert not np.allclose(traj[-1], 1.0)

    def test_unknown_skill_raises(self):
        with pytest.raises(ValueError):
            run_skills([Skill("teleport", "die")], _FakePlanner(), _FakeExecutor(),
                       lambda *a, **k: np.zeros(9), _scene_state(), MotionParams())


class TestDragHandleMotion:
    def test_open_drag_trajectory(self):
        # Linear pos->joint map keeps the cartesian drag continuous.
        def solve_ik(pos, quat, finger, init_qpos=None):
            p = np.asarray(pos, dtype=np.float64)
            return np.array([p[0], p[1], p[2], 0.0, 0.0, 0.0, 0.0, finger, finger])

        ex = MotionExecutor()
        traj = ex.drag_handle(
            np.array([0.55, 0.0, 0.33]), np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([0.30, 0.0, 0.0]), GenesisBackend(solve_ik), np.zeros(9),
            MotionParams(), action="open",
        )
        assert len(traj) > 0
        assert ex.last_motion_trace["action"] == "open"
        # The hand ends translated by +0.30 in x then lifted (retreat height).
        assert traj[-1][0] == pytest.approx(0.55 + 0.30, abs=1e-6)
