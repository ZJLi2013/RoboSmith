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
from robotsmith.execution.gate import ExecutionFailed, Refused
from robotsmith.motion.joint_planner import JointPlanner
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
        self.grasp_quat = np.array([0.0, 1.0, 0.0, 0.0])
        self.pre_grasp_pos = np.array([0.5, 0.0, 0.2])
        self.pre_grasp_quat = self.grasp_quat
        self.retreat_pos = None
        self.metadata = {}


class _FakePlanner:
    def plan(self, *a, **k):
        return [_FakePlan()]

    def plan_place(self, *a, **k):
        return _FakePlan()


def _chain(traj, holding=False, allow=None):
    return SimpleNamespace(trajectory=traj, segments=[
        {"label": "seg", "span": [0, len(traj)], "holding": holding, "allow": allow}])


def _frames(motions):
    return [q for m in motions for q in m.trajectory]


class _FakeExecutor:
    def __init__(self):
        self.calls = []
        self.last_motion_trace = None

    def pick(self, *a, **k):
        self.calls.append("pick")
        return _chain([np.full(9, 1.0)])

    def place(self, *a, **k):
        self.calls.append("place")
        return _chain([np.full(9, 2.0)], holding=True)

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
    def test_dispatch_pick_place_open_close(self, gate):
        ex = _FakeExecutor()
        skills = [
            Skill("pick", "die"),
            Skill("place", "die"),
            Skill("open", "drawer"),
            Skill("close", "drawer"),
        ]
        motions = run_skills(skills, _FakePlanner(), ex, lambda *a, **k: np.zeros(9),
                             _scene_state(), MotionParams(), gate=gate)
        traj = _frames(motions)
        assert ex.calls == ["pick", "place", ("drag", "open"), ("drag", "close")]
        assert [m.gated for m in motions] == [True, True, False, False]
        # Each fake segment is tagged by a distinct constant qpos (pick=1, place=2,
        # drag=3). place prepends its planned transport, so assert the markers and
        # their order rather than an exact length.
        markers = [t[-1] for t in traj if np.allclose(t, t[-1])]
        assert markers == [1.0, 2.0, 3.0, 3.0]

    def test_pick_with_every_candidate_filtered_out_records_counts_and_is_refused(self, gate):
        class EmptyPlanner(_FakePlanner):
            last_filter_counts = {"raw": 8, "z_reject": 8, "accepted": 0}

            def plan(self, *a, **k):
                return []

        ex, state = _FakeExecutor(), _scene_state()
        with pytest.raises(Refused):
            run_skills([Skill("pick", "die")], EmptyPlanner(), ex, lambda *a, **k: np.zeros(9),
                       state, MotionParams(), gate=gate)
        assert ex.calls == []
        trace = state["skill_traces"][0]
        assert trace["no_candidates"] and trace["filter_counts"]["z_reject"] == 8

    def _trial_run(self, gate, lifted, try_all=False):
        class ThreePlans(_FakePlanner):
            def plan(self, *a, **k):
                plans = [_FakePlan() for _ in range(3)]
                for i, p in enumerate(plans):
                    p.metadata = {"candidate_index": i, "kind": "top_down"}
                return plans

        calls = []

        def trial(traj, target):
            calls.append(target)
            gate.env.writes += 1
            return {"lifted": lifted[len(calls) - 1], "lift_m": 0.1, "seconds": 0.5}

        ex, state = _FakeExecutor(), _scene_state()
        state["grasp_trial"], state["grasp_trial_all"] = trial, try_all
        try:
            motions = run_skills([Skill("pick", "die")], ThreePlans(), ex, lambda *a, **k: np.zeros(9), state,
                                 MotionParams(), gate=gate)
        except (Refused, ExecutionFailed):
            motions = None
        return motions, state["skill_traces"][0], calls

    def test_trial_keeps_the_first_candidate_that_lifts(self, gate):
        motions, trace, calls = self._trial_run(gate, [False, True, True])
        assert calls == ["die", "die"] and trace["selected_candidate_index"] == 1
        assert [t["lifted"] for t in trace["trials"]] == [False, True] and trace["trial_seconds"] == 1.0
        assert np.allclose(motions[0].trajectory[-1], 1.0)

    def test_trial_runs_between_two_gate_passes(self, gate):
        motions, _, _ = self._trial_run(gate, [True])
        assert motions[0].fingerprint == gate.env.state_fingerprint() == "1"
        gate.check_current(motions[0])

    def test_trial_counts_a_candidate_whose_motion_fails_as_not_lifted(self, gate, monkeypatch):
        def broken(*a, **k):
            raise RuntimeError("wp1 IK discontinuity")

        monkeypatch.setattr(_FakeExecutor, "pick", broken)
        motions, trace, calls = self._trial_run(gate, [True, True, True])
        assert motions is None and calls == [] and trace["no_trial_success"]
        assert [t["motion_error"] for t in trace["trials"]] == ["wp1 IK discontinuity"] * 3

    def test_a_candidate_the_gate_rejects_is_not_tried(self, gate, monkeypatch):
        seen = []

        def contacts_at(q, held=None, placed=None):
            seen.append(float(q[0]))
            return [(("robot", "hand"), ("wall", "base"))] if len(seen) == 1 else []

        monkeypatch.setattr(gate.env, "contacts_at", contacts_at, raising=False)
        motions, trace, calls = self._trial_run(gate, [True, True, True])
        assert calls == ["die"] and trace["selected_candidate_index"] == 1
        assert trace["trials"][0]["gate"] == {"segment": "seg", "index": 0, "reason": "collision",
                                              "pair": [["robot", "hand"], ["wall", "base"]]}

    def test_trial_all_tries_every_candidate_and_none_lifting_fails_the_step(self, gate):
        _, trace, calls = self._trial_run(gate, [False, True, True], try_all=True)
        assert len(calls) == 3 and trace["selected_candidate_index"] == 1
        motions, trace, _ = self._trial_run(gate, [False, False, False])
        assert motions is None and trace["no_trial_success"] and trace["selected_candidate_index"] is None

    def test_trial_all_verifies_the_chosen_motion_after_the_last_trial(self, gate):
        motions, trace, calls = self._trial_run(gate, [True, True, True], try_all=True)
        assert len(calls) == 3 and trace["selected_candidate_index"] == 0
        assert motions[0].fingerprint == gate.env.state_fingerprint() == "3"
        gate.check_current(motions[0])

    def test_trial_all_skips_a_lifter_the_final_check_rejects(self, gate, monkeypatch):
        rejected = []

        def contacts_at(q, held=None, placed=None):
            if gate.env.writes == 3 and not rejected:
                rejected.append(float(q[0]))
                return [(("robot", "hand"), ("wall", "base"))]
            return []

        monkeypatch.setattr(gate.env, "contacts_at", contacts_at, raising=False)
        motions, trace, _ = self._trial_run(gate, [True, True, False], try_all=True)
        assert trace["selected_candidate_index"] == 1 and rejected
        assert trace["trials"][0]["reverify"]["pair"] == [["robot", "hand"], ["wall", "base"]]
        gate.check_current(motions[0])

    def _pick_place(self, gate, ex):
        state = {**_scene_state(), "positions": {"die": np.array([0.5, 0.0, 0.0]), "mat": np.array([0.4, -0.2, 0.0])}}
        return run_skills([Skill("pick", "die"), Skill("place", "mat")], _FakePlanner(), ex,
                          lambda *a, **k: np.zeros(9), state, MotionParams(), gate=gate), state

    def test_place_carries_the_held_object_along_the_planned_connection(self, gate, monkeypatch):
        starts = []
        monkeypatch.setattr(_FakeExecutor, "place",
                            lambda self, plan, mp, start, *a, **k: starts.append(np.array(start)) or
                            _chain([np.full(9, 2.0)], holding=True))
        motions, state = self._pick_place(gate, _FakeExecutor())
        q0, q_pre = np.full(9, 1.0), np.array([0.0] * 7 + [1.0, 1.0])
        line = JointPlanner(gate.env, gate).connect(q0, q_pre, target="die", label="transport").motion.trajectory
        place = motions[1]
        assert np.allclose(place.trajectory[:len(line)], line) and np.allclose(starts[0], q_pre)
        assert place.spans[0].label == "transport" and place.spans[0].holding
        assert state["skill_traces"][1]["transport"]["ok"] and state["skill_traces"][1]["transport"]["direct"]

    def test_a_place_whose_connection_is_refused_steps_nothing(self, gate, monkeypatch):
        monkeypatch.setattr(gate.env, "contacts_at",
                            lambda q, held=None, placed=None:
                            [(("die", "base"), ("barrier", "base"))] if held and np.allclose(q[:7], 0.0) else [],
                            raising=False)
        ex = _FakeExecutor()
        with pytest.raises(Refused) as exc:
            self._pick_place(gate, ex)
        assert exc.value.label == "search_exhausted" and ex.calls == ["pick"] and gate.env.writes == 0
        transport = exc.value.evidence["transport"]
        assert not transport["ok"] and transport["queries"] >= 1 and transport["reason"]

    def test_place_lets_the_held_object_rest_on_the_target_entity_of_a_surface(self, gate, monkeypatch):
        monkeypatch.setattr(_FakeExecutor, "place",
                            lambda self, *a, **k: _chain([np.full(9, 0.3)], holding=True, allow="rest"))
        monkeypatch.setattr(gate.env, "contacts_at",
                            lambda q, held=None, placed=None:
                            [(("die", "base"), ("mat", "base"))] if np.allclose(q, 0.3) else [],
                            raising=False)
        motions = run_skills([Skill("pick", "die"), Skill("place", "mat.top")], _FakePlanner(), _FakeExecutor(),
                             lambda *a, **k: np.zeros(9), {**_scene_state(), "positions": {
                                 "die": np.array([0.5, 0.0, 0.0]), "mat.top": np.array([0.4, -0.2, 0.0])}},
                             MotionParams(), gate=gate)
        assert len(motions) == 2

    def test_unknown_skill_raises(self, gate):
        with pytest.raises(ValueError):
            run_skills([Skill("teleport", "die")], _FakePlanner(), _FakeExecutor(),
                       lambda *a, **k: np.zeros(9), _scene_state(), MotionParams(), gate=gate)


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
