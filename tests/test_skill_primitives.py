"""SkillPrimitive registry + open/close primitives.

Validates dispatch, the handle-drag resolution, and that the open/close drag
motion produces a valid joint trajectory (with a fake IK solver). Contact-based
opening needs Genesis and is covered by the GPU end-to-end run.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robotsmith.assets.schema import UrdfJoint
from robotsmith.execution.executor import MotionExecutor
from robotsmith.motion.params import MotionParams
from robotsmith.motion.planner import GenesisBackend
from robotsmith.execution.gate import ExecutionFailed, Refused
from robotsmith.grasp.transforms import pose_matrix
from robotsmith.motion.joint_planner import JointPlanner
from robotsmith.motion.plan_state import Attach, Detach
from robotsmith.scenario.load import read_plan_file, steps_to_skills
from robotsmith.grasp.geometric import GeometricGraspPlanner
from robotsmith.scenario_runtime import runner
from robotsmith.scenario_runtime.runner import plan_skill
from robotsmith.skills import Skill, SkillContext
from robotsmith.skills.manipulation_tools import ManipulationTools
from robotsmith.skills.rule_policy import RulePolicy
from robotsmith.assets.geometry import rotate_vec_wxyz
from conftest import pick_skill, place_skill, plan_skills
from robotsmith.skills.frames import (
    _resolve_handle_drag,
    resolve_frame,
)


PLANS = Path(__file__).resolve().parents[1] / "scenarios" / "acceptance" / "plans"


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
        self.retreat_pos = self.pre_grasp_pos
        self.retreat_quat = self.grasp_quat
        self.finger_open, self.finger_closed = 0.04, 0.0
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

    def run(self, name, actions, motion_planner, start, params, **k):
        self.calls.append(name)
        return getattr(self, name)(actions, start)

    def pick(self, actions, start):
        """The list's planned approach, then the pick's marker point."""
        approach = [np.asarray(q) for q in actions[1].path]
        n = len(approach)
        return SimpleNamespace(trajectory=approach + [np.full(9, 1.0)], segments=[
            {"label": "approach", "span": [0, n], "holding": False, "allow": None},
            {"label": "seg", "span": [n, n + 1], "holding": False, "allow": None}])

    def place(self, actions, start):
        return _chain([np.full(9, 2.0)], holding=True)

    def drag_handle(self, *a, **k):
        self.calls.append(("drag", k.get("action")))
        return [np.full(9, 3.0)]


class _WaypointExecutor(_FakeExecutor):
    """The pick's planned approach, then one point per motion of the list: its position and finger width."""

    @staticmethod
    def _points(actions):
        return [np.r_[np.asarray(a.pos, dtype=np.float64), np.zeros(4), a.finger_width, a.finger_width]
                for a in actions if not isinstance(a, (Attach, Detach))]

    def pick(self, actions, start):
        approach, points = [np.asarray(q) for q in actions[1].path], self._points(actions)
        n = len(approach)
        return SimpleNamespace(trajectory=approach + points, segments=[
            {"label": "approach", "span": [0, n], "holding": False, "allow": None},
            {"label": "seg", "span": [n, n + len(points)], "holding": False, "allow": None}])

    def place(self, actions, start):
        return _chain(self._points(actions), holding=True)


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
            pick_skill("die"),
            place_skill("die"),
            Skill("open", "drawer"),
            Skill("close", "drawer"),
        ]
        motions = plan_skills(skills, _FakePlanner(), ex, lambda *a, **k: np.zeros(9),
                             _scene_state(), MotionParams(), gate=gate)
        traj = _frames(motions)
        # pick runs its list from the pre-grasp, then whole; the first place is pick's check that the place is
        # reachable from where the pick ends.
        assert ex.calls == ["pick", "pick", "place", "place", ("drag", "open"), ("drag", "close")]
        assert [m.gated for m in motions] == [True, True, False, False]
        # Each fake segment is tagged by a distinct constant qpos (pick=1, place=2,
        # drag=3). pick and place prepend their planned approach and transport, so assert
        # the markers and their order rather than an exact length.
        markers = [t[-1] for t in traj if np.allclose(t, t[-1])]
        assert markers == [0.0, 1.0, 2.0, 3.0, 3.0]

    def test_pick_without_candidates_is_refused(self, gate):
        class EmptyPlanner(_FakePlanner):
            def plan(self, *a, **k):
                return []

        ex, state = _FakeExecutor(), _scene_state()
        with pytest.raises(Refused):
            plan_skills([pick_skill("die")], EmptyPlanner(), ex, lambda *a, **k: np.zeros(9),
                       state, MotionParams(), gate=gate)
        assert ex.calls == []
        trace = state["skill_traces"][0]
        assert trace["no_candidates"]

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
        try:
            motions = plan_skills([pick_skill("die")], ThreePlans(), ex, lambda *a, **k: np.zeros(9), state,
                                 MotionParams(), gate=gate, trial=trial, try_all=try_all)
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
        return plan_skills([pick_skill("die"), place_skill("mat")], _FakePlanner(), ex,
                          lambda *a, **k: np.zeros(9), state, MotionParams(), gate=gate), state

    def _pick_only(self, gate, plans, ik, trial=None):
        class Plans(_FakePlanner):
            def plan(self, *a, **k):
                return plans

        ex, state = _FakeExecutor(), _scene_state()
        starts = []
        pick = ex.pick
        ex.pick = lambda actions, start: starts.append(np.array(start.q)) or pick(actions, start)
        try:
            motions = plan_skills([pick_skill("die")], Plans(), ex, ik, state, MotionParams(), gate=gate, trial=trial)
        except (Refused, ExecutionFailed):
            motions = None
        return motions, state["skill_traces"][0], starts

    def _pick_then_place(self, gate, plans, trial=None, frames=None):
        class Plans(_FakePlanner):
            def plan(self, *a, **k):
                return plans

        state = {**_scene_state(), "positions": {"die": np.array([0.5, 0.0, 0.0]), "mat": np.array([0.4, -0.2, 0.0])},
                 "target_frames": frames or {}}
        try:
            motions = plan_skills([pick_skill("die"), place_skill("mat")], Plans(), _FakeExecutor(), self._ik,
                                 state, MotionParams(), gate=gate, trial=trial)
        except (Refused, ExecutionFailed):
            motions = None
        return motions, state

    @staticmethod
    def _fake_place(gate, monkeypatch, refuse_below=None):
        """Replace the place tool; return the (held_place_z, posed objects) of each call."""
        calls = []

        def place(self, ctx):
            calls.append((ctx.held_place_z, gate.env.poses[-1][1] if gate.env.poses else None))
            if refuse_below is not None and ctx.held_place_z < refuse_below:
                raise Refused("search_exhausted", {"reason": "no transport to the standoff"})
            return ctx.gate.ungated([np.asarray(ctx.qpos)])

        monkeypatch.setattr(ManipulationTools, "plan_place", place)
        return calls

    @staticmethod
    def _plans(*xs):
        plans = [_FakePlan() for _ in xs]
        for i, (p, x) in enumerate(zip(plans, xs)):
            p.metadata = {"candidate_index": i}
            p.pre_grasp_pos = np.array([x, 0.0, 0.2])
        return plans

    @staticmethod
    def _ik(pos, quat, finger, init_qpos=None):
        return np.r_[np.full(7, pos[0]), 0.0, 0.0]

    def test_pick_reaches_the_first_waypoint_along_the_planned_connection(self, gate):
        motions, trace, starts = self._pick_only(gate, self._plans(0.2), self._ik)
        q_pre = np.r_[np.full(7, 0.2), 0.0, 0.0]
        line = JointPlanner(gate.env, gate).connect(np.zeros(9), q_pre, target="die", label="approach").motion.trajectory
        pick = motions[0]
        assert len(line) > 1 and np.allclose(pick.trajectory[:len(line)], line) and np.allclose(starts[0], q_pre)
        assert pick.spans[0].label == "approach" and pick.spans[0].end == len(line)
        assert trace["trials"][0]["approach"]["ok"]

    def test_a_pick_candidate_whose_connection_is_refused_is_not_tried(self, gate, monkeypatch):
        monkeypatch.setattr(gate.env, "contacts_at",
                            lambda q, held=None, placed=None:
                            [(("robot", "hand"), ("wall", "base"))] if np.allclose(q[:7], 0.2) else [],
                            raising=False)
        tried = []
        _, trace, _ = self._pick_only(gate, self._plans(0.2, 0.3), self._ik,
                                      trial=lambda traj, target: tried.append(target) or
                                      {"lifted": True, "lift_m": 0.1, "seconds": 0.5})
        first, second = trace["trials"]
        assert not first["approach"]["ok"] and first["approach"]["reason"] == "invalid_goal" and "steps" not in first
        assert second["approach"]["ok"] and tried == ["die"] and trace["selected_candidate_index"] == 1

    def test_a_pick_candidate_the_gate_rejects_spends_no_connection(self, gate, monkeypatch):
        monkeypatch.setattr(gate.env, "contacts_at",
                            lambda q, held=None, placed=None:
                            [(("robot", "hand"), ("wall", "base"))] if np.allclose(q, 1.0) else [],
                            raising=False)
        connects = []
        connect = JointPlanner.connect
        monkeypatch.setattr(JointPlanner, "connect", lambda self, *a, **k: connects.append(a) or connect(self, *a, **k))
        motions, trace, _ = self._pick_only(gate, self._plans(0.2, 0.3), self._ik)
        assert motions is None and connects == []
        assert [t["gate"]["segment"] for t in trace["trials"]] == ["seg", "seg"]

    def test_place_carries_the_held_object_along_the_planned_connection(self, gate, monkeypatch):
        starts = []
        monkeypatch.setattr(_FakeExecutor, "place",
                            lambda self, actions, start: starts.append(np.array(start.q)) or
                            _chain([np.full(9, 2.0)], holding=True))
        motions, state = self._pick_place(gate, _FakeExecutor())
        q0, q_pre = np.full(9, 1.0), np.array([0.0] * 7 + [1.0, 1.0])
        line = JointPlanner(gate.env, gate).connect(q0, q_pre, target="die", label="transport").motion.trajectory
        place = motions[1]
        assert np.allclose(place.trajectory[:len(line)], line) and np.allclose(starts[0], q_pre)
        assert place.spans[0].label == "transport" and place.spans[0].holding
        assert state["skill_traces"][1]["transport"]["ok"] and state["skill_traces"][1]["transport"]["direct"]

    def test_a_pick_whose_every_place_is_refused_is_refused_with_the_place_records(self, gate, monkeypatch):
        monkeypatch.setattr(gate.env, "contacts_at",
                            lambda q, held=None, placed=None:
                            [(("die", "base"), ("barrier", "base"))] if held and np.allclose(q[:7], 0.0) else [],
                            raising=False)
        ex = _FakeExecutor()
        with pytest.raises(Refused) as exc:
            self._pick_place(gate, ex)
        assert exc.value.label == "search_exhausted" and ex.calls == ["pick", "pick"] and gate.env.writes == 0
        (trial,) = exc.value.evidence["trials"]
        place = trial["place"]
        assert not place["ok"] and place["label"] == "search_exhausted" and "steps" not in trial
        assert not place["transport"]["ok"] and place["transport"]["queries"] >= 1 and place["transport"]["reason"]

    def test_a_pick_candidate_whose_place_is_refused_is_not_tried(self, gate, monkeypatch):
        calls = self._fake_place(gate, monkeypatch, refuse_below=0.13)
        plans = self._plans(0.2, 0.3)
        plans[1].grasp_pos = np.array([0.5, 0.0, 0.15])
        tried = []
        motions, state = self._pick_then_place(gate, plans, trial=lambda traj, target: tried.append(target) or
                                               {"lifted": True, "lift_m": 0.1, "seconds": 0.5})
        first, second = state["skill_traces"][0]["trials"]
        assert first["place"] == {"ok": False, "label": "search_exhausted", "reason": "no transport to the standoff"}
        assert "steps" not in first and second["place"] == {"ok": True} and tried == ["die"]
        assert state["skill_traces"][0]["selected_candidate_index"] == 1 and len(motions) == 2
        assert [z for z, _ in calls] == pytest.approx([0.12, 0.15, 0.15])
        assert [set(posed or ()) for _, posed in calls] == [{"die"}, {"die"}, set()]

    def test_each_candidate_attaches_its_own_grasp_and_place_starts_from_the_measured_hand(self, gate, monkeypatch):
        grasps, starts = [], []
        monkeypatch.setattr(_FakeExecutor, "pick", lambda self, actions, start:
                            grasps.extend(a.holding for a in actions if isinstance(a, Attach))
                            or _chain([np.full(9, 1.0)]))
        monkeypatch.setattr(_FakeExecutor, "place", lambda self, actions, start:
                            starts.append(start) or _chain([np.full(9, 2.0)], holding=True))
        plans = self._plans(0.2, 0.3)
        plans[1].grasp_pos = np.array([0.5, 0.0, 0.15])
        lifted = iter([False, True])
        motions, _ = self._pick_then_place(gate, plans, trial=lambda traj, target:
                                           {"lifted": next(lifted), "lift_m": 0.1, "seconds": 0.5})
        # Each candidate's list runs twice, from the pre-grasp and whole, with the same grasp.
        assert len(motions) == 2 and len(grasps) == 4 and len(starts) == 3
        assert grasps[0] is grasps[1] and grasps[2] is grasps[3]
        for plan, grasp in zip(plans, grasps[::2]):
            np.testing.assert_allclose(pose_matrix(plan.grasp_pos, plan.grasp_quat) @ grasp.t_hand_object,
                                       pose_matrix(*gate.env.object_pose("die")))
        # ClearEnv's hand and object both sit at the origin, so a measured binding is the identity.
        assert all(s.holding.target == "die" and np.allclose(s.holding.t_hand_object, np.eye(4)) for s in starts)

    def test_a_placeable_pick_is_unchanged_by_the_place_check(self, gate):
        alone, _, _ = self._pick_only(gate, self._plans(0.2), self._ik)
        motions, state = self._pick_then_place(gate, self._plans(0.2))
        assert np.array_equal(np.array(motions[0].trajectory), np.array(alone[0].trajectory))
        assert motions[0].spans == alone[0].spans
        assert [t["skill"] for t in state["skill_traces"]] == ["pick", "place"]
        assert state["skill_traces"][0]["trials"][0]["place"] == {"ok": True}
        gate.check_current(motions[0])

    def test_a_refused_place_check_leaves_no_pose_trace_or_write(self, gate, monkeypatch):
        monkeypatch.setattr(gate.env, "contacts_at",
                            lambda q, held=None, placed=None:
                            [(("die", "base"), ("barrier", "base"))] if held and np.allclose(q[:7], 0.0) else [],
                            raising=False)
        state = {**_scene_state(), "positions": {"die": np.array([0.5, 0.0, 0.0]), "mat": np.array([0.4, -0.2, 0.0])}}
        with pytest.raises(Refused):
            plan_skills([pick_skill("die"), place_skill("mat")], _FakePlanner(), _FakeExecutor(),
                        lambda *a, **k: np.zeros(9), state, MotionParams(), gate=gate)
        assert [t["skill"] for t in state["skill_traces"]] == ["pick"]
        assert gate.env.poses == () and gate.env.writes == 0

    @pytest.mark.parametrize("phase, carried", [("transport", True), ("descend", False), ("all", False)])
    def test_place_clears_the_carry_over_only_when_it_releases(self, gate, monkeypatch, phase, carried):
        monkeypatch.setattr(ManipulationTools, "plan_place", lambda self, ctx: ctx.gate.ungated([np.asarray(ctx.qpos)]))
        ctx = SkillContext(skill=place_skill("mat"), executor=None, motion_planner=None, params=MotionParams(),
                           scene_state=_scene_state(), qpos=np.zeros(9), category="die", gate=gate, held="die",
                           held_category="die", held_place_z=0.12, phase=phase)
        _, category, z = plan_skill(ctx, None, tools=ManipulationTools(gate, None, None, None), policy=RulePolicy(),
                                    trial=None, try_all=False)
        assert (category, z) == (("die", 0.12) if carried else (None, None))

    def test_tools_call_the_gate_and_trial_patched_after_construction(self, gate, monkeypatch):
        tools, seen = ManipulationTools(gate, None, None, None), []
        monkeypatch.setattr(gate, "verify", lambda **k: seen.append("verify") or "voucher")
        monkeypatch.setattr(gate.env, "trial_from_here", lambda traj, target: seen.append(target) or {}, raising=False)
        assert tools.verify({"trajectory": []}) == "voucher" and tools.trial([], "die") == {}
        assert seen == ["verify", "die"]

    @pytest.mark.parametrize("planner, tried", [(None, False), (GeometricGraspPlanner(), True)])
    def test_only_geometric_candidates_are_tried(self, planner, tried):
        env = SimpleNamespace(entity_map={}, planner=planner, solve_ik=lambda *a, **k: np.zeros(9))
        run = runner.EpisodeRun(env, None, executor=None, motion_params=None, dataset=None, record_fn=None,
                                target_frames={}, camera_names=(), env_state_names=(), grasp_selection="trial_all")
        assert (run.trial == run.tools.trial if tried else run.trial is None) and run.try_all is tried

    def test_no_place_check_before_a_place_on_a_dynamic_anchor(self, gate, monkeypatch):
        calls = self._fake_place(gate, monkeypatch)
        motions, state = self._pick_then_place(gate, self._plans(0.2),
                                               frames={"mat": SimpleNamespace(kind="articulated_opening")})
        assert len(motions) == 2 and "place" not in state["skill_traces"][0]["trials"][0]
        assert [posed for _, posed in calls] == [None]

    def test_place_lets_the_held_object_rest_on_the_target_entity_of_a_surface(self, gate, monkeypatch):
        monkeypatch.setattr(_FakeExecutor, "place",
                            lambda self, *a, **k: _chain([np.full(9, 0.3)], holding=True, allow="rest"))
        monkeypatch.setattr(gate.env, "contacts_at",
                            lambda q, held=None, placed=None:
                            [(("die", "base"), ("mat", "base"))] if np.allclose(q, 0.3) else [],
                            raising=False)
        motions = plan_skills([pick_skill("die"), place_skill("mat.top")], _FakePlanner(), _FakeExecutor(),
                             lambda *a, **k: np.zeros(9), {**_scene_state(), "positions": {
                                 "die": np.array([0.5, 0.0, 0.0]), "mat.top": np.array([0.4, -0.2, 0.0])}},
                             MotionParams(), gate=gate)
        assert len(motions) == 2

    @staticmethod
    def _file_skills(lift=None):
        """The public plan's steps as skills, placing on a plain ``mat``, the pick with ``lift`` when given."""
        steps = [{"place": {"target": "mat"}} if "place" in s else {**s, **({"lift": lift} if lift else {})}
                 for s in read_plan_file(PLANS / "f0_open.yaml")["steps"]]
        return steps_to_skills(steps, {"die": None, "mat": None})[0]

    def _waypoint_run(self, gate, skills):
        state = {**_scene_state(), "positions": {"die": np.array([0.5, 0.0, 0.0]), "mat": np.array([0.4, -0.2, 0.0])}}
        return plan_skills(skills, _FakePlanner(), _WaypointExecutor(), self._ik, state, MotionParams(), gate=gate)

    def test_the_default_lift_written_out_sends_the_same_trajectory_to_the_gate(self, gate):
        a = self._waypoint_run(gate, self._file_skills())
        b = self._waypoint_run(gate, self._file_skills({"direction": [0, 0, 1], "dist": 0.10}))
        assert len(a) == len(b) == 2
        for x, y in zip(a, b):
            assert np.array_equal(np.array(x.trajectory), np.array(y.trajectory)) and x.spans == y.spans

    def test_a_changed_lift_changes_the_trajectory_sent_to_the_gate(self, gate):
        pick_a = self._waypoint_run(gate, self._file_skills())[0]
        pick_b = self._waypoint_run(gate, self._file_skills({"direction": [1, 0, 1], "dist": 0.05}))[0]
        assert pick_a.trajectory[-1][:3] == pytest.approx([0.5, 0.0, 0.12 + 0.10])
        assert pick_b.trajectory[-1][:3] == pytest.approx([0.5 + 0.05 / np.sqrt(2), 0.0, 0.12 + 0.05 / np.sqrt(2)])

    def test_unknown_skill_raises(self, gate):
        with pytest.raises(ValueError):
            plan_skills([Skill("teleport", "die")], _FakePlanner(), _FakeExecutor(),
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
