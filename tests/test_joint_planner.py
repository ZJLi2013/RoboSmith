"""JointPlanner: joint-space connections whose every state and edge point is the gate's check_point."""

import itertools

import numpy as np
import pytest

from robotsmith.execution.gate import ExecutionGate, Holding, MotionRejected, Refused
from robotsmith.grasp.transforms import pose_matrix
from robotsmith.motion.joint_planner import STEP_M, JointPlanner

DOWN = np.array([0.0, 1.0, 0.0, 0.0])
HAND_WALL = (("robot", "hand"), ("wall", "base"))
ELBOW_WALL = (("robot", "link4"), ("wall", "base"))
DIE_WALL = (("die", "base"), ("wall", "base"))
LEVER = 0.5


def q(**joints):
    out = np.zeros(9)
    out[7:] = 0.02
    for k, v in joints.items():
        out[int(k[1:])] = v
    return out


def band(j, lo, hi, *, unless=None):
    """Joint ``j`` inside (lo, hi), except where joint ``unless[0]`` is above ``unless[1]``."""
    return lambda x: lo < x[j] < hi and (unless is None or x[unless[0]] <= unless[1])


class _Arm:
    """Joints 0-6 within ±2 rad. The hand sits at (0.5 + q0, q1, 0.9) pointing down and does not move with the other
    joints. ``hits`` are (region, pair, held_only): the pair is reported at any configuration in the region."""

    def __init__(self, hits=()):
        self.hits, self.queries, self.writes = list(hits), [], 0
        self.joint_limits = (np.r_[np.full(7, -2.0), 0.0, 0.0], np.r_[np.full(7, 2.0), 0.04, 0.04])
        self.joined_links = frozenset()
        self.die = None

    def lever_arms(self, held=None):
        return np.r_[np.full(7, LEVER + (0.2 if held else 0.0)), 1.0, 1.0]

    def hand_pose(self, x):
        return np.array([0.5 + x[0], x[1], 0.9]), DOWN

    def object_pose(self, name):
        return self.die

    def contacts_at(self, x, held=None, placed=None):
        self.queries.append((np.array(x), held))
        return [pair for region, pair, held_only in self.hits if (held is not None or not held_only) and region(x)]

    def state_fingerprint(self):
        return str(self.writes)

    def step(self):
        raise AssertionError("a planner stepped physics")


def _planner(arm, max_queries=10_000, **kw):
    # These geometries need 0.6k-4k queries over seeds; MAX_QUERIES is for the scene, not for them.
    gate = ExecutionGate(arm)
    return JointPlanner(arm, gate, max_queries=max_queries, **kw), gate


def _hold(arm, start, offset=(0.0, 0.0, 0.1)):
    t = np.eye(4)
    t[:3, 3] = offset
    arm.die = tuple(_split(pose_matrix(*arm.hand_pose(start)) @ t))
    return Holding("die", t)


def _split(m):
    from robotsmith.grasp.transforms import rotmat_to_quat_wxyz
    return m[:3, 3], rotmat_to_quat_wxyz(m[:3, :3])


def test_a_free_request_connects_directly():
    arm = _Arm()
    planner, gate = _planner(arm)
    c = planner.connect(q(), q(j0=1.0), target="die")
    assert c.direct and c.nodes == 2
    np.testing.assert_allclose(c.motion.trajectory[-1], q(j0=1.0))
    # Both ends, the points between them, then verify's own pass over the path.
    assert c.queries == 2 + len(c.motion.trajectory) - 1
    assert len(arm.queries) == c.queries + len(c.motion.trajectory)
    gate.check_current(c.motion)


def test_an_obstacle_is_gone_around_and_every_point_passes_the_gate():
    wall = band(0, 0.4, 0.6, unless=(1, 0.3))
    arm = _Arm([(wall, HAND_WALL, False)])
    planner, gate = _planner(arm)
    c = planner.connect(q(), q(j0=1.0), target="die")
    assert not c.direct and c.queries <= 1500
    assert not any(wall(x) for x in c.motion.trajectory)
    np.testing.assert_allclose(c.motion.trajectory[-1], q(j0=1.0))
    gate.check_current(c.motion)


def test_an_invalid_start_or_goal_is_refused_after_one_query():
    wall = band(0, 0.4, 0.6)
    planner, _ = _planner(_Arm([(wall, HAND_WALL, False)]))
    with pytest.raises(Refused) as exc:
        planner.connect(q(j0=0.5), q(j0=1.0), target="die")
    assert exc.value.label == "search_exhausted"
    assert exc.value.evidence["reason"] == "invalid_start" and exc.value.evidence["queries"] == 1
    assert exc.value.evidence["rejection"] == {"reason": "collision", "pair": HAND_WALL}
    with pytest.raises(Refused) as exc:
        planner.connect(q(), q(j0=0.5), target="die")
    assert exc.value.evidence["reason"] == "invalid_goal" and exc.value.evidence["queries"] == 2


def test_a_walled_off_goal_exhausts_the_budget():
    arm = _Arm([(band(0, 0.4, 0.6), HAND_WALL, False)])
    planner, _ = _planner(arm, max_queries=300)
    with pytest.raises(Refused) as exc:
        planner.connect(q(), q(j0=1.0), target="die")
    assert exc.value.evidence["reason"] == "budget" and exc.value.evidence["queries"] == 300 == len(arm.queries)


def test_one_long_edge_can_exhaust_the_budget():
    arm = _Arm()
    planner, _ = _planner(arm, max_queries=100)
    with pytest.raises(Refused) as exc:
        planner.connect(q(j0=-2.0), q(j0=2.0), target="die")
    assert exc.value.evidence["reason"] == "budget" and len(arm.queries) == 100


def test_a_held_object_is_checked_on_every_point():
    # The hand alone clears the wall at q1 > 0.3; the die below it needs q1 > 0.6.
    hand_wall, die_wall = band(0, 0.4, 0.6, unless=(1, 0.3)), band(0, 0.4, 0.6, unless=(1, 0.6))
    arm = _Arm([(hand_wall, HAND_WALL, False), (die_wall, DIE_WALL, True)])
    holding = _hold(arm, q())
    planner, gate = _planner(arm)
    c = planner.connect(q(), q(j0=1.0), target="die", holding=holding)
    assert all(held is not None and held[0] == "die" for _, held in arm.queries)
    assert not any(die_wall(x) for x in c.motion.trajectory)
    assert c.motion.holding is holding and all(s.holding for s in c.motion.spans)
    gate.check_current(c.motion)


def test_a_hold_that_does_not_match_the_scene_is_not_issued():
    arm = _Arm()
    holding = _hold(arm, q())
    arm.die = (arm.die[0] + [0.02, 0.0, 0.0], arm.die[1])
    planner, _ = _planner(arm)
    with pytest.raises(MotionRejected) as exc:
        planner.connect(q(), q(j0=1.0), target="die", holding=holding)
    assert exc.value.rejection["reason"] == "binding_mismatch"


def test_an_elbow_only_contact_is_found_while_the_hand_stays_still():
    elbow = band(2, 0.4, 0.6, unless=(3, 0.3))
    arm = _Arm([(elbow, ELBOW_WALL, False)])
    planner, _ = _planner(arm)
    start, goal = q(), q(j2=1.0)
    assert np.allclose(arm.hand_pose(start)[0], arm.hand_pose(goal)[0])
    c = planner.connect(start, goal, target="die")
    assert not c.direct and not any(elbow(x) for x in c.motion.trajectory)


def test_a_held_offset_swept_into_an_obstacle_is_found():
    swept = band(6, 0.7, 0.8, unless=(5, 0.3))
    arm = _Arm([(swept, DIE_WALL, True)])
    holding = _hold(arm, q())
    planner, _ = _planner(arm)
    c = planner.connect(q(), q(j6=1.5), target="die", holding=holding)
    assert not c.direct and not any(swept(x) for x in c.motion.trajectory)
    free = JointPlanner(arm, ExecutionGate(arm)).connect(q(), q(j6=1.5), target="die")
    assert free.direct


def test_edges_move_no_point_more_than_5_mm_per_step():
    arm = _Arm([(band(0, 0.4, 0.6, unless=(1, 0.3)), HAND_WALL, False)])
    planner, _ = _planner(arm)
    c = planner.connect(q(), q(j0=1.0), target="die")
    points = [q(), *c.motion.trajectory]
    lever = arm.lever_arms()
    assert max(float(lever @ np.abs(b - a)) for a, b in itertools.pairwise(points)) <= STEP_M + 1e-12


def test_the_same_seed_gives_the_same_path_and_no_state_changes():
    wall = band(0, 0.4, 0.6, unless=(1, 0.3))
    runs = []
    for _ in range(2):
        arm = _Arm([(wall, HAND_WALL, False)])
        before = arm.state_fingerprint()
        c = _planner(arm, seed=3)[0].connect(q(), q(j0=1.0), target="die")
        assert arm.state_fingerprint() == before
        runs.append((c.queries, np.asarray(c.motion.trajectory)))
    assert runs[0][0] == runs[1][0]
    np.testing.assert_array_equal(runs[0][1], runs[1][1])
