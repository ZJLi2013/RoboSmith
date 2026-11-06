"""The execution gate: every point of a trajectory is checked in the simulator before any of it runs.

The gate holds the legality rules (``check_point``, ``permitted``); ``SimEnv`` answers what touches what at a
configuration (``contacts_at``, ``hand_pose``, ``object_pose``), the robot's ``joint_limits`` and
``joined_links``, and fingerprints the state the answer holds for.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field

import numpy as np

from robotsmith.grasp.transforms import pose_matrix

FINGER_LINKS = ("left_finger", "right_finger")
# Bodies outside ``SimEnv.entity_map`` (table, ground) are reported as this entity.
SUPPORT = "env"
FK_POS_TOL_M, FK_ROT_TOL_RAD = 0.005, 0.05
LIMIT_TOL_RAD = 1e-4
REFUSALS = ("search_exhausted", "unsupported", "infeasible")


@dataclass(frozen=True)
class Span:
    """``trajectory[start:end]`` and what may touch there: with ``holding`` the target rides in the hand;
    ``allow="grip"`` lets the fingers touch the target, ``allow="rest"`` lets the held target touch a rest surface."""

    label: str
    start: int
    end: int
    holding: bool = False
    allow: str | None = None


def _frozen(a) -> np.ndarray:
    a = np.array(a, dtype=np.float64)
    a.setflags(write=False)
    return a


@dataclass(frozen=True, eq=False)
class Holding:
    """The object in the hand, rigidly at hand · ``t_hand_object`` (4x4)."""

    target: str
    t_hand_object: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "t_hand_object", _frozen(self.t_hand_object))


def spans_of(segments: list[dict], offset: int = 0) -> list[Span]:
    """Spans of ``execute_motion_chain`` segments, shifted by ``offset`` points."""
    return [Span(s["label"], s["span"][0] + offset, s["span"][1] + offset, s["holding"], s["allow"])
            for s in segments]


def poses_of(segments: list[dict], offset: int = 0) -> dict[int, tuple]:
    """``{point: (pos, quat_wxyz)}`` of the hand pose each ``execute_motion_chain`` segment's waypoint asks for,
    at the point that reaches it, shifted by ``offset`` points."""
    return {s["reach"] + offset: s["pose"] for s in segments if s.get("reach") is not None}


class MotionRejected(RuntimeError):
    def __init__(self, span: Span, index: int, rejection: dict):
        self.span, self.index, self.rejection = span, index, rejection
        self.pair = rejection.get("pair")
        super().__init__(f"{span.label} point {index}: {rejection}")

    def to_dict(self) -> dict:
        out = {k: v for k, v in self.rejection.items() if k != "pair"}
        if self.pair is not None:
            out["pair"] = [list(p) for p in self.pair]
        return {"segment": self.span.label, "index": self.index, **out}


class StaleMotion(RuntimeError):
    """The scene changed since the motion was verified, or the gate did not issue it."""


class Refused(RuntimeError):
    """A step ends with zero physics steps: ``search_exhausted`` (no candidate in the declared set and budget
    passed the gate; not a proof of insolubility), ``unsupported``, or ``infeasible`` (kept for a checkable proof
    over the action domain; nothing produces it yet)."""

    def __init__(self, label: str, evidence: dict):
        if label not in REFUSALS:
            raise ValueError(f"refusal label {label!r} is not one of {REFUSALS}")
        super().__init__(f"{label}: {evidence.get('reason', '')}")
        self.label, self.evidence = label, evidence


class ExecutionFailed(RuntimeError):
    """A step stepped physics (its candidates' trials) and did not succeed; ``steps`` counts them."""

    label = "execution_failed"

    def __init__(self, evidence: dict, steps: int):
        super().__init__(f"execution_failed after {steps} physics steps: {evidence.get('reason', '')}")
        self.evidence, self.steps = evidence, steps


@dataclass(frozen=True, eq=False)
class VerifiedMotion:
    """A gate's voucher for a trajectory: read-only, and executable only while the state it was checked in holds."""

    trajectory: tuple
    spans: tuple
    fingerprint: str
    checked: int
    gated: bool = True
    holding: Holding | None = field(default=None)


def permitted(pair, span: Span, target: str, rest: set) -> bool:
    (ea, la), (eb, lb) = pair
    if ea == eb == "robot":
        return True
    if "robot" in (ea, eb):
        link, other = (la, eb) if ea == "robot" else (lb, ea)
        return other == target and link in FINGER_LINKS and (span.holding or span.allow == "grip")
    if span.holding and target in (ea, eb):
        return span.allow == "rest" and (eb if ea == target else ea) in rest
    return True


def certified(pair, span: Span, target: str, rest: set) -> bool:
    """``permitted``, and a target that is not held touches only rest surfaces and the robot. For contacts read
    from executed physics, where a free target meeting an obstacle means something pushed it."""
    if not permitted(pair, span, target, rest):
        return False
    (ea, _), (eb, _) = pair
    if span.holding or target not in (ea, eb) or ea == eb:
        return True
    return (eb if ea == target else ea) in {*rest, "robot"}


def _pose_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    cos = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), float(np.arccos(np.clip(cos, -1.0, 1.0)))


class ExecutionGate:
    def __init__(self, env):
        self.env = env
        self._issued = weakref.WeakSet()

    def check_point(self, q, span: Span, *, target: str, rest=(), held=None, placed=None, pose=None) -> dict | None:
        """Why ``q`` may not run under ``span``'s rules, or ``None``: finite values within the joint limits, the hand
        within 5 mm / 0.05 rad of ``pose`` when the point asks for one, no contact between robot links that are not
        joined, and every other contact ``permitted``. ``held`` / ``placed`` pose the target as ``contacts_at``
        takes them. Reads only; the scene is as it was on return or raise."""
        q = np.asarray(q, dtype=np.float64)
        lo, hi = self.env.joint_limits
        if q.shape != lo.shape or not np.all(np.isfinite(q)):
            return {"reason": "invalid_value"}
        out = np.flatnonzero((q < lo - LIMIT_TOL_RAD) | (q > hi + LIMIT_TOL_RAD))
        if out.size:
            return {"reason": "joint_limit", "joints": out.tolist()}
        if pose is not None:
            dp, dr = _pose_error(pose_matrix(*self.env.hand_pose(q)), pose_matrix(*pose))
            if dp > FK_POS_TOL_M or dr > FK_ROT_TOL_RAD:
                return {"reason": "fk_residual", "position_m": round(dp, 5), "angle_rad": round(dr, 5)}
        rest = {SUPPORT, *rest}
        joined = self.env.joined_links
        for pair in self.env.contacts_at(q, held=held, placed=placed):
            (ea, la), (eb, lb) = pair
            if ea == eb == "robot":
                if la != lb and frozenset((la, lb)) not in joined:
                    return {"reason": "self_collision", "pair": pair}
            elif not permitted(pair, span, target, rest):
                return {"reason": "collision", "pair": pair}
        return None

    def verify(self, trajectory, spans, *, target: str, q_start, rest=(), holding: Holding | None = None,
               poses=None) -> VerifiedMotion:
        """``check_point`` every point of ``trajectory`` (which starts from ``q_start``) under its span; raise
        ``MotionRejected`` at the first that fails. ``poses`` maps the points that must reach a hand pose to it.

        Holding spans need ``holding``: where it attaches (the point before the first holding point, or
        ``q_start``) hand · T must sit within 5 mm / 0.05 rad of the target's pose now, else ``binding_mismatch``.
        While holding, the target rides at hand · T; after the last holding point it stays where the hand left it."""
        traj = [np.asarray(q, dtype=np.float64) for q in trajectory]
        spans = tuple(spans)
        poses = poses or {}
        bounds = [0] + [s.end for s in spans]
        if [s.start for s in spans] != bounds[:-1] or bounds[-1] != len(traj):
            raise ValueError(f"spans {[(s.start, s.end) for s in spans]} do not tile {len(traj)} points")
        held_points = [i for s in spans if s.holding for i in range(s.start, s.end)]
        released = None
        if held_points:
            if holding is None or holding.target != target:
                raise ValueError(f"holding spans need a Holding of {target!r}, got {holding}")
            first = held_points[0]
            anchor = traj[first - 1] if first > 0 else np.asarray(q_start, dtype=np.float64)
            dp, dr = _pose_error(pose_matrix(*self.env.hand_pose(anchor)) @ holding.t_hand_object,
                                 pose_matrix(*self.env.object_pose(target)))
            if dp > FK_POS_TOL_M or dr > FK_ROT_TOL_RAD:
                span = next(s for s in spans if s.start <= first < s.end)
                raise MotionRejected(span, first, {"reason": "binding_mismatch", "position_m": round(dp, 5),
                                                   "angle_rad": round(dr, 5)})
            if held_points[-1] < len(traj) - 1:
                released = (target, pose_matrix(*self.env.hand_pose(traj[held_points[-1]])) @ holding.t_hand_object)
        for span in spans:
            held = (target, holding.t_hand_object) if span.holding else None
            placed = released if released and not span.holding and span.start > held_points[-1] else None
            for i in range(span.start, span.end):
                why = self.check_point(traj[i], span, target=target, rest=rest, held=held, placed=placed,
                                       pose=poses.get(i))
                if why is not None:
                    raise MotionRejected(span, i, why)
        return self._issue(traj, spans, len(traj), holding=holding if held_points else None)

    def ungated(self, trajectory) -> VerifiedMotion:
        """A motion outside the gate's scope (articulated drags), stamped with the state it was planned in."""
        traj = [np.asarray(q, dtype=np.float64) for q in trajectory]
        return self._issue(traj, (Span("ungated", 0, len(traj)),), 0, gated=False)

    def _issue(self, traj, spans, checked, *, gated=True, holding=None) -> VerifiedMotion:
        motion = VerifiedMotion(tuple(_frozen(q) for q in traj), tuple(spans), self.env.state_fingerprint(), checked,
                                gated, holding)
        self._issued.add(motion)
        return motion

    def check_current(self, motion: VerifiedMotion) -> None:
        if motion not in self._issued:
            raise StaleMotion("this gate did not issue the motion")
        if self.env.state_fingerprint() != motion.fingerprint:
            raise StaleMotion("the scene changed since this motion was verified")
