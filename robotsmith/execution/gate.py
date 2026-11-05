"""The execution gate: every point of a trajectory is checked in the simulator before any of it runs.

The gate holds the contact rules; ``SimEnv`` answers what touches what at a configuration
(``contacts_at``, ``hand_pose``, ``object_pose``) and fingerprints the state the answer holds for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotsmith.grasp.transforms import pose_matrix

FINGER_LINKS = ("left_finger", "right_finger")
# Bodies outside ``SimEnv.entity_map`` (table, ground) are reported as this entity.
SUPPORT = "env"


@dataclass(frozen=True)
class Span:
    """``trajectory[start:end]`` and what may touch there: with ``holding`` the target rides in the hand;
    ``allow="grip"`` lets the fingers touch the target, ``allow="rest"`` lets the held target touch a rest surface."""

    label: str
    start: int
    end: int
    holding: bool = False
    allow: str | None = None


def spans_of(segments: list[dict], offset: int = 0) -> list[Span]:
    """Spans of ``execute_motion_chain`` segments, shifted by ``offset`` points."""
    return [Span(s["label"], s["span"][0] + offset, s["span"][1] + offset, s["holding"], s["allow"])
            for s in segments]


class MotionRejected(RuntimeError):
    def __init__(self, span: Span, index: int, pair):
        (ea, la), (eb, lb) = pair
        super().__init__(f"{span.label} point {index}: {ea}/{la} touches {eb}/{lb}")
        self.span, self.index, self.pair = span, index, pair

    def to_dict(self) -> dict:
        return {"segment": self.span.label, "index": self.index, "pair": [list(p) for p in self.pair]}


class StaleMotion(RuntimeError):
    """The scene changed since the motion was verified."""


class Infeasible(RuntimeError):
    """No candidate of a step passed the gate (and its trial); the step runs zero frames."""

    def __init__(self, reason: str, detail: dict):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


@dataclass(frozen=True)
class VerifiedMotion:
    trajectory: list
    spans: tuple
    fingerprint: str
    checked: int
    gated: bool = True


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


class ExecutionGate:
    def __init__(self, env):
        self.env = env

    def verify(self, trajectory, spans, *, target: str, q_start, rest=()) -> VerifiedMotion:
        """Check every point of ``trajectory`` (which starts from ``q_start``) against its span's rules; raise
        ``MotionRejected`` at the first forbidden contact.

        While holding, the target sits at hand · T, T taken from the hand at the configuration holding starts
        from (the point before the first holding point, or ``q_start``) and the target's pose now. After
        the last holding point it stays where the hand left it."""
        traj = [np.asarray(q, dtype=np.float64) for q in trajectory]
        spans = tuple(spans)
        bounds = [0] + [s.end for s in spans]
        if [s.start for s in spans] != bounds[:-1] or bounds[-1] != len(traj):
            raise ValueError(f"spans {[(s.start, s.end) for s in spans]} do not tile {len(traj)} points")
        holding = [i for s in spans if s.holding for i in range(s.start, s.end)]
        t_hand_obj = None
        if holding:
            anchor = traj[holding[0] - 1] if holding[0] > 0 else np.asarray(q_start, dtype=np.float64)
            t_hand_obj = (np.linalg.inv(pose_matrix(*self.env.hand_pose(anchor)))
                          @ pose_matrix(*self.env.object_pose(target)))
        released = None
        if holding and holding[-1] < len(traj) - 1:
            released = (target, pose_matrix(*self.env.hand_pose(traj[holding[-1]])) @ t_hand_obj)
        rest = {SUPPORT, *rest}
        for span in spans:
            held = (target, t_hand_obj) if span.holding else None
            placed = released if released and not span.holding and span.start > holding[-1] else None
            for i in range(span.start, span.end):
                for pair in self.env.contacts_at(traj[i], held=held, placed=placed):
                    if not permitted(pair, span, target, rest):
                        raise MotionRejected(span, i, pair)
        return VerifiedMotion(traj, spans, self.env.state_fingerprint(), len(traj))

    def ungated(self, trajectory) -> VerifiedMotion:
        """A motion outside the gate's scope (articulated drags), stamped with the state it was planned in."""
        traj = [np.asarray(q, dtype=np.float64) for q in trajectory]
        return VerifiedMotion(traj, (Span("ungated", 0, len(traj)),), self.env.state_fingerprint(), 0, gated=False)

    def check_current(self, motion: VerifiedMotion) -> None:
        if self.env.state_fingerprint() != motion.fingerprint:
            raise StaleMotion("the scene changed since this motion was verified")
