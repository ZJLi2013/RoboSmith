"""Joint-space connections through the execution gate: a lazy RRT-Connect whose every state and every edge point
is a ``check_point``, and whose path is issued by ``verify``. Nothing here steps physics."""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass

import numpy as np

from robotsmith.execution.gate import (
    ExecutionGate,
    Holding,
    Refused,
    Span,
    VerifiedMotion,
)

MAX_QUERIES = 1500
MAX_SECONDS = 30.0
STEP_M = 0.005
EXTEND_RAD = 0.1
ARM = slice(0, 7)


def subdivide(a, b, lever: np.ndarray) -> list[np.ndarray]:
    """The points after ``a`` up to ``b`` such that no point of the robot or held object moves more than ``STEP_M``
    between two of them: ``lever`` is ``SimEnv.lever_arms``."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = max(1, math.ceil(float(lever @ np.abs(b - a)) / STEP_M))
    return [a + (b - a) * (i / n) for i in range(1, n + 1)]


def _bisection(n: int) -> list[int]:
    """1 .. n-1, midpoints first, so a collision partway along an edge is met early."""
    order, spans = [], [(0, n)]
    while spans:
        nxt = []
        for lo, hi in spans:
            if hi - lo > 1:
                mid = (lo + hi) // 2
                order.append(mid)
                nxt += [(lo, mid), (mid, hi)]
        spans = nxt
    return order


@dataclass(frozen=True)
class Connection:
    motion: VerifiedMotion
    queries: int
    seconds: float
    nodes: int
    direct: bool


class _Budget(Exception):
    pass


class _Tree:
    def __init__(self, root: np.ndarray):
        self.nodes, self.parent, self.alive, self.checked = [root], [-1], [True], [True]

    def add(self, q: np.ndarray, parent: int) -> int:
        self.nodes.append(q)
        self.parent.append(parent)
        self.alive.append(True)
        self.checked.append(False)
        return len(self.nodes) - 1

    def nearest(self, q: np.ndarray) -> int:
        live = [i for i, a in enumerate(self.alive) if a]
        d = np.linalg.norm(np.asarray([self.nodes[i][ARM] for i in live]) - q[ARM], axis=1)
        return live[int(np.argmin(d))]

    def chain(self, i: int) -> list[int]:
        out = []
        while i >= 0:
            out.append(i)
            i = self.parent[i]
        return out

    def cut(self, i: int) -> None:
        self.alive[i] = False
        for j in range(i + 1, len(self.nodes)):
            if not self.alive[self.parent[j]]:
                self.alive[j] = False


class JointPlanner:
    """Connects two configurations with the fingers held where they start. ``env`` gives ``joint_limits`` and
    ``lever_arms``; whether a point may run is only ``gate.check_point``'s answer, and the connection is
    ``gate.verify``'s voucher. A query budget of ``max_queries`` or ``max_seconds`` ends the search, whichever
    comes first; a refusal is ``Refused("search_exhausted")`` with the reason and zero physics steps."""

    def __init__(self, env, gate: ExecutionGate, *, seed: int = 0, max_queries: int = MAX_QUERIES,
                 max_seconds: float = MAX_SECONDS, extend_rad: float = EXTEND_RAD):
        self.env, self.gate, self.seed = env, gate, seed
        self.max_queries, self.max_seconds, self.extend_rad = max_queries, max_seconds, extend_rad

    def connect(self, q_start, q_goal, *, target: str, rest=(), holding: Holding | None = None,
                label: str = "move_to") -> Connection:
        a, b = np.asarray(q_start, dtype=np.float64), np.asarray(q_goal, dtype=np.float64)
        if not np.array_equal(a[ARM.stop:], b[ARM.stop:]):
            raise ValueError("a connection keeps the fingers where they start")
        t0, queries = time.perf_counter(), 0
        span = Span(label, 0, 1, holding=holding is not None)
        held = (target, holding.t_hand_object) if holding is not None else None
        lever = self.env.lever_arms(held)

        def legal(q) -> bool:
            nonlocal queries, why
            if queries >= self.max_queries or time.perf_counter() - t0 >= self.max_seconds:
                raise _Budget
            queries += 1
            why = self.gate.check_point(q, span, target=target, rest=rest, held=held)
            return why is None

        def clear(p, q) -> bool:
            points = subdivide(p, q, lever)
            return all(legal(points[i - 1]) for i in _bisection(len(points)))

        def refuse(reason, **evidence):
            raise Refused("search_exhausted", {"reason": reason, "queries": queries,
                                               "seconds": round(time.perf_counter() - t0, 3), **evidence})

        why, nodes, direct = None, 2, False
        try:
            if not legal(a):
                refuse("invalid_start", rejection=why)
            if not legal(b):
                refuse("invalid_goal", rejection=why)
            if clear(a, b):
                waypoints, direct = [a, b], True
            else:
                waypoints, nodes = self._search(a, b, legal, clear)
        except _Budget:
            refuse("budget")
        path = [p for w0, w1 in itertools.pairwise(waypoints) for p in subdivide(w0, w1, lever)]
        motion = self.gate.verify(path, [Span(label, 0, len(path), holding=span.holding)], target=target,
                                  q_start=a, rest=rest, holding=holding)
        return Connection(motion, queries, round(time.perf_counter() - t0, 3), nodes, direct)

    def _search(self, a, b, legal, clear) -> tuple[list[np.ndarray], int]:
        """Grow a tree from each end, checking only new nodes; once they meet, check the path's unchecked edges
        and cut the subtree under the first that fails."""
        rng = np.random.default_rng(self.seed)
        lo, hi = (lim[ARM] for lim in self.env.joint_limits)
        trees = [_Tree(a), _Tree(b)]
        start = trees[0]

        def step(tree, toward) -> int | None:
            near = tree.nearest(toward)
            d = toward - tree.nodes[near]
            dist = float(np.linalg.norm(d[ARM]))
            q = toward.copy() if dist <= self.extend_rad else tree.nodes[near] + d * (self.extend_rad / dist)
            return tree.add(q, near) if legal(q) else None

        while True:
            grow, other = trees
            sample = a.copy()
            sample[ARM] = rng.uniform(lo, hi)
            new = step(grow, sample)
            if new is not None:
                last = None
                while (k := step(other, grow.nodes[new])) is not None:
                    last = k
                    if np.allclose(other.nodes[k], grow.nodes[new]):
                        break
                else:
                    last = None
                if last is not None:
                    path = self._checked(start, grow, new, other, last, clear)
                    if path is not None:
                        return path, sum(sum(t.alive) for t in trees)
            trees.reverse()

    @staticmethod
    def _checked(start, grow, new, other, last, clear) -> list[np.ndarray] | None:
        """The trees' path from start to goal, checked from each node to the farthest later one it clears; an edge
        of the trees themselves that fails cuts the subtree below it."""
        goal = other if grow is start else grow
        head, tail = start.chain(new if grow is start else last), goal.chain(last if grow is start else new)
        nodes = [start.nodes[i] for i in reversed(head)] + [goal.nodes[i] for i in tail[1:]]
        # The tree edge into each next node: (tree, child index).
        edges = [(start, i) for i in reversed(head[:-1])] + [(goal, i) for i in tail[:-1]]
        out, i = [nodes[0]], 0
        while i < len(nodes) - 1:
            j = next((j for j in range(len(nodes) - 1, i + 1, -1) if clear(nodes[i], nodes[j])), i + 1)
            if j == i + 1:
                tree, child = edges[i]
                if not tree.checked[child]:
                    if not clear(nodes[i], nodes[j]):
                        tree.cut(child)
                        return None
                    tree.checked[child] = True
            out.append(nodes[j])
            i = j
        return out
