"""``PlanState``: the predicted joints and what the hand holds; only ``attach`` / ``detach`` change the holding.
In an action list they are the ``Attach`` / ``Detach`` items, which move nothing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotsmith.execution.gate import Holding
from robotsmith.grasp.transforms import pose_matrix


@dataclass(frozen=True)
class PlanState:
    q: np.ndarray
    holding: Holding | None = None


@dataclass(frozen=True)
class Attach:
    holding: Holding


@dataclass(frozen=True)
class Detach:
    pass


def binding(target: str, hand_pose: tuple, object_pose: tuple) -> Holding:
    """``target`` held at ``hand⁻¹ · object``. A prediction passes the candidate's grasp pose as ``hand_pose``;
    after execution it is the measured hand pose."""
    return Holding(target, np.linalg.inv(pose_matrix(*hand_pose)) @ pose_matrix(*object_pose))


def attach(state: PlanState, held: Holding) -> PlanState:
    if state.holding is not None:
        raise ValueError(f"attach {held.target!r} while holding {state.holding.target!r}")
    return PlanState(state.q, held)


def detach(state: PlanState) -> PlanState:
    if state.holding is None:
        raise ValueError("detach with nothing held")
    return PlanState(state.q)
