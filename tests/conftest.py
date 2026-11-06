import numpy as np
import pytest

from robotsmith.execution.gate import ExecutionGate


class ClearEnv:
    """A scene where nothing touches anything; ``writes`` stands in for the real-write count."""

    writes = 0
    joint_limits = (np.full(9, -10.0), np.full(9, 10.0))
    joined_links = frozenset()

    def contacts_at(self, qpos, held=None, placed=None):
        return []

    def lever_arms(self, held=None):
        return np.r_[np.full(7, 0.01), 1.0, 1.0]

    def hand_pose(self, qpos):
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    def object_pose(self, name):
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    def state_fingerprint(self):
        return str(self.writes)


@pytest.fixture
def gate():
    return ExecutionGate(ClearEnv())
