import numpy as np
import pytest

from robotsmith.execution.gate import ExecutionGate


class ClearEnv:
    """A scene where nothing touches anything; ``writes`` stands in for the real-write count."""

    writes = 0

    def contacts_at(self, qpos, held=None, placed=None):
        return []

    def hand_pose(self, qpos):
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    def object_pose(self, name):
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    def state_fingerprint(self):
        return str(self.writes)


@pytest.fixture
def gate():
    return ExecutionGate(ClearEnv())
