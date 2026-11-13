"""RoboSmithEmbodiment's declarations as Inspect Robots reads them; no simulator is built."""

from pathlib import Path

import pytest

pytest.importorskip("inspect_robots")

from inspect_robots import ToolAdapter  # noqa: E402
from inspect_robots.conformance import assert_embodiment_conformant  # noqa: E402

from robotsmith.eval.embodiment import RoboSmithEmbodiment  # noqa: E402

F0 = Path(__file__).resolve().parents[1] / "scenarios" / "acceptance" / "f0_open.yaml"


@pytest.fixture
def embodiment(tmp_path):
    return RoboSmithEmbodiment(F0, seed=0, attempt=1, log_path=tmp_path / "judge.jsonl")


def test_the_embodiment_is_conformant(embodiment):
    assert_embodiment_conformant(embodiment.info)


def test_the_agent_gets_move_to_on_the_fingertip_and_no_judge_tools(embodiment):
    adapter = ToolAdapter()
    adapter.bind(embodiment.info)
    assert "move_to" in adapter.tools.names
    assert set(adapter.tools.names) <= {"move_to", "sleep", "done", "give_up"}
    assert not embodiment.info.observation_space.cameras
    assert not hasattr(embodiment, "contribute_tools")


def test_objects_are_declared_as_privileged_state(embodiment):
    keys = {f.key for f in embodiment.info.observation_space.state.fields}
    assert keys == {"eef_state", "object_pos.die", "object_pos.mat"}
    assert "privileged" in embodiment.info.docs
