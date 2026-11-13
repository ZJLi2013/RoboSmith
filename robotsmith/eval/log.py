"""The judge log of one attempt: JSON lines, ``header`` first, then ``physics`` and ``control`` lines, ``final``
last. Every line is flushed as it is written, so an attempt killed mid-run leaves a readable prefix without a
``final``. An attempt owns its file: opening a path that exists is an error, never an append."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


class JudgeLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("x", encoding="utf-8")

    def _write(self, kind: str, **fields) -> None:
        self._fh.write(json.dumps({"kind": kind, **fields}, default=_plain) + "\n")
        self._fh.flush()

    def header(self, *, seed: int, attempt: int, scenario: str, scenario_sha256: str, success: dict,
               target: str, rest: list[str]) -> None:
        self._write("header", seed=seed, attempt=attempt, scenario=scenario, scenario_sha256=scenario_sha256,
                    success=success, target=target, rest=sorted(rest))

    def physics(self, k: int, contacts, held: bool) -> None:
        self._write("physics", k=k, contacts=[[list(a), list(b)] for a, b in contacts], held=held)

    def control(self, step: int, action, world: dict) -> None:
        self._write("control", step=step, action=action, world=world)

    def final(self, *, ended: str, success: bool, state: dict) -> None:
        self._write("final", ended=ended, success=success, state=state)

    def close(self) -> None:
        self._fh.close()


def read(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
