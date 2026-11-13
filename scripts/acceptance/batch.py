"""A batch of acceptance instances that survives a hung simulator: rows are written as they finish, and a worker
process that runs past an attempt's timeout is killed from outside and resumed from the rows so far."""

import json
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

INSTANCE_TIMEOUT_S = 600


def run_batch(seeds, one, fh, prior=(), failed=None, show=None, running: Path | None = None) -> list[dict]:
    """``one(seed, attempt)`` for each seed, its row appended to ``fh`` as it finishes. An instance that raises is
    retried once; both attempts stay in ``fh`` and the last one is the instance's reading. The attempt number is
    the one the row carries, so anything an attempt writes can name it. ``prior`` are the rows of an
    earlier run of the batch: a seed they settle is not run again. ``running`` names the attempt in progress for
    ``supervise``: written before it starts, removed once its row is. ``failed`` is the verdict an error row
    carries; ``show`` the keys printed per row (all when None)."""
    failed = failed or {}

    def write(row):
        fh.write(json.dumps(row, default=str) + "\n")
        fh.flush()
        print(json.dumps({k: row.get(k) for k in show} if show else row, default=str), flush=True)

    rows = []
    for seed in seeds:
        past = [r for r in prior if r["seed"] == seed]
        attempt = len(past)
        row = past[-1] if past else None
        while attempt < 2 and (row is None or row.get("outcome") == "error"):
            attempt += 1
            if running:
                tmp = running.with_suffix(".tmp")
                tmp.write_text(json.dumps({"seed": seed, "attempt": attempt, "started": time.time()}))
                _while_read(os.replace, tmp, running)
            t0 = time.perf_counter()
            try:
                row = one(seed, attempt)
            except Exception:  # noqa: BLE001 -- an instance's crash is a reading, the batch goes on
                row = {"outcome": "error", **failed, "why": traceback.format_exc()}
            row = {"seed": seed, "attempt": attempt, **row, "seconds": round(time.perf_counter() - t0, 1)}
            write(row)
            if running:
                _while_read(running.unlink, missing_ok=True)
        rows.append(row)
    return rows


def _while_read(change, *args, **kwargs) -> None:
    # Windows refuses to replace or delete a file another process has open: here, supervise reading the mark.
    for _ in range(99):
        try:
            return change(*args, **kwargs)
        except PermissionError:
            time.sleep(0.01)
    change(*args, **kwargs)


def supervise(argv, log: Path, timeout: float = INSTANCE_TIMEOUT_S, failed=None, poll: float = 1.0) -> int:
    """Run the worker ``argv`` (a ``run_batch`` over ``log`` with ``running=log.with_suffix(".running")``) in its own
    process group and return its exit code. An attempt still running after ``timeout`` seconds gets its process
    group killed, an error row in ``log`` unless the worker wrote one first, and the worker is started again with
    ``--resume``. A step stuck in the simulator cannot be interrupted from inside the process, so this one can't
    be a thread. One supervisor per log."""
    running = log.with_suffix(".running")
    resume = []
    while True:
        running.unlink(missing_ok=True)
        if os.name == "nt":
            proc = subprocess.Popen([*argv, *resume], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            proc = subprocess.Popen([*argv, *resume], start_new_session=True)
        mark = None
        while proc.poll() is None:
            time.sleep(poll)
            try:
                mark = json.loads(running.read_text())
            except FileNotFoundError:
                mark = None
            except PermissionError:  # Windows: the worker is replacing the mark right now.
                continue
            if mark and time.time() - mark["started"] > timeout:
                if os.name == "nt":
                    proc.kill()
                else:
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                break
        else:
            return proc.returncode
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
        if not any(r.get("seed") == mark["seed"] and r.get("attempt") == mark["attempt"] for r in rows):
            with log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"seed": mark["seed"], "attempt": mark["attempt"], "outcome": "error",
                                     **(failed or {}), "why": f"killed after {timeout} s", "seconds": timeout})
                         + "\n")
        resume = ["--resume"]
