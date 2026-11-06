"""A batch of acceptance instances that survives a hung simulator: rows are written as they finish, and a rerun
with the rows so far goes on from the instance that hung."""

import json
import os
import threading
import time
import traceback

INSTANCE_TIMEOUT_S = 600
HUNG = 3


def run_batch(seeds, one, fh, prior=(), timeout=None, failed=None, show=None,
              on_hang=lambda: os._exit(HUNG)) -> list[dict]:
    """``one(seed)`` for each seed, its row appended to ``fh`` as it finishes. An instance that raises is retried
    once; both attempts stay in ``fh`` and the last one is the instance's reading. ``prior`` are the rows of an
    earlier run of the batch: a seed they settle is not run again. An attempt still running after ``timeout``
    seconds is written as an error and ``on_hang`` ends the process, since a hung simulator step cannot be
    interrupted from Python; rerunning with the rows so far retries it. ``failed`` is the verdict an error row
    carries; ``show`` the keys printed per row (all when None)."""
    lock = threading.Lock()
    failed = failed or {}

    def write(row):
        with lock:
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
        print(json.dumps({k: row.get(k) for k in show} if show else row, default=str), flush=True)

    def hang(seed, attempt):
        write({"seed": seed, "attempt": attempt, "outcome": "error", **failed,
               "why": f"no result after {timeout} s", "seconds": timeout})
        on_hang()

    rows = []
    for seed in seeds:
        past = [r for r in prior if r["seed"] == seed]
        attempt = len(past)
        row = past[-1] if past else None
        while attempt < 2 and (row is None or row.get("outcome") == "error"):
            attempt += 1
            watchdog = threading.Timer(timeout, hang, (seed, attempt)) if timeout else None
            if watchdog:
                watchdog.daemon = True
                watchdog.start()
            t0 = time.perf_counter()
            try:
                row = one(seed)
            except Exception:  # noqa: BLE001 -- an instance's crash is a reading, the batch goes on
                row = {"outcome": "error", **failed, "why": traceback.format_exc()}
            if watchdog:
                watchdog.cancel()
            row = {"seed": seed, "attempt": attempt, **row, "seconds": round(time.perf_counter() - t0, 1)}
            write(row)
        rows.append(row)
    return rows
