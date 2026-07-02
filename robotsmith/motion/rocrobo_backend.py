"""rocRobo motion backend — collision-free IK / plan_motion over a warm serve.

`RocRoboBackend` talks to a warm ``python -m rocrobo.serve --serve`` process over
JSONL (the torch process never imports jax) and implements the ``MotionPlanner``
protocol; on any serve failure it degrades to a Genesis fallback so data
generation never blocks. The ``world`` / ``attach`` payloads it sends are built
by the authoring layer in ``rocrobo_world.py`` and passed into ``solve_ik`` /
``plan_motion`` by the caller — this module does not construct them.

Deployment: robotsmith (torch) and rocRobo (jax) co-reside on the GPU host; the
client lazily starts a single long-lived
``docker exec -i <container> python -m rocrobo.serve --serve`` and holds its
stdin/stdout (process-level singleton). ``world`` primitive shapes are kept
stable to avoid serve-side recompilation.
"""

from __future__ import annotations

import json
import logging
import os
import select
import subprocess
import threading
from typing import Sequence

import numpy as np

from robotsmith.motion.constants import N_ARM_JOINTS as _N_ARM_JOINTS
from robotsmith.motion.planner import GenesisBackend, PlanResult, Waypoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serve client (warm process, JSONL over docker exec)
# ---------------------------------------------------------------------------


class RocRoboServeClient:
    """Long-lived JSONL client to a warm ``rocrobo.serve --serve`` process."""

    def __init__(
        self,
        container: str | None = None,
        *,
        serve_argv: Sequence[str] | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self._container = container or os.environ.get(
            "ROCROBO_CONTAINER", "rocrobo_dev"
        )
        self._serve_argv = list(serve_argv) if serve_argv else None
        # Per-request wall-clock budget. The first request after a cold JAX
        # compilation cache can legitimately take minutes (XLA compile, GPU idle
        # / CPU-bound); once the on-disk cache is warm every shape is a fast cache
        # load, so this is a *stuck-serve* guard, not a latency target. Generous by
        # default; override with ROCROBO_REQUEST_TIMEOUT_S.
        if timeout_s is None:
            timeout_s = float(os.environ.get("ROCROBO_REQUEST_TIMEOUT_S", "300"))
        self._timeout_s = timeout_s
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.available = True  # set False after a hard failure

    def _argv(self) -> list[str]:
        if self._serve_argv is not None:
            return self._serve_argv
        # rocRobo is mounted at /rocrobo; the package lives under
        # /rocrobo/rocRobo/core and resolves its sphere assets via ROCROBO_ASSETS
        # (NOT CWD). Keep CWD at /rocrobo top level — do NOT cd into
        # /rocrobo/pyroki/examples, which holds a stray ``rocrobo.py`` shim that
        # shadows the package under ``python -m`` and breaks ``rocrobo.serve``.
        workdir = os.environ.get("ROCROBO_WORKDIR", "/rocrobo")
        pythonpath = os.environ.get("ROCROBO_PYTHONPATH", "/rocrobo/rocRobo/core")
        assets = os.environ.get("ROCROBO_ASSETS", "/rocrobo/pyroki/examples/assets")
        # Persist XLA compiled executables to disk so the multi-minute first-call
        # JIT is paid ONCE *ever* (not once per fresh serve / per run): the warm
        # serve only amortises within a single process, but each scenario run spawns
        # a new serve, so without this every run recompiles from scratch. The cache
        # dir lives on the /rocrobo bind mount, so it survives container restarts.
        # MIN_*=0 caches even quick/small compiles (avoids partial warm caches).
        cache_dir = os.environ.get("ROCROBO_JAX_CACHE", f"{workdir}/.jax_cache")
        return [
            "docker",
            "exec",
            "-i",
            "-w",
            workdir,
            "-e",
            f"PYTHONPATH={pythonpath}",
            "-e",
            f"ROCROBO_ASSETS={assets}",
            "-e",
            f"JAX_COMPILATION_CACHE_DIR={cache_dir}",
            "-e",
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0",
            "-e",
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0",
            self._container,
            "python",
            "-u",
            "-m",
            "rocrobo.serve",
            "--serve",
        ]

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        argv = self._argv()
        logger.info("[rocrobo] starting warm serve: %s", " ".join(argv))
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            self._proc = None

    def request(self, payload: dict) -> dict:
        """Send one JSON request line, return the parsed JSON response line.

        Bounded by ``self._timeout_s``: a serve that does not answer within the
        budget is treated as stuck — we tear it down and mark the client
        unavailable so the caller degrades to the Genesis fallback instead of
        blocking the whole rollout forever (the readline below is an unbounded
        blocking read otherwise). The serve is single-request/response over JSONL,
        so we cannot safely reuse a process whose in-flight response we abandoned;
        ``available=False`` keeps the rest of the run on the fallback path.
        """
        if not self.available:
            raise RuntimeError("rocrobo serve marked unavailable")
        with self._lock:
            self._ensure_started()
            assert self._proc is not None and self._proc.stdin and self._proc.stdout
            try:
                self._proc.stdin.write(json.dumps(payload) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self.available = False
                self._kill()
                raise RuntimeError(f"rocrobo serve io error: {exc}") from exc
            ready, _, _ = select.select(
                [self._proc.stdout], [], [], self._timeout_s
            )
            if not ready:
                self.available = False
                self._kill()
                raise RuntimeError(
                    f"rocrobo serve timeout after {self._timeout_s:.0f}s "
                    f"(op={payload.get('op')})"
                )
            try:
                line = self._proc.stdout.readline()
            except (BrokenPipeError, OSError) as exc:
                self.available = False
                self._kill()
                raise RuntimeError(f"rocrobo serve io error: {exc}") from exc
            if not line:
                self.available = False
                self._kill()
                raise RuntimeError("rocrobo serve closed (empty response)")
            return json.loads(line)


_CLIENT_SINGLETON: RocRoboServeClient | None = None


def get_serve_client(**kwargs) -> RocRoboServeClient:
    """Process-level singleton serve client (one warm process per run)."""
    global _CLIENT_SINGLETON
    if _CLIENT_SINGLETON is None:
        _CLIENT_SINGLETON = RocRoboServeClient(**kwargs)
    return _CLIENT_SINGLETON


# ---------------------------------------------------------------------------
# Backend (MotionPlanner)
# ---------------------------------------------------------------------------


def _resample(
    traj: np.ndarray, src_dt: float | None, dst_dt: float
) -> list[np.ndarray]:
    """Re-time a joint trajectory onto the simulator's control rate.

    The planner emits waypoints spaced ``src_dt`` seconds apart; the sim steps
    every ``dst_dt`` seconds. These rarely match, so we linearly interpolate the
    path onto a fresh time grid at ``dst_dt`` (e.g. a 0.5 s path at src_dt=0.1
    → 6 points becomes dst_dt=0.02 → 26 points). If ``src_dt`` is unknown we
    cannot align by time, so the waypoints are returned as-is (one per step).

    Args:
        traj: joint trajectory, shape ``(n_waypoints, n_joints)``.
        src_dt: seconds between planner waypoints (rocRobo retiming dt); None/0
            means "no timing info" → return points unchanged.
        dst_dt: simulator control step in seconds (the rate to resample onto).
    """
    traj = np.asarray(traj, dtype=np.float64)
    n = len(traj)
    if n <= 1 or not src_dt or src_dt <= 0:
        return [row.copy() for row in traj]
    total_t = (n - 1) * src_dt
    n_out = max(int(round(total_t / dst_dt)) + 1, 2)
    src_times = np.arange(n) * src_dt
    dst_times = np.linspace(0.0, total_t, n_out)
    out = np.empty((n_out, traj.shape[1]), dtype=np.float64)
    for j in range(traj.shape[1]):
        out[:, j] = np.interp(dst_times, src_times, traj[:, j])
    return [row.copy() for row in out]


class RocRoboBackend:
    """``MotionPlanner`` backed by a warm rocRobo serve, with Genesis fallback.

    rocRobo returns 7 arm DOF; this backend pads to a 9-dim qpos (7 arm + 2
    finger) using the requested finger width. On any serve failure it falls back
    to the supplied Genesis ``solve_ik`` (collision-blind) so runs never block.
    """

    collision_aware = True

    def __init__(
        self,
        solve_ik_fallback,
        *,
        client: RocRoboServeClient | None = None,
        n_finger: int = 2,
        base_z: float = 0.0,
    ) -> None:
        self._fallback = GenesisBackend(solve_ik_fallback)
        self._client = client if client is not None else get_serve_client()
        self._n_finger = n_finger
        # rocRobo plans with the base at z=0 but the Franka is table-mounted at
        # world z=base_z; shift every pose/world primitive by -base_z into the base
        # frame before sending, else the table-height ground makes plans infeasible.
        self._base_z = float(base_z)

    def _to_base_pos(self, pos) -> list[float]:
        p = [float(v) for v in np.asarray(pos, dtype=np.float64)]
        if len(p) >= 3:
            p[2] -= self._base_z
        return p

    def _to_base_world(self, world) -> list[dict]:
        if not self._base_z:
            return list(world)
        shifted: list[dict] = []
        for prim in world:
            p = dict(prim)
            if "center" in p:  # box
                c = list(p["center"])
                c[2] = float(c[2]) - self._base_z
                p["center"] = c
            if "point" in p:  # halfspace
                pt = list(p["point"])
                pt[2] = float(pt[2]) - self._base_z
                p["point"] = pt
            shifted.append(p)
        return shifted

    def _pad(self, q_arm, finger_pos: float) -> np.ndarray:
        q = np.zeros(_N_ARM_JOINTS + self._n_finger, dtype=np.float64)
        q[:_N_ARM_JOINTS] = np.asarray(q_arm, dtype=np.float64)[:_N_ARM_JOINTS]
        q[_N_ARM_JOINTS:] = float(finger_pos)
        return q

    def solve_ik(
        self, pos, quat, finger_pos, *, init_qpos=None, world=None, attach=None
    ):
        if world is None or not self._client.available:
            return self._fallback.solve_ik(pos, quat, finger_pos, init_qpos=init_qpos)
        payload = {
            "op": "solve_ik",
            "pos": self._to_base_pos(pos),
            "quat": [float(v) for v in np.asarray(quat, dtype=np.float64)],
            "world": self._to_base_world(world),
        }
        # ``attach`` (spheres + T_ee_obj) is EE/object-relative, so it needs no
        # base-frame shift like world/pos do; pass it through verbatim.
        if attach:
            payload["attach"] = attach
        if init_qpos is not None:
            payload["seed"] = [
                float(v)
                for v in np.asarray(init_qpos, dtype=np.float64)[:_N_ARM_JOINTS]
            ]
        try:
            resp = self._client.request(payload)
        except (RuntimeError, ValueError) as exc:
            logger.warning("[rocrobo] solve_ik fallback to Genesis: %s", exc)
            return self._fallback.solve_ik(pos, quat, finger_pos, init_qpos=init_qpos)
        if not resp.get("ok", False) or "q" not in resp:
            logger.warning(
                "[rocrobo] solve_ik not ok (%s), fallback", resp.get("reason")
            )
            return self._fallback.solve_ik(pos, quat, finger_pos, init_qpos=init_qpos)
        return self._pad(resp["q"], finger_pos)

    def plan_motion(
        self, q_start, waypoints, *, world=None, control_dt=None, attach=None
    ):
        wps: list[Waypoint] = list(waypoints)
        if world is None or not self._client.available or not wps:
            return PlanResult(success=False, reason="no-world-or-serve")
        payload = {
            "op": "plan_motion",
            "q_start": [
                float(v)
                for v in np.asarray(q_start, dtype=np.float64)[:_N_ARM_JOINTS]
            ],
            "waypoints": [
                {
                    "pos": self._to_base_pos(w.pos),
                    "quat": [float(v) for v in np.asarray(w.quat, dtype=np.float64)],
                }
                for w in wps
            ],
            "world": self._to_base_world(world),
        }
        # ``attach`` (spheres + T_ee_obj) is EE/object-relative — no base shift.
        if attach:
            payload["attach"] = attach
        try:
            resp = self._client.request(payload)
        except (RuntimeError, ValueError) as exc:
            logger.warning("[rocrobo] plan_motion failed: %s", exc)
            return PlanResult(success=False, reason=str(exc))
        if not resp.get("ok", False) or "traj" not in resp:
            reason = resp.get("reason") or resp.get("error") or "no-traj"
            # Debug-level: a miss is an expected, recoverable outcome here (e.g. the
            # adaptive hover sweep probes heights until one clears). The caller that
            # actually degrades to a collision-blind fallback logs loudly. Metrics
            # map an aggregate reason to a sub-check (end_pos_mm / world_min_m).
            logger.debug(
                "[rocrobo] plan_motion miss (%s) metrics=%s",
                reason,
                resp.get("metrics"),
            )
            return PlanResult(success=False, reason=str(reason))
        # Prefer the retimed trajectory (``timed.q`` over ``total_s``); fall back
        # to the raw geometric path when retiming was not produced.
        timed = resp.get("timed")
        if timed and timed.get("q"):
            traj_arm = np.asarray(timed["q"], dtype=np.float64)
            total_s = float(timed.get("total_s") or 0.0)
            src_dt = (
                total_s / (len(traj_arm) - 1)
                if len(traj_arm) > 1 and total_s > 0
                else None
            )
        else:
            traj_arm = np.asarray(resp["traj"], dtype=np.float64)
            src_dt = None
        if traj_arm.ndim != 2 or traj_arm.shape[1] < _N_ARM_JOINTS:
            return PlanResult(success=False, reason="bad-traj-shape")
        dst_dt = control_dt or 1.0 / 30.0
        arm_path = _resample(traj_arm[:, :_N_ARM_JOINTS], src_dt, dst_dt)
        # Finger schedule: rocRobo plans the arm only; hold the last waypoint's
        # finger width across the segment (per-segment routing keeps it constant).
        finger = float(wps[-1].finger_width)
        trajectory = [self._pad(q, finger) for q in arm_path]
        return PlanResult(
            success=True,
            trajectory=trajectory,
            quality=resp.get("metrics", {}),
        )
