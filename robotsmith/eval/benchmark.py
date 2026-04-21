"""RoboSmithBenchmark — vla-eval Benchmark implementation.

Wraps Genesis sim scenes driven by TaskSpec as a vla-eval benchmark.
Each episode: reset scene → render obs → receive action → step → repeat → predicate.

Usage with vla-eval:
    vla-eval run --config configs/eval/robotsmith_pick_cube.yaml --no-docker
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Any

import numpy as np

try:
    from vla_eval.benchmarks.base import Benchmark, StepResult
    from vla_eval.types import Action, EpisodeResult, Observation, Task
except ImportError:
    raise ImportError(
        "vla-eval not installed. Run: pip install vla-eval"
    )

from robotsmith.tasks.presets import TASK_PRESETS
from robotsmith.tasks.task_spec import TaskSpec
from robotsmith.tasks.predicates import evaluate_predicate


JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "joint5", "joint6", "joint7",
    "finger_joint1", "finger_joint2",
]
HOME_QPOS = np.array(
    [0, -0.3, 0, -2.2, 0, 2.0, 0.79, 0.04, 0.04], dtype=np.float32,
)
KP = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100], dtype=np.float32)
KV = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10], dtype=np.float32)
FORCE_LOWER = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100], dtype=np.float32)
FORCE_UPPER = np.array([87, 87, 87, 87, 12, 12, 12, 100, 100], dtype=np.float32)
CUBE_SIZE = (0.04, 0.04, 0.04)
MAX_DELTA_RAD = 0.15
SETTLE_STEPS = 100
FPS = 30


def _ensure_display():
    """Start Xvfb if DISPLAY is not set (headless Linux)."""
    if os.environ.get("DISPLAY"):
        return
    try:
        proc = subprocess.Popen(
            ["Xvfb", ":99", "-screen", "0", "1280x1024x24", "-ac",
             "+extension", "GLX"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        os.environ["DISPLAY"] = ":99"
        time.sleep(2)
        if proc.poll() is None:
            print(f"[display] Xvfb started (PID={proc.pid})")
    except FileNotFoundError:
        pass


def _to_numpy(t) -> np.ndarray:
    arr = t.cpu().numpy() if hasattr(t, "cpu") else np.array(t)
    return arr[0] if arr.ndim > 1 else arr


def _render_cam(cam) -> np.ndarray:
    rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
    arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.array(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.uint8)


def _smooth_action(prev: np.ndarray, target: np.ndarray, max_delta: float = MAX_DELTA_RAD) -> np.ndarray:
    delta = np.clip(target - prev, -max_delta, max_delta)
    return prev + delta


class RoboSmithBenchmark(Benchmark):
    """Genesis-based eval benchmark for vla-eval-harness."""

    def __init__(
        self,
        tasks: list[str] | None = None,
        seed: int = 42,
        fps: int = FPS,
        settle_steps: int = SETTLE_STEPS,
        x_range: tuple[float, float] = (0.4, 0.7),
        y_range: tuple[float, float] = (-0.2, 0.2),
        success_lift_m: float = 0.05,
        cpu: bool = False,
        **kwargs: Any,
    ):
        self._task_names = tasks or ["pick_cube"]
        self._seed = seed
        self._fps = fps
        self._settle_steps = settle_steps
        self._x_range = x_range
        self._y_range = y_range
        self._success_lift_m = success_lift_m
        self._cpu = cpu

        self._scene = None
        self._franka = None
        self._cube = None
        self._cam_up = None
        self._cam_side = None
        self._motors_dof = None
        self._task_spec: TaskSpec | None = None
        self._rng = np.random.default_rng(seed)

        self._current_qpos = HOME_QPOS.copy()
        self._step_count = 0
        self._episode_start_time = 0.0
        self._cube_initial_z = 0.0
        self._episode_success = False
        self._gs_initialized = False

    def _init_genesis(self) -> None:
        if self._gs_initialized:
            return

        _ensure_display()

        import genesis as gs
        import torch  # noqa: F401 — genesis needs torch in scope

        backend = gs.cpu if self._cpu else gs.gpu
        gs.init(backend=backend, logging_level="warning")

        cube_z = CUBE_SIZE[2] / 2.0
        self._cube_z = cube_z

        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1.0 / self._fps, substeps=4),
            rigid_options=gs.options.RigidOptions(
                enable_collision=True,
                enable_joint_limit=True,
                box_box_detection=True,
            ),
            show_viewer=False,
        )

        self._scene.add_entity(gs.morphs.Plane())

        self._cube = self._scene.add_entity(
            morph=gs.morphs.Box(size=CUBE_SIZE, pos=(0.55, 0.0, cube_z)),
            material=gs.materials.Rigid(friction=4.0),
            surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
        )

        self._franka = self._scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        )

        self._cam_up = self._scene.add_camera(
            res=(640, 480), pos=(0.55, 0.55, 0.55),
            lookat=(0.55, 0.0, 0.10), fov=45, GUI=False,
        )
        self._cam_side = self._scene.add_camera(
            res=(640, 480), pos=(0.55, -0.55, cube_z + 0.25),
            lookat=(0.55, 0.0, cube_z + 0.10), fov=50, GUI=False,
        )

        self._scene.build()

        self._motors_dof = [
            self._franka.get_joint(name).dofs_idx_local[0]
            for name in JOINT_NAMES
        ]
        self._franka.set_dofs_kp(KP, self._motors_dof)
        self._franka.set_dofs_kv(KV, self._motors_dof)
        self._franka.set_dofs_force_range(FORCE_LOWER, FORCE_UPPER, self._motors_dof)

        self._gs_initialized = True

    # ---- vla-eval Benchmark ABC ----

    def get_tasks(self) -> list[Task]:
        return [{"name": name} for name in self._task_names]

    async def start_episode(self, task: Task) -> None:
        import torch

        self._init_genesis()

        task_name = task["name"]
        if task_name not in TASK_PRESETS:
            raise ValueError(
                f"Unknown task '{task_name}'. Available: {list(TASK_PRESETS.keys())}"
            )
        self._task_spec = TASK_PRESETS[task_name]

        cx = self._rng.uniform(*self._x_range)
        cy = self._rng.uniform(*self._y_range)
        self._cube_xy = (cx, cy)

        import genesis as gs

        self._franka.set_dofs_position(HOME_QPOS, self._motors_dof)
        self._franka.control_dofs_position(HOME_QPOS, self._motors_dof)
        self._franka.zero_all_dofs_velocity()

        self._cube.set_pos(
            torch.tensor([cx, cy, self._cube_z], dtype=torch.float32,
                         device=gs.device).unsqueeze(0)
        )
        self._cube.set_quat(
            torch.tensor([1, 0, 0, 0], dtype=torch.float32,
                         device=gs.device).unsqueeze(0)
        )
        self._cube.zero_all_dofs_velocity()

        for _ in range(self._settle_steps):
            self._scene.step()

        self._current_qpos = HOME_QPOS.copy()
        self._step_count = 0
        self._episode_start_time = time.monotonic()
        self._cube_initial_z = self._cube_z
        self._episode_success = False

    async def apply_action(self, action: Action) -> None:
        action_arr = action.get("actions", action.get("action"))
        if action_arr is None:
            raise ValueError("Action must contain 'actions' or 'action' key")

        action_np = np.asarray(action_arr, dtype=np.float32).flatten()

        n_dofs = len(JOINT_NAMES)
        if action_np.shape[0] > n_dofs:
            action_np = action_np[:n_dofs]
        elif action_np.shape[0] < n_dofs:
            padded = self._current_qpos.copy()
            padded[:action_np.shape[0]] = action_np
            action_np = padded

        target = _smooth_action(self._current_qpos, action_np)
        self._current_qpos = target.copy()

        self._franka.control_dofs_position(target, self._motors_dof)
        self._scene.step()
        self._step_count += 1

    async def get_observation(self) -> Observation:
        state = _to_numpy(self._franka.get_dofs_position(self._motors_dof)).astype(np.float32)
        img_up = _render_cam(self._cam_up)
        img_side = _render_cam(self._cam_side)

        return {
            "images": {
                "up": img_up,
                "side": img_side,
            },
            "state": state,
            "task_description": self._task_spec.instruction if self._task_spec else "",
        }

    async def is_done(self) -> bool:
        if self._task_spec is None:
            return True

        cube_pos = _to_numpy(self._cube.get_pos())
        env_state = {
            "object_positions": {"cube": cube_pos.copy()},
            "initial_positions": {"cube": np.array([
                self._cube_xy[0], self._cube_xy[1], self._cube_initial_z,
            ])},
        }

        try:
            success = evaluate_predicate(
                self._task_spec.success_fn, env_state, self._task_spec.success_params,
            )
        except Exception:
            success = False

        if success:
            self._episode_success = True
            return True

        if self._step_count >= self._task_spec.episode_length:
            return True

        return False

    async def get_time(self) -> float:
        return time.monotonic() - self._episode_start_time

    async def get_result(self) -> EpisodeResult:
        return {
            "success": self._episode_success,
            "steps": self._step_count,
            "task": self._task_spec.name if self._task_spec else "",
        }

    def get_metric_keys(self) -> dict[str, str]:
        return {"success": "mean"}

    def get_metadata(self) -> dict[str, Any]:
        return {
            "benchmark": "robotsmith",
            "tasks": self._task_names,
            "seed": self._seed,
            "fps": self._fps,
            "action_space": "joint_position_9d",
        }

    def cleanup(self) -> None:
        self._scene = None
        self._gs_initialized = False
