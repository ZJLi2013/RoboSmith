"""Generic scenario runtime: a segment-level closed-loop episode driver.

One unified loop drives every scenario: reset+settle, then for each subtask
re-read the world's live state, plan only that segment, step it and record its
frames into the SAME episode. Anchoring each subtask to live object/joint poses
is what lets e.g. ``close`` grasp the handle where the drawer actually is after
``place`` may have nudged it. A rigid pick/place scene is the degenerate case
where the re-read changes nothing.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from robotsmith.grasp.geometric import GeometricGraspPlanner
from robotsmith.motion.chain import MotionInfeasible
from robotsmith.scenario_runtime import handoff
from robotsmith.sim.franka import HOME_QPOS, to_numpy_f32
from robotsmith.sim.world_state import collect_world_state, settle_window
from robotsmith.skills import SKILL_DEFS, plan_segment, resolve_frame, skill_phases
from robotsmith.tasks import evaluate_success
from robotsmith.tasks.task_spec import TaskSpec

logger = logging.getLogger(__name__)


def _maybe_motion_planner(env):
    """Construct the rocRobo collision-free backend when enabled (else None).

    Gated by ``ROCROBO_BACKEND``. The backend degrades to Genesis IK when the
    warm serve is unreachable.
    """
    if not os.environ.get("ROCROBO_BACKEND"):
        return None
    from robotsmith.motion.rocrobo_backend import RocRoboBackend

    logger.info("[rocrobo] collision-free backend enabled (ROCROBO_BACKEND)")
    # Franka base is mounted on the table at world z=table_surface_z; rocRobo plans
    # with its base at z=0, so the backend rebases all poses/world geometry.
    return RocRoboBackend(env.solve_ik, base_z=float(getattr(env, "table_surface_z", 0.0)))


@dataclass
class ScenarioEpisodeResult:
    """Outcome of one driven episode (the frames are already in the dataset)."""

    initial_positions: dict[str, np.ndarray]
    skill_traces: list[dict]
    frame_count: int
    handoffs: list[dict]
    final_qpos: np.ndarray
    # The settle the last step recorded and judged, still the world's current state; None when the
    # episode ended on a step that did not settle.
    final_window: dict | None = None


def run_scenario_episode(
    env,
    task_spec: TaskSpec,
    *,
    executor,
    motion_params,
    dataset,
    record_fn: Callable,
    settle_steps: int = 30,
    target_frames: dict | None = None,
    camera_names: tuple[str, ...] = ("up", "wrist"),
    env_state_names: tuple[str, ...] | list[str] = (),
) -> ScenarioEpisodeResult:
    """Drive one episode as a segment-level closed loop into ``dataset``.

    ``record_fn`` steps the sim and appends a segment's frames to the episode
    (the real one is ``gen.recorder.record_episode``; injected so this driver
    stays free of the Genesis/torch stack and unit-testable). The caller
    evaluates success and calls ``dataset.save_episode()`` once after this
    returns, so a task records as a single continuous episode/video.
    """
    reset_scene(env, settle_steps)
    run = EpisodeRun(env, task_spec, executor=executor, motion_params=motion_params, dataset=dataset,
                     record_fn=record_fn, target_frames=target_frames or {}, camera_names=camera_names,
                     env_state_names=env_state_names)
    run.run_steps(list(task_spec.skills), list(task_spec.step_bindings))
    return run.result()


def reset_scene(env, settle_steps: int) -> None:
    # Reset free bodies to their resolved world pose (full xyz, not just XY) and
    # settle. The resolver is the single source of truth for the spawn z (shelf /
    # in-drawer starts keep their height); the sim must not recompute it from the
    # table surface. Articulated assets stay anchored at their built pose, and
    # fixtures (fixed-base weldments like a shelf) are anchored at load time and
    # must not be re-set.
    articulated = set(getattr(env, "articulated_joint_dofs", {}) or {})
    obj_xyz_map = {
        name: (float(pos[0]), float(pos[1]), float(pos[2]))
        for name, pos in _resolved_entity_positions(env).items()
        if name in env.entity_map
        and name not in articulated
        and not _is_fixture(env, name)
    }
    env.reset(obj_xyz_map, settle_steps=settle_steps)


class EpisodeRun:
    """The live state of one episode after reset: ``run_steps`` executes skills from wherever the arm and the held
    object are, so a caller can stop at a failed step and continue with new steps in the same world."""

    def __init__(self, env, task_spec: TaskSpec, *, executor, motion_params, dataset, record_fn: Callable,
                 target_frames: dict, camera_names, env_state_names):
        self.env, self.task_spec, self.executor, self.motion_params = env, task_spec, executor, motion_params
        self.dataset, self.record_fn, self.target_frames = dataset, record_fn, target_frames
        self.camera_names, self.env_state_names = camera_names, env_state_names
        self.initial_positions = collect_object_positions(env)
        self.skill_traces: list[dict] = []
        self.handoffs: list[dict] = []
        self.skills: list = []
        self.qpos = HOME_QPOS.copy()
        self.held_category: str | None = None
        self.held_place_z: float | None = None
        self.held: str | None = None
        self.frame_count = 0
        self.window = None
        self.motion_planner = _maybe_motion_planner(env)

    def _record_step(self, q):
        self.record_fn(self.env, self.dataset, [q], self.task_spec, camera_names=self.camera_names,
                       env_state_names=self.env_state_names)

    def run_steps(self, skills: list, bindings: list[dict], *, stop_on_fail: bool = False) -> dict | None:
        """Execute ``skills`` with their ``bindings``; with ``stop_on_fail`` return the handoff of the first step
        judged false (the remaining steps are not run), else ``None``."""
        env = self.env
        planner_kw = {"motion_planner": self.motion_planner} if self.motion_planner is not None else {}
        for k, (skill, binding) in enumerate(zip(skills, list(bindings) + [None] * (len(skills) - len(bindings)))):
            i = len(self.skills)
            self.skills.append(skill)
            planned = self.skills + list(skills[k + 1:])
            skill_def = SKILL_DEFS.get(skill.name)
            # Per-phase closed loop: re-sense + re-anchor before each phase, not just
            # at the subtask boundary. ``skill_phases`` returns one phase for static
            # anchors (== legacy per-subtask loop) and splits dynamic-anchor places
            # into transport/descend so descend re-reads the recoiled drawer.
            phases = skill_phases(skill, self.target_frames.get(skill.target))
            step_start = None
            for phase in phases:
                scene_state = _live_scene_state(env, self.target_frames, self.skill_traces)
                if step_start is None:
                    step_start = {n: np.asarray(p).tolist() for n, p in collect_object_positions(env).items()}
                obj = skill.target if skill.name == "pick" else self.held
                try:
                    seg, self.held_category, self.held_place_z = plan_segment(
                        skill, i, planned, env.planner, self.executor, env.solve_ik,
                        scene_state, self.motion_params, self.qpos, self.held_category, self.held_place_z,
                        phase=phase, **planner_kw,
                    )
                except MotionInfeasible as exc:
                    entry = {"skill": skill.name, "phase": phase, "target": skill.target, "object": obj,
                             "done": False, "cause": "infeasible", "label": exc.label, "reason": exc.reason}
                    self.handoffs.append(entry)
                    self.window = None
                    if stop_on_fail:
                        return entry
                    break
                samples = self.record_fn(
                    env,
                    self.dataset,
                    seg,
                    self.task_spec,
                    primary_entity=env.entity_map.get(obj) if obj else None,
                    camera_names=self.camera_names,
                    env_state_names=self.env_state_names,
                )
                entry = handoff.check(
                    skill, phase, seg, samples or [], held=obj,
                    held_start=scene_state["positions"].get(obj) if obj else None,
                    target=scene_state["positions"].get(skill.target),
                    joints_before=scene_state["joint_positions"],
                    joints_after=env.get_joint_positions() if hasattr(env, "get_joint_positions") else {},
                    trace=self.skill_traces[-1] if self.skill_traces else None,
                    rocrobo=self.motion_planner is not None,
                )
                self.handoffs.append(entry)
                if skill.name == "pick":
                    self.held = skill.target
                elif skill.name == "place" and phase != "transport":
                    self.held = None
                self.frame_count += len(seg)
                if seg:
                    self.qpos = seg[-1]
                self.window = None
                if skill_def is not None and skill_def.post is not None:
                    if phase != phases[-1]:
                        entry["done"] = None
                    else:
                        b = binding or {"status": "unchecked", "reason": "no step binding"}
                        self.window = _judge_completion(env, entry, skill, skill_def, b, self.qpos,
                                                        self._record_step, step_start)
                        self.frame_count += entry["completion"].get("settle_frames", 0)
                        if stop_on_fail and entry["done"] is False:
                            return entry
        return None

    def result(self) -> ScenarioEpisodeResult:
        return ScenarioEpisodeResult(
            initial_positions=self.initial_positions,
            skill_traces=self.skill_traces,
            frame_count=self.frame_count,
            handoffs=self.handoffs,
            final_qpos=self.qpos,
            final_window=self.window,
        )


def _judge_completion(env, entry: dict, skill, skill_def, binding: dict, hold, record_step,
                      step_start: dict) -> dict | None:
    """Judge the step's bound completion after a recorded ``settle_s`` hold; return the window it judged
    (``None`` for a zero hold, which the task judge must not reuse).

    An unchecked binding records why and claims nothing. The judged state (one world-state read, the
    window, and the object positions read before the step was planned) is kept on the entry, so the step
    verdict re-judges offline like the task verdict.
    """
    node = skill_def.post(skill, binding)
    if node is None:
        entry.update(done=None, completion=dict(binding))
        return None
    frames = []

    def step(q):
        record_step(q)
        frames.append(q)

    window = settle_window(env, hold, settle_s=skill_def.settle_s, step=step)
    state = {**collect_world_state(env), "window": window, "step_start_positions": step_start}
    done = bool(evaluate_success(node, state))
    entry.update(done=done, completion={**binding, "settle_frames": len(frames), "judge_state": state})
    if not done and skill.name == "place":
        entry.setdefault("cause", "release?")
    return window if window["steps"] else None


def evaluate_scenario_task(
    env,
    task_spec: TaskSpec,
    *,
    initial_positions: dict[str, np.ndarray],
    reference_positions: dict[str, np.ndarray] | None = None,
    hold,
    settle_s: float = 0.5,
    window: dict | None = None,
) -> tuple[bool, dict]:
    """Settle for ``settle_s`` holding the arm at ``hold`` (the last commanded qpos), then judge the task.
    ``window`` is a settle the last step already recorded; it is reused instead of settling again.
    Returns the verdict and the exact state it judged, which re-judges offline to the same verdict."""
    if window is None:
        window = settle_window(env, hold, settle_s=settle_s)
    env_state = {
        **collect_world_state(env),
        "window": window,
        "initial_positions": {k: np.asarray(v).tolist() for k, v in initial_positions.items()},
        "target_positions": {k: np.asarray(v).tolist() for k, v in (reference_positions or {}).items()},
    }
    return evaluate_success(task_spec.success, env_state), env_state


def collect_object_positions(env) -> dict[str, np.ndarray]:
    return {
        name: to_numpy_f32(ent.get_pos()).copy()
        for name, ent in env.entity_map.items()
    }


# Frame kinds that resolve to a *fixed* world marker (no joint travel), so they
# are valid static success/diagnostic targets. Articulated / opening anchors move
# with a joint and are deliberately excluded (their tasks assert success via joint
# predicates or a physical container, not a frozen marker).
_STATIC_TARGET_FRAME_KINDS = ("world", "placement")


def resolve_static_target_positions(env, target_frames: dict | None) -> dict[str, np.ndarray]:
    """World xyz of statically-anchored targets, resolved against the live world.

    Uses the same frame seam (``resolve_frame``) the motion layer uses, so a
    placement affordance on a fixture (e.g. a shelf surface, possibly yawed)
    becomes the exact world point the place aimed at. Lets success/diagnostics
    treat asset-anchored static targets as markers — the gap that left
    ``on_placement`` targets out of ``target_positions`` (KeyError at eval).
    """
    if not target_frames:
        return {}
    scene_state = _live_scene_state(env, target_frames, [])
    return {
        name: np.asarray(scene_state["positions"][name], dtype=np.float32)
        for name, frame in target_frames.items()
        if getattr(frame, "kind", "world") in _STATIC_TARGET_FRAME_KINDS
    }


def _resolved_entity_positions(env) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    for name, placed in env.placed_map.items():
        positions[name] = np.asarray(placed.position, dtype=np.float32)
    return positions


def _is_fixture(env, name: str) -> bool:
    """True if the named entity is a fixed-base fixture (anchored at load time)."""
    placed = env.placed_map.get(name)
    asset = getattr(placed, "asset", None)
    return bool(getattr(asset, "is_fixture", False))


def _live_scene_state(
    env,
    target_frames: dict,
    skill_traces: list[dict],
) -> dict:
    scene_state = {
        "home_qpos": HOME_QPOS.copy(),
        "positions": collect_object_positions(env),
        "object_heights": env.object_heights,
        "assets": env.asset_map,
        "object_quats": env.object_quats,
        "table_surface_z": env.table_surface_z,
        "joint_positions": (
            env.get_joint_positions() if hasattr(env, "get_joint_positions") else {}
        ),
        "skill_traces": skill_traces,
        # Frames carry each target_position's articulated anchor (.parent); skills
        # use them for phase-aware collision-world exemption of the manipulated link.
        "target_frames": target_frames,
    }
    # Geometric candidates are judged by trying them from the live scene (snapshot, replay, restore).
    if isinstance(getattr(env, "planner", None), GeometricGraspPlanner):
        scene_state["grasp_trial"] = env.trial_from_here
    # Resolve every named target against the live world so e.g. a tray slot
    # tracks the drawer's current opening instead of a static "fully open" point.
    for name, frame in target_frames.items():
        scene_state["positions"][name] = np.asarray(
            resolve_frame(frame, scene_state), dtype=np.float32
        )
    return scene_state
