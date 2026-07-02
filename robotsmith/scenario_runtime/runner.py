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

from robotsmith.sim.franka import HOME_QPOS, to_numpy_f32
from robotsmith.skills import plan_segment, resolve_frame, skill_phases
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
    # with its base at z=0, so the backend rebases all poses/world geometry
    # (drawer_feature.md, coordinate-frame fix).
    return RocRoboBackend(env.solve_ik, base_z=float(getattr(env, "table_surface_z", 0.0)))


@dataclass
class ScenarioEpisodeResult:
    """Outcome of one driven episode (the frames are already in the dataset)."""

    initial_positions: dict[str, np.ndarray]
    skill_traces: list[dict]
    frame_count: int


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
    target_frames = target_frames or {}

    # Reset free bodies to their resolved world pose (full xyz, not just XY) and
    # settle. The resolver is the single source of truth for the spawn z (shelf /
    # in-drawer starts keep their height); the sim must not recompute it from the
    # table surface. Articulated assets stay anchored at their built pose, and
    # fixtures (fixed-base weldments like a shelf) are anchored at load time and
    # must not be re-set. See docs/design.md §5.1 and docs/features/feature12_pick_place_into_drawer.md §5.
    articulated = set(getattr(env, "articulated_joint_dofs", {}) or {})
    obj_xyz_map = {
        name: (float(pos[0]), float(pos[1]), float(pos[2]))
        for name, pos in _resolved_entity_positions(env).items()
        if name in env.entity_map
        and name not in articulated
        and not _is_fixture(env, name)
    }
    env.reset(obj_xyz_map, settle_steps=settle_steps)

    initial_positions = collect_object_positions(env)
    skill_traces: list[dict] = []
    qpos = HOME_QPOS.copy()
    held_category: str | None = None
    held_place_z: float | None = None
    frame_count = 0
    motion_planner = _maybe_motion_planner(env)

    planner_kw = {"motion_planner": motion_planner} if motion_planner is not None else {}
    for i, skill in enumerate(task_spec.skills):
        # Per-phase closed loop: re-sense + re-anchor before each phase, not just
        # at the subtask boundary. ``skill_phases`` returns one phase for static
        # anchors (== legacy per-subtask loop) and splits dynamic-anchor places
        # into transport/descend so descend re-reads the recoiled drawer.
        for phase in skill_phases(skill, target_frames.get(skill.target)):
            scene_state = _live_scene_state(env, target_frames, skill_traces)
            seg, held_category, held_place_z = plan_segment(
                skill, i, task_spec.skills, env.planner, executor, env.solve_ik,
                scene_state, motion_params, qpos, held_category, held_place_z,
                phase=phase, **planner_kw,
            )
            record_fn(
                env,
                dataset,
                seg,
                task_spec,
                primary_entity=None,
                camera_names=camera_names,
                env_state_names=env_state_names,
            )
            frame_count += len(seg)
            if seg:
                qpos = seg[-1]

    return ScenarioEpisodeResult(
        initial_positions=initial_positions,
        skill_traces=skill_traces,
        frame_count=frame_count,
    )


def evaluate_scenario_task(
    env,
    task_spec: TaskSpec,
    *,
    initial_positions: dict[str, np.ndarray],
    reference_positions: dict[str, np.ndarray] | None = None,
) -> bool:
    """Evaluate a generated scenario task against the live final state."""

    object_positions: dict[str, np.ndarray] = {}
    for name, ent in env.entity_map.items():
        object_positions[name] = to_numpy_f32(ent.get_pos()).copy()
    env_state = {
        "object_positions": object_positions,
        "initial_positions": initial_positions,
        "target_positions": reference_positions or {},
        "joint_positions": (
            env.get_joint_positions() if hasattr(env, "get_joint_positions") else {}
        ),
    }
    return evaluate_success(task_spec.success, env_state)


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
        "object_scales": env.object_scales,
        "table_surface_z": env.table_surface_z,
        "joint_positions": (
            env.get_joint_positions() if hasattr(env, "get_joint_positions") else {}
        ),
        "skill_traces": skill_traces,
        # Frames carry each target_position's articulated anchor (.parent); skills
        # use them for phase-aware collision-world exemption of the manipulated link.
        "target_frames": target_frames,
    }
    # Resolve every named target against the live world so e.g. a tray slot
    # tracks the drawer's current opening instead of a static "fully open" point.
    for name, frame in target_frames.items():
        scene_state["positions"][name] = np.asarray(
            resolve_frame(frame, scene_state), dtype=np.float32
        )
    return scene_state
