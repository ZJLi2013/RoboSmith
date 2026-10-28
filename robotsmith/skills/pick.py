"""``pick`` primitive: select an executable grasp and lift the object.

Candidates are tried in simulation (``scene_state["grasp_trial"]``) and the first that lifts the target is
executed; without a trial callback the planner's first candidate is used. Motion goes to ``MotionExecutor``;
the grasped payload is frozen as an attached collision object (ACO) so downstream transport/place plan around it.
"""

from __future__ import annotations

import logging

import numpy as np

from robotsmith.grasp.planner import GraspPlan
from robotsmith.grasp.transforms import pose_matrix
from robotsmith.motion.params import MotionParams
from robotsmith.motion.rocrobo_world import (
    PANDA_FINGER_LINKS,
    build_obstacle_world,
    build_payload_spheres,
)
from robotsmith.skills.base import Skill, SkillContext
from robotsmith.skills.registry import register

logger = logging.getLogger(__name__)


def _pick_trace_entry(skill: Skill, category: str, plan) -> dict:
    """Flatten the selected grasp plan into a pick ``skill_trace`` record."""
    def _vec(v):
        return [float(x) for x in v] if v is not None else None

    return {
        "skill": skill.name,
        "target": skill.target,
        "category": category,
        "selected_candidate_index": plan.metadata.get("candidate_index"),
        "selected_grasp_pos": _vec(plan.grasp_pos),
        "selected_pre_grasp_pos": _vec(plan.pre_grasp_pos),
        "selected_retreat_pos": _vec(plan.retreat_pos),
    }


def _candidates(ctx: SkillContext) -> tuple[list[GraspPlan], dict]:
    """The planner's grasp candidates for the target, and how many each of its filters removed."""
    skill = ctx.skill
    scene_state = ctx.scene_state
    obj_h = (
        skill.params.get("object_height")
        or scene_state.get("object_heights", {}).get(skill.target)
    )
    plans = ctx.planner.plan(
        ctx.obj_pos,
        object_quat=scene_state.get("object_quats", {}).get(skill.target),
        category=ctx.category,
        asset=scene_state.get("assets", {}).get(skill.target),
        object_height=obj_h,
    )
    return plans, dict(getattr(ctx.planner, "last_filter_counts", {}) or {})


def _select_pick_plan(ctx: SkillContext) -> GraspPlan | None:
    """The planner's first candidate, for callers that run without a grasp trial."""
    skill = ctx.skill
    plans, filter_counts = _candidates(ctx)
    if not plans:
        ctx.trace({"skill": skill.name, "target": skill.target, "category": ctx.category,
                   "filter_counts": filter_counts, "no_candidates": True})
        return None
    ctx.trace(_pick_trace_entry(skill, ctx.category, plans[0]) | {"filter_counts": filter_counts})
    return plans[0]


def _pick_motion_params(ctx: SkillContext) -> MotionParams:
    """Pick timing = base params, but skip the lift hold when a place follows."""
    params = ctx.params
    next_is_place = (
        ctx.index + 1 < len(ctx.skills) and ctx.skills[ctx.index + 1].name == "place"
    )
    return MotionParams(
        max_joint_vel=params.max_joint_vel,
        fine_joint_vel=params.fine_joint_vel,
        dt=params.dt,
        min_steps=params.min_steps,
        finger_steps=params.finger_steps,
        grasp_settle_steps=params.grasp_settle_steps,
        approach_steps=params.approach_steps,
        descend_steps=params.descend_steps,
        grasp_hold_steps=params.grasp_hold_steps,
        lift_steps=params.lift_steps,
        lift_hold_steps=0 if next_is_place else params.lift_hold_steps,
        transport_steps=params.transport_steps,
        place_descend_steps=params.place_descend_steps,
        release_steps=params.release_steps,
        retreat_steps=params.retreat_steps,
    )


def _pick_seed_transit(
    ctx: SkillContext, plan: GraspPlan
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Reset to home + pre-aim the base joint at the grasp azimuth, with transit.

    Two failure modes this guards against:
      1. After an articulated drag (open/close) the arm is left in an extended
         pose; descending IK to a tabletop grasp gets stuck there.
      2. From the y=0 neutral home, Genesis IK under-rotates the base joint and
         lands short of off-axis (large |y|) targets, closing beside the object.
    The transit is a no-op when the arm already starts at this seed.
    """
    params = ctx.params
    home_qpos = ctx.scene_state.get("home_qpos")
    q0 = np.asarray(ctx.qpos, dtype=np.float64)
    seed = (
        np.asarray(home_qpos, dtype=np.float64).copy()
        if home_qpos is not None
        else q0.copy()
    )
    seed[0] = float(np.arctan2(float(plan.grasp_pos[1]), float(plan.grasp_pos[0])))
    transit: list[np.ndarray] = []
    if not np.allclose(q0[:7], seed[:7], atol=1e-3):
        steps = max(params.approach_steps, params.min_steps)
        transit = [q0 + (seed - q0) * (i + 1) / steps for i in range(steps)]
    return seed, transit


def _capture_payload_aco(ctx: SkillContext, plan: GraspPlan) -> None:
    """Freeze the grasped payload as an ACO (spheres + T_ee_obj) on the executor.

    T_ee_obj is frozen at the grasp instant (inv(T_world_ee) @ T_world_obj) — a
    live object pose at place time would drift with physical slip. Spheres are in
    the object frame; serve rides them on the EE via T_ee_obj. Stored on the
    executor (persists across skills); cleared on release in ``place``.
    """
    skill = ctx.skill
    asset = ctx.scene_state.get("assets", {}).get(skill.target)
    payload_spheres = build_payload_spheres(asset)
    if not payload_spheres:
        ctx.executor.held_attach = None
        return
    quats = ctx.scene_state.get("object_quats", {})
    obj_quat = np.asarray(quats.get(skill.target, (1.0, 0.0, 0.0, 0.0)), dtype=np.float64)
    T_world_ee = pose_matrix(
        np.asarray(plan.grasp_pos, dtype=np.float64),
        np.asarray(plan.grasp_quat, dtype=np.float64),
    )
    T_world_obj = pose_matrix(np.asarray(ctx.obj_pos, dtype=np.float64), obj_quat)
    T_ee_obj = np.linalg.inv(T_world_ee) @ T_world_obj
    ctx.executor.held_attach = {
        "ee_link": "hand",
        "T_ee_obj": [[float(v) for v in row] for row in T_ee_obj],
        "spheres": payload_spheres,
        "exempt_self_links": list(PANDA_FINGER_LINKS),
    }


def _select_by_trial(ctx: SkillContext, trial) -> tuple[GraspPlan, list[np.ndarray]] | None:
    """Try candidates in planner order from the same start state; keep the first that lifts the target.

    ``trial(trajectory, target)`` replays the trajectory in simulation and returns
    ``{"lifted", "lift_m", "seconds"}``. With ``scene_state["grasp_trial_all"]`` every candidate is
    tried, for probes that need the executable fraction.
    """
    plans, filter_counts = _candidates(ctx)
    try_all = bool(ctx.scene_state.get("grasp_trial_all"))
    trials, chosen = [], None
    for plan in plans:
        try:
            traj = _pick_motion(ctx, plan)
            result = trial(traj, ctx.skill.target)
        except RuntimeError as exc:
            traj, result = None, {"lifted": False, "lift_m": None, "seconds": 0.0, "motion_error": str(exc)}
        md = plan.metadata
        trials.append({"candidate_index": md.get("candidate_index"), "kind": md.get("kind"),
                       "closing_width": md.get("closing_width"), **result})
        if result["lifted"] and chosen is None:
            chosen = (plan, traj)
            if not try_all:
                break
    ctx.trace({
        "skill": ctx.skill.name, "target": ctx.skill.target, "category": ctx.category,
        "filter_counts": filter_counts, "no_candidates": not plans, "no_trial_success": chosen is None,
        "selected_candidate_index": chosen[0].metadata.get("candidate_index") if chosen else None,
        "trials": trials, "trial_seconds": round(sum(t["seconds"] for t in trials), 3),
    })
    return chosen


def _pick_motion(ctx: SkillContext, plan: GraspPlan) -> list[np.ndarray]:
    """Base-aim transit plus the executor's pick segment for ``plan``."""
    seed, transit = _pick_seed_transit(ctx, plan)
    # Collision-aware backend: build the obstacle world from every scene fixture so
    # the free pick segments (home→pre-grasp, lift/retreat) route through
    # plan_motion instead of punching a straight IK line through a shelf/cabinet.
    # The grasp target is role `object`, not a fixture, so it is never in the world
    # (no exemption needed). Collision-blind runs pass world=None (unchanged).
    world = (
        build_obstacle_world(
            ctx.scene_state,
            ground_z=float(ctx.scene_state.get("table_surface_z", 0.0)),
        )
        if ctx.motion_planner.collision_aware
        else None
    )
    seg = ctx.executor.pick(
        plan, ctx.motion_planner, seed, _pick_motion_params(ctx), world=world
    )
    return transit + seg


@register("pick")
def run_pick(ctx: SkillContext) -> list[np.ndarray]:
    trial = ctx.scene_state.get("grasp_trial")
    if trial is not None:
        chosen = _select_by_trial(ctx, trial)
        plan, traj = chosen if chosen else (None, None)
    else:
        plan = _select_pick_plan(ctx)
        traj = _pick_motion(ctx, plan) if plan is not None else None
    if plan is None:
        logger.debug("[skills] no executable grasp candidate; skip pick execution")
        return [
            ctx.qpos.copy()
            for _ in range(max(ctx.params.min_steps, ctx.params.finger_steps))
        ]
    ctx.held_category = ctx.category
    # Grasp-relative offset: EE-flange height above the picked object's *support
    # surface* (object bottom = center − half-height), NOT above the table. A
    # later place re-adds this to the destination marker's world z so the object
    # bottom lands on that surface — correct for table→table AND shelf→shelf.
    # For a tabletop pick this equals the old (grasp_z − table_z). See
    # docs/features/feature12_pick_place_into_drawer.md (§5 W2).
    obj_h = float(ctx.scene_state.get("object_heights", {}).get(ctx.skill.target, 0.0) or 0.0)
    support_z = float(ctx.obj_pos[2]) - obj_h / 2.0
    ctx.held_place_z = max(float(plan.grasp_pos[2]) - support_z, 0.0)
    _capture_payload_aco(ctx, plan)
    return traj
