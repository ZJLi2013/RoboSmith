"""``pick`` primitive: select an executable grasp and lift the object.

Delegates candidate selection to ``robotsmith.grasp.feasibility`` and motion to
``MotionExecutor``; freezes the grasped payload as an attached collision object
(ACO) so downstream transport/place plan around it.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from robotsmith.grasp.feasibility import select_best_grasp_plan
from robotsmith.grasp.planner import GraspPlan
from robotsmith.grasp.transforms import pose_matrix
from robotsmith.motion.params import MotionParams
from robotsmith.motion.rocrobo_world import PANDA_FINGER_LINKS, build_payload_spheres
from robotsmith.skills.base import Skill, SkillContext
from robotsmith.skills.frames import _float_or_none
from robotsmith.skills.registry import register

logger = logging.getLogger(__name__)


def _policy_buckets(value) -> list[str] | None:
    if isinstance(value, str):
        return [value] if value else None
    if isinstance(value, (list, tuple)):
        buckets = [str(bucket) for bucket in value if bucket]
        return buckets or None
    return None


def _pick_trace_entry(skill: Skill, category: str, object_scale: float, plan) -> dict:
    """Flatten the selected grasp plan into a pick ``skill_trace`` record."""
    md = plan.metadata
    grasp_pos = plan.grasp_pos
    pre_grasp_pos = plan.pre_grasp_pos
    retreat_pos = plan.retreat_pos
    p0_ok_count = md.get("feasibility_hard_ok_count")
    p0_candidate_count = md.get("feasibility_candidate_count")
    return {
        "skill": skill.name,
        "target": skill.target,
        "category": category,
        "object_scale": float(object_scale),
        "selected_candidate_index": md.get("candidate_index"),
        "selected_policy_bucket": md.get("selected_policy_bucket"),
        "selected_policy_bucket_rank": md.get("selected_policy_bucket_rank"),
        "requested_policy_bucket": md.get("requested_policy_bucket"),
        "requested_policy_bucket_missing": md.get("requested_policy_bucket_missing"),
        "selected_approach_bin": md.get("approach_bin"),
        "selected_orientation_bin": md.get("orientation_bin"),
        "selected_grasp_pos": (
            [float(v) for v in grasp_pos] if grasp_pos is not None else None
        ),
        "selected_pre_grasp_pos": (
            [float(v) for v in pre_grasp_pos] if pre_grasp_pos is not None else None
        ),
        "selected_retreat_pos": (
            [float(v) for v in retreat_pos] if retreat_pos is not None else None
        ),
        "hard_ok": md.get("feasibility_hard_ok"),
        "no_feasible_grasp": bool(md.get("no_feasible_grasp", False)),
        "p0_feasible": (
            f"{p0_ok_count}/{p0_candidate_count}"
            if p0_ok_count is not None and p0_candidate_count is not None
            else None
        ),
        "p0_feasible_count": p0_ok_count,
        "p0_candidate_count": p0_candidate_count,
        "p0_support_ok_count": md.get("feasibility_support_ok_count"),
        "p0_support_reject_count": md.get("feasibility_support_reject_count"),
        "p0_bucket_candidate_counts": md.get("feasibility_bucket_candidate_counts"),
        "p0_bucket_hard_ok_counts": md.get("feasibility_bucket_hard_ok_counts"),
        "score_exec": _float_or_none(md.get("score_exec")),
    }


def _select_pick_plan(ctx: SkillContext) -> GraspPlan:
    """Resolve grasp candidates for the target and pick the most executable one."""
    skill = ctx.skill
    scene_state = ctx.scene_state
    assets = scene_state.get("assets", {})
    quats = scene_state.get("object_quats", {})
    scales = scene_state.get("object_scales", {})
    asset = assets.get(skill.target)
    obj_h = (
        skill.params.get("object_height")
        or scene_state.get("object_heights", {}).get(skill.target)
    )
    bucket_priority = _policy_buckets(
        getattr(asset.metadata, "grasp_policy_bucket", None)
        if asset is not None
        else None
    )
    plans = ctx.planner.plan(
        ctx.obj_pos,
        object_quat=quats.get(skill.target),
        category=ctx.category,
        asset=asset,
        object_height=obj_h,
        scale=scales.get(skill.target, 1.0),
    )
    if len(plans) > 1:
        plan = select_best_grasp_plan(
            plans,
            ctx.motion_planner.solve_ik,
            ctx.qpos,
            table_z=float(scene_state.get("table_surface_z", 0.0)),
            asset=asset,
            object_quat=quats.get(skill.target),
            object_scale=scales.get(skill.target, 1.0),
            category=ctx.category,
            bucket_priority=bucket_priority,
        )
    else:
        plan = plans[0]
    ctx.trace(
        _pick_trace_entry(skill, ctx.category, scales.get(skill.target, 1.0), plan)
    )
    return plan


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
    scales = ctx.scene_state.get("object_scales", {})
    asset = ctx.scene_state.get("assets", {}).get(skill.target)
    payload_spheres = build_payload_spheres(
        asset, scale=float(scales.get(skill.target, 1.0))
    )
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


@register("pick")
def run_pick(ctx: SkillContext) -> list[np.ndarray]:
    plan = _select_pick_plan(ctx)
    if (
        plan.metadata.get("no_feasible_grasp")
        and os.environ.get("GRASP_EVAL_SKIP_NO_FEASIBLE", "1") == "1"
    ):
        logger.debug("[skills] no feasible grasp candidate; skip pick execution")
        return [
            ctx.qpos.copy()
            for _ in range(max(ctx.params.min_steps, ctx.params.finger_steps))
        ]
    seed, transit = _pick_seed_transit(ctx, plan)
    seg = ctx.executor.pick(plan, ctx.motion_planner, seed, _pick_motion_params(ctx))
    table_z = float(ctx.scene_state.get("table_surface_z", 0.0))
    ctx.held_category = ctx.category
    ctx.held_place_z = max(float(plan.grasp_pos[2]) - table_z, 0.0)
    _capture_payload_aco(ctx, plan)
    return transit + seg
