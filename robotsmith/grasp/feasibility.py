"""P0 robot/table feasibility filtering for grasp candidates.

This module intentionally stays narrow: it filters GraspGen candidates that are
not robot-executable because IK fails, the grasp would require unsupported
bottom access, or the motion is otherwise kinematically invalid. It does not
attempt to evaluate contact stability; that belongs in a later contact-aware
evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Callable

import numpy as np

from robotsmith.grasp.bucket_policy import assign_policy_bucket
from robotsmith.grasp.planner import GraspPlan, Waypoint
from robotsmith.grasp.transforms import quat_wxyz_to_matrix
from robotsmith.motion.chain import _ik_joint_margin, evaluate_motion_chain
from robotsmith.motion.constants import N_ARM_JOINTS as _N_ARM_JOINTS

logger = logging.getLogger(__name__)


@dataclass
class EvaluatedGraspCandidate:
    plan: GraspPlan
    score_exec: float
    joint_dist: float
    policy_bucket: str
    hard_ok: bool


@dataclass
class GraspCandidateEvaluation:
    candidates: list[EvaluatedGraspCandidate]
    hard_ok_candidates: list[EvaluatedGraspCandidate]
    bucket_candidate_counts: dict[str, int]
    bucket_hard_ok_counts: dict[str, int]
    n_hard_ok: int
    n_support_ok: int


def _ik_chain_metrics(
    plan: GraspPlan,
    solve_ik: Callable,
    init_qpos: np.ndarray,
) -> dict:
    """Solve the full waypoint chain with seed continuity."""
    waypoints = plan.waypoints
    if not waypoints:
        waypoints = [
            Waypoint(plan.pre_grasp_pos, plan.pre_grasp_quat, plan.finger_open),
            Waypoint(plan.grasp_pos, plan.grasp_quat, plan.finger_open),
            Waypoint(plan.retreat_pos, plan.retreat_quat, plan.finger_closed),
        ]

    max_final_approach = float(
        os.environ.get("GRASP_EVAL_FINAL_APPROACH_JOINT_STEP_RAD", "0.8")
    )
    metrics = evaluate_motion_chain(
        waypoints,
        solve_ik,
        init_qpos,
        max_final_approach_joint_step=max_final_approach,
    )
    return {
        "ok": metrics.ok,
        "min_joint_margin": metrics.min_joint_margin,
        "total_joint_dist": metrics.total_joint_dist,
        "final_approach_joint_step": metrics.final_approach_joint_step,
    }


def _skipped_ik_metrics() -> dict:
    """IK metrics placeholder when an earlier P0 stage already rejected."""
    return {
        "ok": False,
        "min_joint_margin": np.nan,
        "total_joint_dist": np.nan,
        "final_approach_joint_step": np.nan,
    }


def _support_band_metrics(plan: GraspPlan, support_z: float) -> dict:
    """Reject grasps whose finger line enters the bottom no-entry band.

    ``support_z`` is the surface the object rests on (its own bottom = center −
    half-height), NOT necessarily the global table — so this works for objects
    on a shelf / inside a fixture as well as on the table. The support band is
    the thin bottom layer just above that surface; a grasp whose finger line
    enters it is asking for space between the object bottom and its support,
    which does not exist.
    """
    band_m = float(os.environ.get("GRASP_EVAL_SUPPORT_BAND_M", "0.01"))
    # ``finger_open`` is full jaw opening; each finger pad is half of that
    # distance from the grasp center along the closing axis.
    half_span = max(float(plan.finger_open) * 0.5, 0.02)
    R = quat_wxyz_to_matrix(plan.grasp_quat).astype(np.float64)
    finger_axis = R[:, 0]
    center = np.asarray(plan.grasp_pos, dtype=np.float64)
    ts = np.linspace(-half_span, half_span, 9)
    axis_points = center[None, :] + ts[:, None] * finger_axis[None, :]
    z_min = float(axis_points[:, 2].min())
    z_max = float(axis_points[:, 2].max())
    band_lo = float(support_z)
    band_hi = float(support_z + band_m)
    finger_intersects = bool(z_min <= band_hi and z_max >= band_lo)

    rejected = finger_intersects
    return {
        "ok": not rejected,
        "rejected": rejected,
        "stage": "support_band",
        "band_m": band_m,
        "band_z": [band_lo, band_hi],
        "finger_axis_z_range": [z_min, z_max],
        "finger_axis_rejected": finger_intersects,
    }


def _feasibility_score(
    plan: GraspPlan,
    solve_ik: Callable,
    init_qpos: np.ndarray,
    *,
    support_z: float,
) -> tuple[float, GraspPlan, dict]:
    """Apply P0 checks and rank passing candidates mostly by planner quality."""
    support = _support_band_metrics(plan, support_z)
    ik = (
        _ik_chain_metrics(plan, solve_ik, init_qpos)
        if support["ok"]
        else _skipped_ik_metrics()
    )
    hard_ok = ik["ok"] and support["ok"]
    if hard_ok:
        # P0 only proves robot/table executability. Keep kinematic terms as
        # tiny tie-breakers so they do not dominate grasp/contact quality.
        score = (
            float(plan.quality)
            + 1e-3 * ik["min_joint_margin"]
            - 1e-4 * ik["total_joint_dist"]
        )
    else:
        score = -100.0
        score += 0.01 * float(plan.quality)
        if np.isfinite(ik["min_joint_margin"]):
            score += min(ik["min_joint_margin"], 0.0)
    diag = {
        "ik": ik,
        "support": support,
        "hard_ok": hard_ok,
    }
    return float(score), plan, diag


def _feasibility_metadata(score: float, diag: dict) -> dict:
    ik = diag["ik"]
    support = diag["support"]
    return {
        "feasibility_hard_ok": bool(diag["hard_ok"]),
        "feasibility_support_band_m": support.get("band_m", np.nan),
        "feasibility_ik_min_margin": ik["min_joint_margin"],
        "feasibility_total_joint_dist": ik["total_joint_dist"],
        "feasibility_final_approach_joint_step": ik["final_approach_joint_step"],
        "score_exec": score,
    }


def evaluate_grasp_candidates(
    plans: list[GraspPlan],
    solve_ik: Callable,
    init_qpos: np.ndarray,
    *,
    support_z: float = 0.0,
) -> GraspCandidateEvaluation:
    """Run P0 checks and bucket bookkeeping without choosing a final policy."""
    candidates: list[EvaluatedGraspCandidate] = []
    hard_ok_candidates: list[EvaluatedGraspCandidate] = []
    bucket_candidate_counts: dict[str, int] = {}
    bucket_hard_ok_counts: dict[str, int] = {}
    n_hard_ok = 0
    n_support_ok = 0

    for plan in plans:
        policy_bucket = assign_policy_bucket(plan, support_z)
        bucket_candidate_counts[policy_bucket] = (
            bucket_candidate_counts.get(policy_bucket, 0) + 1
        )
        score, eval_plan, diag = _feasibility_score(
            plan,
            solve_ik,
            init_qpos,
            support_z=support_z,
        )
        if bool(diag.get("support", {}).get("ok", False)):
            n_support_ok += 1
        metadata = _feasibility_metadata(score, diag)
        plan.metadata = {**plan.metadata, **metadata}
        eval_plan.metadata = {**eval_plan.metadata, **metadata}

        candidate = EvaluatedGraspCandidate(
            plan=eval_plan,
            score_exec=float(score),
            joint_dist=float(diag["ik"]["total_joint_dist"]),
            policy_bucket=policy_bucket,
            hard_ok=bool(diag["hard_ok"]),
        )
        candidates.append(candidate)
        if candidate.hard_ok:
            n_hard_ok += 1
            bucket_hard_ok_counts[policy_bucket] = (
                bucket_hard_ok_counts.get(policy_bucket, 0) + 1
            )
            hard_ok_candidates.append(candidate)

    return GraspCandidateEvaluation(
        candidates=candidates,
        hard_ok_candidates=hard_ok_candidates,
        bucket_candidate_counts=bucket_candidate_counts,
        bucket_hard_ok_counts=bucket_hard_ok_counts,
        n_hard_ok=n_hard_ok,
        n_support_ok=n_support_ok,
    )


def select_from_bucket_policy(
    evaluation: GraspCandidateEvaluation,
    bucket_priority: tuple[str, ...] | list[str],
) -> tuple[GraspPlan, float, float] | None:
    """Select from explicit bucket evidence/policy; never invents a priority."""
    for rank, bucket in enumerate(bucket_priority):
        bucket_candidates = [
            candidate
            for candidate in evaluation.hard_ok_candidates
            if candidate.policy_bucket == bucket
        ]
        if bucket_candidates:
            selected = max(
                bucket_candidates,
                key=lambda candidate: (candidate.score_exec, -candidate.joint_dist),
            )
            selected.plan.metadata = {
                **selected.plan.metadata,
                "selected_policy_bucket": bucket,
                "selected_policy_bucket_rank": rank,
            }
            return selected.plan, selected.score_exec, selected.joint_dist
    return None


def _select_by_score_exec(
    evaluation: GraspCandidateEvaluation,
) -> tuple[GraspPlan, float, float]:
    pool = evaluation.hard_ok_candidates or evaluation.candidates
    selected = max(
        pool,
        key=lambda candidate: (candidate.score_exec, -candidate.joint_dist),
    )
    selected.plan.metadata = {
        **selected.plan.metadata,
        "selected_policy_bucket": selected.policy_bucket,
        "selected_policy_bucket_rank": None,
    }
    return selected.plan, selected.score_exec, selected.joint_dist


def _attach_evaluation_summary(
    plan: GraspPlan,
    evaluation: GraspCandidateEvaluation,
    n_plans: int,
) -> None:
    plan.metadata = {
        **plan.metadata,
        "feasibility_hard_ok_count": evaluation.n_hard_ok,
        "feasibility_candidate_count": n_plans,
        "feasibility_support_ok_count": evaluation.n_support_ok,
        "feasibility_support_reject_count": n_plans - evaluation.n_support_ok,
        "feasibility_bucket_candidate_counts": evaluation.bucket_candidate_counts,
        "feasibility_bucket_hard_ok_counts": evaluation.bucket_hard_ok_counts,
    }
    if evaluation.n_hard_ok == 0:
        plan.metadata["no_feasible_grasp"] = True


def select_best_grasp_plan(
    plans: list[GraspPlan],
    solve_ik: Callable,
    init_qpos: np.ndarray,
    *,
    support_z: float = 0.0,
    asset=None,
    object_quat: np.ndarray | None = None,
    object_scale: float = 1.0,
    category: str = "",
    bucket_priority: tuple[str, ...] | list[str] | None = None,
) -> GraspPlan:
    """Pick the most executable candidate from a multi-candidate plan list."""
    use_feasibility_eval = any(p.metadata.get("source") == "graspgen" for p in plans)
    init_arm = np.asarray(init_qpos, dtype=np.float64)[:_N_ARM_JOINTS]

    if not use_feasibility_eval:
        best_plan: GraspPlan = plans[0]
        best_score: float = -np.inf
        best_dev: float = np.inf
        for plan in plans:
            q = solve_ik(
                plan.grasp_pos, plan.grasp_quat, plan.finger_open, init_qpos=init_qpos
            )
            margin = _ik_joint_margin(q)
            dev = float(
                np.linalg.norm(
                    np.asarray(q, dtype=np.float64)[:7] - init_arm,
                )
            )
            better = margin > best_score + 1e-6 or (
                abs(margin - best_score) <= 1e-6 and dev < best_dev
            )
            if better:
                best_plan = plan
                best_score = margin
                best_dev = dev
            plan.metadata = {
                **plan.metadata,
                "ik_joint_margin": margin,
                "ik_arm_dev": dev,
            }
        sel = best_plan.metadata.get("yaw_index", 0)
        n = best_plan.metadata.get("n_yaw_candidates", len(plans))
        logger.debug(
            f"[grasp] selected candidate {sel}/{n} "
            f"(joint_margin={best_score:+.3f}, arm_dev={best_dev:.3f} rad)"
        )
        return best_plan

    evaluation = evaluate_grasp_candidates(
        plans,
        solve_ik,
        init_qpos,
        support_z=support_z,
    )
    policy_selection = (
        select_from_bucket_policy(evaluation, bucket_priority)
        if bucket_priority
        else None
    )
    if policy_selection is None:
        best_plan, best_score, best_dev = _select_by_score_exec(evaluation)
        if bucket_priority:
            best_plan.metadata = {
                **best_plan.metadata,
                "requested_policy_bucket": bucket_priority[0],
                "requested_policy_buckets": list(bucket_priority),
                "requested_policy_bucket_missing": True,
                "no_feasible_grasp": True,
            }
    else:
        best_plan, best_score, best_dev = policy_selection
        selected_bucket = best_plan.metadata.get(
            "selected_policy_bucket",
            bucket_priority[0],
        )
        best_plan.metadata = {
            **best_plan.metadata,
            "requested_policy_bucket": selected_bucket,
            "requested_policy_buckets": list(bucket_priority),
            "requested_policy_bucket_missing": False,
        }
    _attach_evaluation_summary(best_plan, evaluation, len(plans))
    sel = best_plan.metadata.get("yaw_index", 0)
    # Reached only on the feasibility-eval path (the non-eval path returned above).
    logger.debug(
        f"[grasp] p0 feasibility: feasible={evaluation.n_hard_ok}/{len(plans)}; "
        f"support_ok={evaluation.n_support_ok}/{len(plans)}; "
        f"bucket={best_plan.metadata.get('selected_policy_bucket', best_plan.metadata.get('policy_bucket', 'none'))}; "
        f"selected #{best_plan.metadata.get('candidate_index', sel)} "
        f"score_exec={best_score:+.3f} "
        f"joint_margin={best_plan.metadata.get('feasibility_ik_min_margin', np.nan):+.3f} "
        f"joint_dist={best_plan.metadata.get('feasibility_total_joint_dist', np.nan):.3f} "
        f"final_step={best_plan.metadata.get('feasibility_final_approach_joint_step', np.nan):.3f} "
        f"support_band={best_plan.metadata.get('feasibility_support_band_m', np.nan):.3f}m"
    )
    return best_plan
