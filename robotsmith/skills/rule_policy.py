"""The fixed pick and place rules over ``ManipulationTools``.

A pick gates, then trials, the grasp candidates in planner order from the same start state and keeps the first that
passes both, verified again after the last trial. A candidate whose following single-phase place is refused from
where its pick ends is not tried. A place is the tools' place for the step.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from robotsmith.execution.gate import ExecutionFailed, MotionRejected, Refused, VerifiedMotion
from robotsmith.grasp.planner import GraspPlan
from robotsmith.grasp.transforms import pose_matrix
from robotsmith.skills.base import Skill, SkillContext, _resolve_skill_category
from robotsmith.skills.manipulation_tools import ManipulationTools


class RulePolicy:
    def pick(self, ctx: SkillContext, tools: ManipulationTools, *, place_follows: bool, place_next: Skill | None,
             trial, try_all: bool) -> tuple[VerifiedMotion, float]:
        """The verified pick of ``ctx.skill.target`` and the grasp-relative place height it hands to the next place.

        ``place_follows`` drops the lift hold; ``place_next`` is the following place when it is planned in one
        phase, checked from where each candidate's pick ends. ``trial`` is ``tools.trial`` or ``None``; without it
        the first candidate that passes the gate is used. ``try_all`` trials every candidate, for probes that need
        the executable fraction. When none passes, raises ``Refused("search_exhausted")`` if no trial ran, else
        ``ExecutionFailed`` with the trials' physics steps."""
        params = replace(ctx.params, lift_hold_steps=0 if place_follows else ctx.params.lift_hold_steps)
        plans = tools.grasp_candidates(ctx)
        target = ctx.skill.target
        trials, passing = [], []
        for plan in plans:
            md = plan.metadata
            record = {"candidate_index": md.get("candidate_index"), "kind": md.get("kind"),
                      "closing_width": md.get("closing_width"), "lifted": False, "lift_m": None, "seconds": 0.0}
            try:
                request, record["approach"] = tools.plan_pick(ctx, plan, params)
                motion = tools.verify(request)
                placeable = place_next is None or _place_check(ctx, tools, plan, request, place_next, record)
                if placeable and trial is not None:
                    record.update(trial(motion.trajectory, target), steps=len(motion.trajectory))
                    if record["lifted"]:
                        motion = tools.verify(request)
            except MotionRejected as exc:
                record["gate"] = exc.to_dict()
            except Refused as exc:
                record["approach"] = {"ok": False, **exc.evidence}
            except RuntimeError as exc:
                record["motion_error"] = str(exc)
            else:
                record["gate"] = "passed"
                if placeable and (trial is None or record["lifted"]):
                    passing.append((plan, request, motion, record))
            trials.append(record)
            if passing and not try_all:
                break
        chosen = None
        for plan, request, motion, record in passing:
            if try_all:
                # Every later trial stepped physics and voided this motion's voucher.
                try:
                    motion = tools.verify(request)
                except MotionRejected as exc:
                    record["reverify"] = exc.to_dict()
                    continue
            chosen = (plan, motion)
            break
        ctx.trace({
            "skill": ctx.skill.name, "target": target, "category": ctx.category,
            "no_candidates": not plans, "no_trial_success": chosen is None,
            "selected_candidate_index": chosen[0].metadata.get("candidate_index") if chosen else None,
            "trials": trials, "trial_seconds": round(sum(t["seconds"] for t in trials), 3),
        })
        if chosen is None:
            evidence = {"reason": "no pick candidate passed the gate and its trial", "candidates": len(plans),
                        "trials": trials}
            steps = sum(t.get("steps", 0) for t in trials)
            if steps:
                raise ExecutionFailed(evidence, steps)
            raise Refused("search_exhausted", evidence)
        plan, motion = chosen
        return motion, _held_place_z(ctx, plan)

    def place(self, ctx: SkillContext, tools: ManipulationTools) -> VerifiedMotion:
        return tools.plan_place(ctx)


def _place_check(ctx: SkillContext, tools: ManipulationTools, plan: GraspPlan, request: dict, place: Skill,
                 record: dict) -> bool:
    """Plan ``place`` with the arm at the end of this candidate's pick and the target at its grasp in the hand, and
    record the outcome as ``record["place"]``. The scene is unchanged on return, the planned motion is discarded and
    its trace does not reach the run's ``skill_traces``."""
    q_end = np.asarray(request["trajectory"][-1], dtype=np.float64)
    target = ctx.skill.target
    held_at = pose_matrix(*tools.env.hand_pose(q_end)) @ request["holding"].t_hand_object
    place_ctx = replace(ctx, skill=place, scene_state={**ctx.scene_state, "skill_traces": []},
                        qpos=q_end, category=_resolve_skill_category(place, ctx.scene_state.get("assets", {}),
                                                                     ctx.category),
                        held=target, held_category=ctx.category, held_place_z=_held_place_z(ctx, plan), phase="all")
    with tools.posed(q_end, {target: held_at}):
        try:
            tools.plan_place(place_ctx)
        except Refused as exc:
            record["place"] = {"ok": False, "label": exc.label, **exc.evidence}
            return False
    record["place"] = {"ok": True}
    return True


def _held_place_z(ctx: SkillContext, plan: GraspPlan) -> float:
    # Grasp-relative offset: EE-flange height above the picked object's *support
    # surface* (object bottom = center − half-height), NOT above the table. A
    # later place re-adds this to the destination marker's world z so the object
    # bottom lands on that surface — correct for table→table AND shelf→shelf.
    # For a tabletop pick this equals the old (grasp_z − table_z).
    obj_h = float(ctx.scene_state.get("object_heights", {}).get(ctx.skill.target, 0.0) or 0.0)
    support_z = float(ctx.obj_pos[2]) - obj_h / 2.0
    return max(float(plan.grasp_pos[2]) - support_z, 0.0)
