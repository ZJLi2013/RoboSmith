"""``open`` / ``close`` primitives: drag an articulated handle along its joint.

Both share one ``_run_drag`` body — the only difference is travel direction.
The handle anchor and drag vector come from ``robotsmith.skills.frames``; the
Cartesian drag itself runs in ``MotionExecutor``.
"""

from __future__ import annotations

from robotsmith.execution.gate import VerifiedMotion
from robotsmith.grasp.transforms import TOP_DOWN_QUAT
from robotsmith.skills.base import SkillContext
from robotsmith.skills.frames import _resolve_handle_drag
from robotsmith.skills.registry import bound_leaf, register


def _run_drag(ctx: SkillContext, *, opening: bool) -> VerifiedMotion:
    handle_world, move_vec, _ = _resolve_handle_drag(ctx, opening=opening)
    action = "open" if opening else "close"
    # Top-down grasp on the knob neck: fingers close across the thin stem and the
    # outboard ball is the pull-direction stop (form closure), so the drawer
    # follows the hand. Backing off is vertical and the drag stays at constant
    # height — no base-joint fold.
    grasp_quat = TOP_DOWN_QUAT.copy()
    seg = ctx.executor.drag_handle(
        handle_world,
        grasp_quat,
        move_vec,
        ctx.motion_planner,
        ctx.qpos,
        ctx.params,
        action=action,
    )
    ctx.trace(
        {
            "skill": ctx.skill.name,
            "target": ctx.skill.target,
            "handle_world": [float(v) for v in handle_world],
            "move_vec": [float(v) for v in move_vec],
            "motion_trace": getattr(ctx.executor, "last_motion_trace", None),
        }
    )
    return ctx.gate.ungated(seg)


@register("open", post=bound_leaf)
def run_open(ctx: SkillContext) -> VerifiedMotion:
    return _run_drag(ctx, opening=True)


@register("close", post=bound_leaf)
def run_close(ctx: SkillContext) -> VerifiedMotion:
    return _run_drag(ctx, opening=False)
