"""``open`` / ``close`` primitives: drag an articulated handle along its joint.

Both share one ``_run_drag`` body — the only difference is travel direction.
The handle anchor, drag vector, and collision-world link exemptions come from
``robotsmith.skills.frames``; the Cartesian drag itself runs in ``MotionExecutor``.
"""

from __future__ import annotations

import numpy as np

from robotsmith.grasp.transforms import TOP_DOWN_QUAT
from robotsmith.motion.rocrobo_world import build_obstacle_world
from robotsmith.skills.base import SkillContext
from robotsmith.skills.frames import _handle_links, _resolve_handle_drag
from robotsmith.skills.registry import register


def _run_drag(ctx: SkillContext, *, opening: bool) -> list[np.ndarray]:
    handle_world, move_vec, moving_link = _resolve_handle_drag(ctx, opening=opening)
    action = "open" if opening else "close"
    # Top-down grasp on the knob neck: fingers close across the thin stem and the
    # outboard ball is the pull-direction stop (form closure), so the drawer
    # follows the hand. Backing off is vertical and the drag stays at constant
    # height — no base-joint fold.
    grasp_quat = TOP_DOWN_QUAT.copy()
    # The inbound `approach` free segment is routed through collision-free
    # plan_motion (descend/drag/push/lift/retreat are Cartesian inside drag_handle).
    # Its goal sits on the handle knob — the only part meant to be touched — so we
    # exempt just the handle link from the planner world (phase-aware exemption).
    # The drawer body and the fixed carcass (`base`) stay hard obstacles, so the
    # planner must find a genuinely collision-free approach onto the knob (which
    # stands ~70 mm off the drawer face) instead of being allowed to graze the
    # whole drawer. Assets whose handle still rides the drawer link fall back to
    # exempting that moving link.
    exempt = _handle_links(ctx) or ([moving_link] if moving_link else [])
    drag_kwargs = {}
    if ctx.motion_planner.collision_aware:
        drag_kwargs["world"] = build_obstacle_world(
            ctx.scene_state,
            ground_z=float(ctx.scene_state.get("table_surface_z", 0.0)),
            exempt_links=exempt,
        )
    seg = ctx.executor.drag_handle(
        handle_world,
        grasp_quat,
        move_vec,
        ctx.motion_planner,
        ctx.qpos,
        ctx.params,
        action=action,
        **drag_kwargs,
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
    return seg


@register("open")
def run_open(ctx: SkillContext) -> list[np.ndarray]:
    return _run_drag(ctx, opening=True)


@register("close")
def run_close(ctx: SkillContext) -> list[np.ndarray]:
    return _run_drag(ctx, opening=False)
