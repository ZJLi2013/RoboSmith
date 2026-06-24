"""Episode object placement helpers for scripted rollouts."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from robotsmith.tasks import find_leaf
from robotsmith.tasks.task_spec import TaskSpec


def _sample_xy(rng: random.Random, x_range, y_range):
    return (rng.uniform(*x_range), rng.uniform(*y_range))


def _sample_spaced(rng, n, x_range, y_range, min_dist=0.10):
    pts = []
    for _ in range(n):
        for _try in range(200):
            x, y = _sample_xy(rng, x_range, y_range)
            if all(np.hypot(x - px, y - py) >= min_dist for px, py in pts):
                pts.append((x, y))
                break
        else:
            pts.append(_sample_xy(rng, x_range, y_range))
    return pts


def _sample_collision_free(rng, n, x_range, y_range, occupied, min_dist=0.12):
    pts = []
    for _ in range(n):
        for _try in range(500):
            x, y = _sample_xy(rng, x_range, y_range)
            all_pts = occupied + pts
            if all(np.hypot(x - px, y - py) >= min_dist for px, py in all_pts):
                pts.append((x, y))
                break
        else:
            pts.append(_sample_xy(rng, x_range, y_range))
    return pts


def _sample_line_targets(
    rng,
    n,
    x_range,
    y_range,
    occupied,
    axis="y",
    spacing=0.12,
    min_dist=0.10,
):
    """Sample collision-free target XYs along one workspace axis."""
    along_range = y_range if axis == "y" else x_range
    cross_range = x_range if axis == "y" else y_range
    total_span = (n - 1) * spacing

    for _try in range(500):
        cross_val = rng.uniform(*cross_range)
        along_start = rng.uniform(along_range[0], along_range[1] - total_span)
        pts = []
        for i in range(n):
            along_val = along_start + i * spacing
            xy = (cross_val, along_val) if axis == "y" else (along_val, cross_val)
            pts.append(xy)
        if all(np.hypot(px - ox, py - oy) >= min_dist for px, py in pts for ox, oy in occupied):
            return pts

    return pts


def derive_skill_targets(task_spec: TaskSpec) -> tuple[list[str], list[str]]:
    pick_names: list[str] = []
    place_names: list[str] = []
    for sk in task_spec.skills:
        if sk.name == "pick" and sk.target not in pick_names:
            pick_names.append(sk.target)
        elif sk.name == "place" and sk.target not in place_names:
            place_names.append(sk.target)
    return pick_names, place_names


def build_episode_positions(
    env: Any,
    rng: random.Random,
    pick_names: list[str],
    place_names: list[str],
    task_spec: TaskSpec | None = None,
) -> dict[str, np.ndarray]:
    x_range = env.x_range
    y_range = env.y_range
    positions: dict[str, np.ndarray] = {}

    n_pick = len(pick_names)
    pick_xys = _sample_spaced(rng, n_pick, x_range, y_range, min_dist=0.10)

    for name, (x, y) in zip(pick_names, pick_xys):
        z = env.get_initial_z(name)
        positions[name] = np.array([x, y, z])

    if place_names:
        align_axis = None
        aligned = find_leaf(task_spec.success, "objects_aligned") if task_spec else None
        if aligned is not None:
            align_axis = aligned.params.get("axis", "y")

        if align_axis and len(place_names) > 1:
            place_xys = _sample_line_targets(
                rng,
                len(place_names),
                x_range,
                y_range,
                occupied=list(pick_xys),
                axis=align_axis,
                spacing=0.12,
                min_dist=0.10,
            )
        else:
            place_xys = _sample_collision_free(
                rng,
                len(place_names),
                x_range,
                y_range,
                occupied=list(pick_xys),
                min_dist=0.12,
            )
        for name, (px, py) in zip(place_names, place_xys):
            z = env.get_initial_z(pick_names[0])
            positions[name] = np.array([px, py, z])

    return positions
