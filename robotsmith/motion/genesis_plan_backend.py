"""Genesis ``plan_path`` motion backend: free segments planned against the simulator's own collision geometry.

``plan_path`` ignores every contact pair already present at its start or goal for the whole path, so a goal pressed
against a fixture would exempt that fixture everywhere. The backend therefore checks the trajectory it returns, point
by point with the held object riding on the hand, and rejects any contact. Planning moves the live scene; it runs
inside ``SimEnv.preserved_state`` and only sets qpos, so control targets are untouched.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from robotsmith.grasp.transforms import pose_matrix, rotmat_to_quat_wxyz
from robotsmith.motion.constants import N_ARM_JOINTS
from robotsmith.motion.params import MotionParams
from robotsmith.motion.planner import PlanResult

logger = logging.getLogger(__name__)

# Same endpoint gate as rocRobo's ``evaluate``.
REACH_POS_TOL_M = 0.010
REACH_ROT_TOL_DEG = 5.0
CONTACT_PENETRATION_M = 0.001
FINGER_LINKS = ("left_finger", "right_finger")
FIXED_BASE_LINK = "link0"


def time_parameterize(path: np.ndarray, max_joint_vel: float, control_dt: float) -> np.ndarray:
    """Resample ``path`` so every control step moves each joint by at most ``max_joint_vel * control_dt``.

    Points closer than one step to the last kept point are dropped, longer edges are subdivided; endpoints are kept.
    """
    step = max_joint_vel * control_dt
    kept = [path[0]]
    for i in range(1, len(path) - 1):
        if np.max(np.abs(path[i + 1] - kept[-1])) > step:
            kept.append(path[i])
    kept.append(path[-1])
    out = [kept[0]]
    for a, b in zip(kept[:-1], kept[1:]):
        n = max(1, int(math.ceil(float(np.max(np.abs(b - a))) / step)))
        out.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
    return np.asarray(out)


def rotation_error_deg(q1, q2) -> float:
    q1, q2 = (np.asarray(q, dtype=np.float64) / np.linalg.norm(q) for q in (q1, q2))
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(q1, q2))))))


def _np(x) -> np.ndarray:
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def _flat(x) -> np.ndarray:
    return _np(x).reshape(-1)


class GenesisPlanBackend:
    collision_aware = True

    def __init__(self, env, *, params: MotionParams | None = None):
        self._env = env
        self._franka = env.franka
        self._max_joint_vel = (params or MotionParams()).max_joint_vel
        self._links = {link.idx: link.name for link in self._franka.links}
        names = {id(e): n for n, e in env.entity_map.items()}
        self._owners = [(e.geom_start, e.geom_end, names.get(id(e), f"entity{e.idx}"))
                        for e in self._franka._solver.entities]

    def _owner(self, geom: int) -> str | None:
        return next((n for s, e, n in self._owners if s <= geom < e), None)

    def solve_ik(self, pos, quat, finger_pos, *, init_qpos=None, world=None, attach=None):
        return self._env.solve_ik(pos, quat, finger_pos, init_qpos=init_qpos)

    def _hand_pose(self, q) -> tuple[np.ndarray, np.ndarray]:
        self._franka.set_qpos(q, zero_velocity=False)
        hand = self._env.end_effector
        return _flat(hand.get_pos()).astype(np.float64), _flat(hand.get_quat()).astype(np.float64)

    def reach_error(self, q, pos, quat) -> tuple[float, float]:
        """Position (m) and orientation (deg) error of the hand at ``q`` from the requested pose."""
        p, r = self._hand_pose(q)
        return float(np.linalg.norm(p - np.asarray(pos, dtype=np.float64))), rotation_error_deg(r, quat)

    def contacts_at(self, q, held=None, t_ee_obj=None) -> list[dict]:
        """Contacts deeper than ``CONTACT_PENETRATION_M`` between the arm or the held object and the rest of the scene."""
        p, r = self._hand_pose(q)
        if held is not None:
            t = pose_matrix(p, r) @ t_ee_obj
            held.set_pos(t[:3, 3])
            held.set_quat(rotmat_to_quat_wxyz(t[:3, :3]))
        self._franka._solver._kernel_detect_collision()
        robot_geoms = range(self._franka.geom_start, self._franka.geom_end)
        held_geoms = range(held.geom_start, held.geom_end) if held is not None else range(0)
        found = []
        for entity in (self._franka, held) if held is not None else (self._franka,):
            c = entity.get_contacts()
            ga, gb, la, lb = (_flat(c[k]).astype(int) for k in ("geom_a", "geom_b", "link_a", "link_b"))
            pen = _flat(c["penetration"]).astype(np.float64)
            valid = _flat(c["valid_mask"]).astype(bool) if "valid_mask" in c else np.ones(len(ga), bool)
            for a, b, link_a, link_b, d, ok in zip(ga, gb, la, lb, pen, valid):
                if not ok or d <= CONTACT_PENETRATION_M:
                    continue
                names = {self._links.get(int(link_a)), self._links.get(int(link_b))}
                robot_side = (a in robot_geoms) + (b in robot_geoms)
                held_side = (a in held_geoms) + (b in held_geoms)
                if robot_side == 2 or FIXED_BASE_LINK in names:
                    continue
                if robot_side and held_side and names & set(FINGER_LINKS):
                    continue
                other = b if a in robot_geoms or a in held_geoms else a
                found.append({"links": sorted(n for n in names if n), "other": self._owner(int(other)),
                              "penetration_m": round(float(d), 5)})
        return found

    def first_contact(self, trajectory, attach=None) -> dict | None:
        held = self._env.entity_map[attach["target"]] if attach and attach.get("target") else None
        t_ee_obj = np.asarray(attach["T_ee_obj"], dtype=np.float64) if held is not None else None
        for i, q in enumerate(trajectory):
            hits = self.contacts_at(q, held, t_ee_obj)
            if hits:
                return {"index": i, "contacts": hits}
        return None

    def plan_motion(self, q_start, waypoints, *, world=None, control_dt=None, attach=None):
        wps = list(waypoints)
        if world is None or not wps:
            return PlanResult(success=False, reason="no-world")
        held = self._env.entity_map[attach["target"]] if attach and attach.get("target") else None
        kw = {"ee_link_name": "hand", "with_entity": held} if held is not None else {}
        q = np.asarray(q_start, dtype=np.float64).copy()
        path = [q[:N_ARM_JOINTS]]
        with self._env.preserved_state():
            for wp in wps:
                q[N_ARM_JOINTS:] = wp.finger_width
                goal = np.asarray(self.solve_ik(wp.pos, wp.quat, wp.finger_width, init_qpos=q), dtype=np.float64)
                pos_err, rot_err = self.reach_error(goal, wp.pos, wp.quat)
                if pos_err >= REACH_POS_TOL_M or rot_err >= REACH_ROT_TOL_DEG:
                    return PlanResult(success=False, reason="ik-miss",
                                      quality={"ik_pos_m": pos_err, "ik_rot_deg": rot_err})
                seg, valid = self._franka.plan_path(qpos_goal=goal, qpos_start=q, return_valid_mask=True, **kw)
                if not bool(_flat(valid)[0]):
                    return PlanResult(success=False, reason="plan-path-failed")
                path.extend(_np(seg).astype(np.float64).reshape(-1, goal.shape[0])[1:, :N_ARM_JOINTS])
                q = goal
            traj = time_parameterize(np.asarray(path), self._max_joint_vel, float(control_dt or 0.01))
            finger = wps[-1].finger_width
            full = [np.concatenate([p, [finger, finger]]) for p in traj]
            hit = self.first_contact(full, attach)
        if hit is not None:
            return PlanResult(success=False, reason="posthoc-contact", quality=hit)
        return PlanResult(success=True, trajectory=full)
