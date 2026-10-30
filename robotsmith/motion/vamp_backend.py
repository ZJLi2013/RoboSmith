"""VAMP motion backend: collision-aware free-segment planning on the CPU (vector-accelerated RRT-Connect).

Goals are solved with the Genesis IK the caller provides; VAMP checks them against the collision world, plans a
joint path, and the path is timed here so no joint moves more than ``max_joint_vel * control_dt`` per step.
The Franka is table-mounted at world z=``base_z``; VAMP plans with its base at the origin.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from robotsmith.motion.constants import N_ARM_JOINTS
from robotsmith.motion.planner import PlanResult
from robotsmith.motion.params import MotionParams

logger = logging.getLogger(__name__)

TABLE_HALF_EXTENT_M = 2.0
TABLE_HALF_THICKNESS_M = 0.05
# VAMP's Panda end-effector frame (panda_grasptarget) sits this far along the hand's +z from RoboSmith's hand link.
GRASPTARGET_OFFSET_M = 0.105
# Clearance kept between the table cutout and the base spheres that reach below the table plane.
BASE_CUTOUT_MARGIN_M = 0.005


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.asarray(q, dtype=np.float64) / np.linalg.norm(q))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_euler_xyz(r: np.ndarray) -> list[float]:
    """Euler angles (x, y, z) with ``R = Rz(z) @ Ry(y) @ Rx(x)``."""
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    if abs(r[2, 0]) < 1 - 1e-9:
        roll, yaw = math.atan2(r[2, 1], r[2, 2]), math.atan2(r[1, 0], r[0, 0])
    else:
        roll, yaw = math.atan2(-r[1, 2], r[1, 1]), 0.0
    return [roll, pitch, yaw]


def matrix_to_quat_xyzw(r: np.ndarray) -> list[float]:
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return [x, y, z, w]


def table_slabs(top: float, cutout: float) -> list[dict]:
    """The table under ``top`` as four slabs around a square hole of half-size ``cutout`` at the robot base."""
    h, t, c = TABLE_HALF_EXTENT_M, TABLE_HALF_THICKNESS_M, cutout
    z = top - t
    strip = (h - c) / 2
    return [
        {"center": [c + strip, 0.0, z], "euler_xyz": [0.0, 0.0, 0.0], "half_extents": [strip, h, t]},
        {"center": [-c - strip, 0.0, z], "euler_xyz": [0.0, 0.0, 0.0], "half_extents": [strip, h, t]},
        {"center": [0.0, c + strip, z], "euler_xyz": [0.0, 0.0, 0.0], "half_extents": [c, strip, t]},
        {"center": [0.0, -c - strip, z], "euler_xyz": [0.0, 0.0, 0.0], "half_extents": [c, strip, t]},
    ]


def to_vamp_primitives(world: list[dict], base_z: float, base_cutout: float = 0.0) -> list[dict]:
    """RoboSmith world primitives in the robot base frame, as VAMP cuboid arguments (center, euler_xyz, half_extents).
    An upward halfspace (the table the arm is mounted on) leaves a ``base_cutout`` hole around the base."""
    out = []
    for prim in world:
        if prim["type"] == "box":
            c = np.asarray(prim["center"], dtype=np.float64) - [0.0, 0.0, base_z]
            out.append({"center": c.tolist(),
                        "euler_xyz": matrix_to_euler_xyz(quat_wxyz_to_matrix(prim["quat"])),
                        "half_extents": (np.asarray(prim["extent"], dtype=np.float64) / 2).tolist()})
        elif prim["type"] == "halfspace":
            if list(prim["normal"]) != [0.0, 0.0, 1.0]:
                raise ValueError(f"only upward halfspaces are supported, got normal {prim['normal']}")
            top = float(prim["point"][2]) - base_z
            if base_cutout > 0:
                out.extend(table_slabs(top, base_cutout))
            else:
                out.append({"center": [0.0, 0.0, top - TABLE_HALF_THICKNESS_M], "euler_xyz": [0.0, 0.0, 0.0],
                            "half_extents": [TABLE_HALF_EXTENT_M, TABLE_HALF_EXTENT_M, TABLE_HALF_THICKNESS_M]})
        else:
            raise ValueError(f"unsupported world primitive {prim['type']!r}")
    return out


def hand_to_grasptarget() -> np.ndarray:
    """``T_grasptarget_hand``: hand-link coordinates expressed in VAMP's end-effector frame (panda_grasptarget)."""
    t = np.eye(4)
    t[2, 3] = -GRASPTARGET_OFFSET_M
    return t


def time_parameterize(path: np.ndarray, max_joint_vel: float, control_dt: float) -> np.ndarray:
    """Subdivide each edge so no joint moves more than ``max_joint_vel * control_dt`` per control step."""
    step = max_joint_vel * control_dt
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(math.ceil(float(np.max(np.abs(b - a))) / step)))
        out.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
    return np.asarray(out)


class VampBackend:
    collision_aware = True

    def __init__(self, solve_ik, *, base_z: float = 0.0, params: MotionParams | None = None, n_finger: int = 2,
                 joint_limits: tuple | None = None):
        import vamp

        self._vamp, self._robot = vamp, vamp.panda
        self._solve_ik = solve_ik
        self._base_z = float(base_z)
        self._max_joint_vel = (params or MotionParams()).max_joint_vel
        self._n_finger = n_finger
        self._limits = None if joint_limits is None else tuple(np.asarray(v, dtype=np.float64)[:N_ARM_JOINTS]
                                                                for v in joint_limits)
        _, _, self._plan_settings, self._simp_settings = vamp.configure_robot_and_planner_with_kwargs("panda", "rrtc")
        self._rng = self._robot.halton()
        self._base_cutout = self._below_plane_extent()

    def _below_plane_extent(self) -> float:
        """Half-size of the square around the base that holds every robot sphere reaching below z=0 at rest."""
        rest = np.zeros(N_ARM_JOINTS, dtype=np.float32)
        low = [s for s in self._robot.fk(rest) if s.z - s.r < 0.0]
        if not low:
            return 0.0
        return max(max(abs(s.x), abs(s.y)) + s.r for s in low) + BASE_CUTOUT_MARGIN_M

    def solve_ik(self, pos, quat, finger_pos, *, init_qpos=None, world=None, attach=None):
        return self._solve_ik(pos, quat, finger_pos, init_qpos=init_qpos)

    def environment(self, world: list[dict], attach: dict | None = None):
        env = self._vamp.Environment()
        for p in to_vamp_primitives(world, self._base_z, self._base_cutout):
            env.add_cuboid(self._vamp.Cuboid(p["center"], p["euler_xyz"], p["half_extents"]))
        if attach:
            t = hand_to_grasptarget() @ np.asarray(attach["T_ee_obj"], dtype=np.float64)
            att = self._vamp.Attachment(t[:3, 3].tolist(), matrix_to_quat_xyzw(t[:3, :3]))
            for s in attach["spheres"]:
                att.add_sphere(self._vamp.Sphere(list(s["center"]), float(s["radius"])))
            env.attach(att)
        return env

    def _arm(self, q) -> np.ndarray:
        return np.asarray(q, dtype=np.float32)[:N_ARM_JOINTS]

    def _pad(self, q_arm, finger_pos: float) -> np.ndarray:
        q = np.zeros(N_ARM_JOINTS + self._n_finger, dtype=np.float64)
        q[:N_ARM_JOINTS] = q_arm
        q[N_ARM_JOINTS:] = float(finger_pos)
        return q

    def plan_motion(self, q_start, waypoints, *, world=None, control_dt=None, attach=None):
        wps = list(waypoints)
        if world is None or not wps:
            return PlanResult(success=False, reason="no-world")
        env = self.environment(world, attach)
        q = self._arm(q_start)
        if not self._robot.validate(q, env):
            return PlanResult(success=False, reason="start-invalid")
        path = [q.astype(np.float64)]
        for wp in wps:
            goal = self._arm(self._solve_ik(wp.pos, wp.quat, wp.finger_width, init_qpos=self._pad(q, wp.finger_width)))
            if not self._robot.validate(goal, env):
                return PlanResult(success=False, reason="goal-invalid")
            res = self._robot.rrtc(q, goal, env, self._plan_settings, self._rng)
            if not res.solved:
                return PlanResult(success=False, reason="rrtc-failed")
            simple = self._robot.simplify(res.path, env, self._simp_settings, self._rng)
            pts = np.asarray(simple.path.numpy(), dtype=np.float64)
            path.extend(pts[1:])
            q = goal
        path = np.asarray(path)
        if self._limits is not None and (np.any(path < self._limits[0]) or np.any(path > self._limits[1])):
            return PlanResult(success=False, reason="outside-sim-joint-limits")
        traj = time_parameterize(path, self._max_joint_vel, float(control_dt or 0.01))
        finger = wps[-1].finger_width
        return PlanResult(success=True, trajectory=[self._pad(p, finger) for p in traj])
