"""Genesis simulation environment for data collection.

Encapsulates scene loading, Franka setup, cameras, IK,
and reset logic. No episode/recording concerns.
"""
from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from robotsmith.sim.franka import (
    JOINT_NAMES, N_DOFS, HOME_QPOS, KP, KV,
    FORCE_LOWER, FORCE_UPPER, to_numpy,
)
from robotsmith.assets.schema import Asset
from robotsmith.grasp.geometric import GeometricGraspPlanner
from robotsmith.grasp.planner import GraspPlanner
from robotsmith.grasp.transforms import pose_matrix, rotmat_to_quat_wxyz
from robotsmith.scenes.config import SceneConfig

logger = logging.getLogger(__name__)


TARGET_MARKER_SIZE = (0.06, 0.06, 0.005)

# Viscous damping on passive articulated DOFs (e.g. drawer slide) so the
# drag-release velocity kick bleeds off instead of coasting the joint fully
# closed. Genesis-0.4.5 default (0.3) is far too low on MI300; 20.0 holds >0.1
# open even under a -2 m/s kick.
PASSIVE_JOINT_DAMPING = 20.0

# A healthy settle moves an object a few millimetres (die_01 lands on a face: 2.6 mm, 6 deg);
# beyond these the object rolled, tipped or fell, and the episode no longer starts where it was placed.
SETTLE_MAX_XY_M = 0.05
SETTLE_MAX_DROP_M = 0.05
SETTLE_MAX_TILT_DEG = 30.0


class SceneInvalid(RuntimeError):
    """Objects did not stay where they were placed through the settle steps."""


def _z_axis(q_wxyz) -> np.ndarray:
    w, x, y, z = (float(v) for v in q_wxyz)
    return np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])


def settle_violations(spawned: dict, settled: dict) -> list[str]:
    """Objects whose settled pose left the spawn pose; both map name -> (xyz, quat_wxyz)."""
    out = []
    for name, (p0, q0) in spawned.items():
        p1, q1 = settled[name]
        dxy = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        drop = float(p0[2] - p1[2])
        tilt = float(np.degrees(np.arccos(np.clip(_z_axis(q0) @ _z_axis(q1), -1.0, 1.0))))
        if dxy > SETTLE_MAX_XY_M or drop > SETTLE_MAX_DROP_M or tilt > SETTLE_MAX_TILT_DEG:
            out.append(f"{name}: moved {dxy:.3f} m, dropped {drop:.3f} m, tilted {tilt:.0f} deg")
    return out


def ensure_display():
    """Start Xvfb if no DISPLAY is set (headless rendering)."""
    if os.environ.get("DISPLAY"):
        return
    xvfb = subprocess.run(["which", "Xvfb"], capture_output=True)
    if xvfb.returncode != 0:
        return
    proc = subprocess.Popen(
        ["Xvfb", ":99", "-screen", "0", "1280x1024x24",
         "-ac", "+extension", "GLX"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = ":99"
    time.sleep(2)
    if proc.poll() is None:
        logger.debug("[display] Xvfb started (PID=%s)", proc.pid)


def render_cam(cam) -> np.ndarray:
    """Render an RGB image from a Genesis camera."""
    rgb, _, _, _ = cam.render(rgb=True, depth=False,
                              segmentation=False, normal=False)
    arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.array(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.uint8)


@dataclass
class SimEnv:
    """Wraps a Genesis scene with Franka + objects for data collection."""

    scene: object
    franka: object
    scene_config: SceneConfig

    entity_map: dict[str, object] = field(default_factory=dict)
    placed_map: dict = field(default_factory=dict)
    object_heights: dict[str, float] = field(default_factory=dict)

    cam_up: object = None
    cam_wrist: object = None
    end_effector: object = None
    motors_dof: list[int] = field(default_factory=list)
    arm_dof: list[int] = field(default_factory=list)
    finger_dof: list[int] = field(default_factory=list)
    asset_map: dict[str, Asset] = field(default_factory=dict)
    object_quats: dict[str, np.ndarray] = field(default_factory=dict)
    planner: Optional[GraspPlanner] = None
    table_surface_z: float = 0.0

    articulated_joint_dofs: dict[str, dict[str, int]] = field(default_factory=dict)
    """object name -> {joint name -> local DOF index} for articulated assets."""
    articulated_joint_init: dict[str, dict[str, float]] = field(default_factory=dict)
    """object name -> {joint name -> initial qpos} applied (passively) on reset."""

    on_step: Optional[Callable[["SimEnv"], None]] = None
    """Called after every physics step; for observers that must see each one (e.g. contact monitors)."""

    _default_finger_open: float = 0.04
    _writes: int = 0

    # ---- Factory ----

    @classmethod
    def build(
        cls,
        scene_config: SceneConfig,
        *,
        assets_root: str | Path | None = None,
        gs_module=None,
        seed: int = 42,
        fps: int = 30,
        cpu: bool = False,
        box_box_detection: bool = True,
        settle_steps: int = 30,
    ) -> SimEnv:
        """Create a fully-initialized SimEnv from a SceneConfig."""
        from robotsmith.assets.library import AssetLibrary
        from robotsmith.scenes.backend import ProgrammaticSceneBackend
        from robotsmith.scenes.genesis_loader import load_resolved_scene

        if gs_module is None:
            import genesis as gs_module  # noqa: N811
        gs = gs_module

        if not cpu:
            gs.init(backend=gs.gpu, logging_level="warning")
        else:
            gs.init(backend=gs.cpu, logging_level="warning")

        if assets_root is None:
            assets_root = str(
                Path(__file__).resolve().parent.parent.parent / "assets"
            )
        library = AssetLibrary(str(assets_root))
        backend = ProgrammaticSceneBackend(seed=seed)
        resolved = backend.resolve(scene_config, library)
        logger.debug("[scene] %s", resolved.summary())

        handle = load_resolved_scene(
            resolved, gs_module=gs, fps=fps,
            box_box_detection=box_box_detection,
        )

        table_z = (scene_config.table_height
                   + scene_config.table_size[2] / 2.0)

        entity_map: dict[str, object] = {}
        placed_map: dict = {}
        object_heights: dict[str, float] = {}
        asset_map: dict[str, Asset] = {}
        object_quats: dict[str, np.ndarray] = {}
        for name, entity, po in zip(
            handle.object_names, handle.objects, handle.placed
        ):
            entity_map[name] = entity
            placed_map[name] = po
            object_heights[name] = po.object_height_m
            asset_map[name] = po.asset
            object_quats[name] = np.array(po.quaternion, dtype=np.float32)

        # Overview camera: single source of truth is the scenario's
        # SceneConfig.camera_position / camera_target. No env overrides.
        cam_up = handle.scene.add_camera(
            res=(640, 480),
            pos=tuple(scene_config.camera_position),
            lookat=tuple(scene_config.camera_target),
            fov=45, GUI=False,
        )
        cam_wrist = handle.scene.add_camera(
            res=(640, 480),
            pos=(0.05, 0.0, -0.08),
            lookat=(0.0, 0.0, 0.10),
            fov=65, GUI=False,
        )

        handle.scene.build()

        franka = handle.franka
        motors_dof = [
            franka.get_joint(name).dofs_idx_local[0]
            for name in JOINT_NAMES
        ]
        arm_dof = motors_dof[:7]
        finger_dof = motors_dof[7:]

        franka.set_dofs_kp(KP, motors_dof)
        franka.set_dofs_kv(KV, motors_dof)
        franka.set_dofs_force_range(FORCE_LOWER, FORCE_UPPER, motors_dof)

        end_effector = franka.get_link("hand")

        # Attach wrist camera
        from genesis.utils.geom import pos_lookat_up_to_T
        wrist_offset_T = pos_lookat_up_to_T(
            torch.tensor([0.05, 0.0, -0.08],
                         dtype=gs.tc_float, device=gs.device),
            torch.tensor([0.0, 0.0, 0.10],
                         dtype=gs.tc_float, device=gs.device),
            torch.tensor([0.0, 0.0, -1.0],
                         dtype=gs.tc_float, device=gs.device),
        )
        try:
            cam_wrist.attach(rigid_link=end_effector,
                             offset_T=wrist_offset_T)
        except TypeError:
            cam_wrist.attach(end_effector, wrist_offset_T)
        logger.debug("[cam] wrist camera attached to franka hand link")

        env = cls(
            scene=handle.scene,
            franka=franka,
            scene_config=scene_config,
            entity_map=entity_map,
            placed_map=placed_map,
            object_heights=object_heights,
            asset_map=asset_map,
            object_quats=object_quats,
            cam_up=cam_up,
            cam_wrist=cam_wrist,
            end_effector=end_effector,
            motors_dof=motors_dof,
            arm_dof=arm_dof,
            finger_dof=finger_dof,
            planner=GeometricGraspPlanner(),
            table_surface_z=table_z,
        )

        # Resolve articulated joint DOFs and initial joint state (passive joints).
        env._init_articulation()

        # Settle
        franka.set_dofs_position(HOME_QPOS, motors_dof)
        franka.control_dofs_position(HOME_QPOS, motors_dof)
        env._apply_joint_init()
        for _ in range(settle_steps):
            handle.scene.step()

        env.sync_object_states()
        return env

    def trial_from_here(self, trajectory: list, target: str, *, lift_m: float = 0.02) -> dict:
        """Replay ``trajectory`` from the current scene state without rendering, report whether ``target``
        ended at least ``lift_m`` higher, then restore the whole scene (bodies, joints, velocities)."""
        snapshot = self.scene.get_state()
        qpos = to_numpy(self.franka.get_dofs_position(self.motors_dof))
        z0 = float(to_numpy(self.entity_map[target].get_pos()).reshape(-1)[2])
        t0 = time.perf_counter()
        try:
            for q in trajectory:
                self.franka.control_dofs_position(q, self.motors_dof)
                self.step()
            dz = float(to_numpy(self.entity_map[target].get_pos()).reshape(-1)[2]) - z0
        finally:
            self.scene.reset(snapshot)
            self.franka.control_dofs_position(qpos, self.motors_dof)
        return {"lifted": dz > lift_m, "lift_m": round(dz, 4), "seconds": round(time.perf_counter() - t0, 3)}

    def step(self) -> None:
        """One physics step. Real steps and ``reset`` go through here, so ``state_fingerprint`` changes
        even when they leave the same joint values; the static queries' restores do not count."""
        self.scene.step()
        self._writes += 1
        if self.on_step is not None:
            self.on_step(self)

    def contacts_now(self) -> list[tuple[tuple[str, str], ...]]:
        """Contacts the last physics step resolved that involve the arm or a named entity:
        ``[((entity, link), (entity, link)), ...]``, links outside them as ``("env", "?")``. Reads only."""
        owner = {lk.idx: (name, lk.name) for name, ent in [("robot", self.franka), *self.entity_map.items()]
                 for lk in ent.links}
        pairs = set()
        for ent in [self.franka, *self.entity_map.values()]:
            c = ent.get_contacts()
            a, b = to_numpy(c["link_a"]).reshape(-1), to_numpy(c["link_b"]).reshape(-1)
            if "valid_mask" in c:
                m = to_numpy(c["valid_mask"]).reshape(-1).astype(bool)
                a, b = a[m], b[m]
            for i, j in zip(a, b):
                pairs.add(tuple(sorted((owner.get(int(i), ("env", "?")), owner.get(int(j), ("env", "?"))))))
        return sorted(pairs)

    # ---- Static queries: each poses bodies inside a snapshot and restores it ----

    @contextmanager
    def _snapshot(self):
        snapshot = self.scene.get_state()
        try:
            yield
        finally:
            self.scene.reset(snapshot)

    def _set_arm(self, qpos) -> None:
        self.franka.set_dofs_position(np.asarray(qpos, dtype=np.float32), self.motors_dof, zero_velocity=True)

    def _hand(self) -> tuple[np.ndarray, np.ndarray]:
        return (to_numpy(self.end_effector.get_pos()).reshape(-1).astype(np.float64),
                to_numpy(self.end_effector.get_quat()).reshape(-1).astype(np.float64))

    def _geom_owner(self) -> dict[int, tuple[str, str]]:
        owner: dict[int, tuple[str, str]] = {}
        named = [("robot", self.franka), *self.entity_map.items()]
        known = {id(e) for _, e in named}
        others = [("env", e) for e in getattr(self.scene, "entities", []) if id(e) not in known]
        for name, ent in [*others, *named]:
            for geom in getattr(ent, "geoms", []):
                owner[geom.idx] = (name, geom.link.name)
        return owner

    def object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """``(pos, quat_wxyz)`` of ``name`` as ``set_pos`` / ``set_quat`` take it."""
        ent = self.entity_map[name]
        return (to_numpy(ent.get_pos()).reshape(-1).astype(np.float64),
                to_numpy(ent.get_quat()).reshape(-1).astype(np.float64))

    def hand_pose(self, qpos) -> tuple[np.ndarray, np.ndarray]:
        """Hand-link ``(pos, quat_wxyz)`` at ``qpos``; the scene is unchanged on return."""
        with self._snapshot():
            self._set_arm(qpos)
            return self._hand()

    def _put(self, name: str, t_world: np.ndarray) -> None:
        ent = self.entity_map[name]
        ent.set_pos(t_world[:3, 3].astype(np.float32), zero_velocity=True)
        ent.set_quat(rotmat_to_quat_wxyz(t_world[:3, :3]).astype(np.float32), zero_velocity=True, relative=False)

    def contacts_at(self, qpos, held: tuple[str, np.ndarray] | None = None,
                    placed: tuple[str, np.ndarray] | None = None) -> list[tuple[tuple[str, str], ...]]:
        """Contacts with the arm at ``qpos``, with ``held=(name, T_hand_obj)`` that object at
        ``hand · T_hand_obj`` and with ``placed=(name, T_world)`` that object at ``T_world``:
        ``[((entity, link), (entity, link)), ...]``, read by collision detection without stepping.
        The scene is unchanged on return."""
        owner = self._geom_owner()
        with self._snapshot():
            self._set_arm(qpos)
            if held is not None:
                name, t_hand_obj = held
                self._put(name, pose_matrix(*self._hand()) @ np.asarray(t_hand_obj, dtype=np.float64))
            if placed is not None:
                self._put(placed[0], np.asarray(placed[1], dtype=np.float64))
            pairs = self.scene.sim.rigid_solver.detect_collision()
        unknown = ("env", "?")
        return [(owner.get(int(a), unknown), owner.get(int(b), unknown)) for a, b in pairs]

    @cached_property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """``(lower, upper)`` of the arm and finger DOFs, in ``motors_dof`` order."""
        lo, hi = self.franka.get_dofs_limit(self.motors_dof)
        return (to_numpy(lo).reshape(-1).astype(np.float64), to_numpy(hi).reshape(-1).astype(np.float64))

    @cached_property
    def joined_links(self) -> frozenset:
        """Robot link-name pairs that touch by construction: parent and child, or two children of one parent."""
        name = {lk.idx: lk.name for lk in self.franka.links}
        parent = {lk.name: name.get(lk.parent_idx) for lk in self.franka.links}
        return frozenset(frozenset((a, b)) for a in parent for b in parent
                         if a != b and (parent[a] == b or parent[b] == a or parent[a] == parent[b] is not None))

    def lever_arms(self, held: tuple[str, np.ndarray] | None = None) -> np.ndarray:
        """Per DOF in ``motors_dof`` order, r such that a step dq moves no point of the robot, nor of
        ``held=(name, T_hand_obj)`` riding in the hand, further than ``r @ |dq|``. For an arm joint: the longest path
        from its anchor along link origins to a link it moves plus that link's collision-AABB radius (the held
        object counts as |T's offset| + its AABB radius from the hand origin), with a finger link's path stretched
        by its full travel; 1 for the finger DOFs. Lengths between the origins of one rigid link do not change with
        the pose and an AABB radius only overstates, so the current pose gives a bound for every pose. Reads only."""

        def radius(box, p) -> float:
            lo, hi = np.asarray(box.cpu().numpy() if hasattr(box, "cpu") else box, dtype=np.float64).reshape(-1, 2, 3)[0]
            return float(np.linalg.norm(np.maximum(np.abs(lo - p), np.abs(hi - p))))

        links = {lk.idx: lk for lk in self.franka.links}
        origin = {i: to_numpy(lk.get_pos()).reshape(-1)[:3].astype(np.float64) for i, lk in links.items()}
        children: dict[int, list[int]] = {}
        for lk in links.values():
            children.setdefault(lk.parent_idx, []).append(lk.idx)
        lo, hi = self.joint_limits
        travel = {self.franka.get_joint(n).link.idx: float(hi[k] - lo[k]) for k, n in enumerate(JOINT_NAMES) if k >= 7}
        extra = {}
        if held is not None:
            name, t = held
            pos, _ = self.object_pose(name)
            extra[self.end_effector.idx] = (float(np.linalg.norm(np.asarray(t, dtype=np.float64)[:3, 3]))
                                            + radius(self.entity_map[name].get_AABB(), pos))

        def reach(i: int) -> float:
            own = radius(links[i].get_AABB(), origin[i]) if links[i].n_geoms else 0.0
            return max([own, extra.get(i, 0.0)]
                       + [float(np.linalg.norm(origin[c] - origin[i])) + travel.get(c, 0.0) + reach(c)
                          for c in children.get(i, [])])

        out = np.ones(len(self.motors_dof))
        for k, name in enumerate(JOINT_NAMES[:7]):
            joint = self.franka.get_joint(name)
            anchor = to_numpy(joint.get_anchor_pos()).reshape(-1)[:3].astype(np.float64)
            out[k] = float(np.linalg.norm(origin[joint.link.idx] - anchor)) + reach(joint.link.idx)
        return out

    def hold(self, qpos, name: str, t_hand_obj) -> None:
        """A real write, without stepping: the arm at ``qpos`` and ``name`` at hand · ``t_hand_obj``, both at rest."""
        self._writes += 1
        self._set_arm(qpos)
        self.franka.control_dofs_position(np.asarray(qpos, dtype=np.float32), self.motors_dof)
        self._put(name, pose_matrix(*self._hand()) @ np.asarray(t_hand_obj, dtype=np.float64))

    def state_fingerprint(self) -> str:
        """Hash of the real-write count and every rigid entity's joint positions and velocities."""
        h = hashlib.sha1(str(self._writes).encode())
        for ent in [self.franka, *self.entity_map.values()]:
            if getattr(ent, "n_dofs", 0):
                h.update(to_numpy(ent.get_qpos()).astype(np.float32).tobytes())
                h.update(to_numpy(ent.get_dofs_velocity()).astype(np.float32).tobytes())
        return h.hexdigest()

    # ---- Workspace ----

    @property
    def workspace_xy(self) -> tuple[tuple[float, float], tuple[float, float]]:
        ws = self.scene_config.workspace_xy
        return (ws[0][0], ws[0][1]), (ws[1][0], ws[1][1])

    @property
    def x_range(self) -> tuple[float, float]:
        ws = self.scene_config.workspace_xy
        return (ws[0][0], ws[1][0])

    @property
    def y_range(self) -> tuple[float, float]:
        ws = self.scene_config.workspace_xy
        return (ws[0][1], ws[1][1])

    # ---- IK solver ----

    def solve_ik(self, pos, quat=None, finger_pos=None,
                 init_qpos=None) -> np.ndarray:
        if quat is None:
            raise ValueError("solve_ik requires an explicit end-effector quat")
        if finger_pos is None:
            finger_pos = self._default_finger_open
        ik_kwargs: dict = dict(
            link=self.end_effector,
            pos=np.array(pos, dtype=np.float32),
            quat=np.array(quat, dtype=np.float32),
        )
        if init_qpos is not None:
            ik_kwargs["init_qpos"] = np.array(init_qpos, dtype=np.float32)
        qpos = to_numpy(self.franka.inverse_kinematics(**ik_kwargs))
        target = np.zeros(N_DOFS, dtype=np.float32)
        target[:7] = qpos[:7]
        target[7] = finger_pos
        target[8] = finger_pos
        return target

    # ---- Articulation ----

    def _init_articulation(self) -> None:
        """Map each articulated object's joints to local DOF indices and resolve
        the initial joint state.

        Initial qpos = 0, overlaid by the per-scenario ``ObjectPlacement.joint_init``
        override.
        Joints must be passive (no position actuator): the robot opens/closes
        them through contact, then the slide must *stay where it was dragged*.

        Two passive-joint properties are set here:
        - ``kp=0``: defensive, zeroes Genesis's default position PD so no latent
          spring toward qpos0=0 can appear.
        - ``dofs_damping`` (= ``PASSIVE_JOINT_DAMPING``): viscous damping that
          bleeds the drag-release kick so the drawer doesn't coast fully closed
          and crowd the following ``place``. Pure viscous (zero force at rest),
          so slow drag-open is barely affected.

        Only the initial qpos is set on reset.
        """
        scene_joint_init = {
            o.name_override: o.joint_init
            for o in self.scene_config.objects
            if getattr(o, "joint_init", None)
        }
        for name, asset in self.asset_map.items():
            if not getattr(asset, "is_articulated", False):
                continue
            ent = self.entity_map.get(name)
            if ent is None:
                continue
            dof_map: dict[str, int] = {}
            for joint in asset.movable_joints:
                try:
                    dof_map[joint.name] = ent.get_joint(joint.name).dofs_idx_local[0]
                except Exception as exc:  # pragma: no cover - sim-specific
                    logger.warning(
                        "[articulation] %s joint %s dof lookup failed: %s",
                        name, joint.name, exc,
                    )
            if not dof_map:
                continue
            self.articulated_joint_dofs[name] = dof_map
            dofs = list(dof_map.values())
            try:
                ent.set_dofs_kp(np.zeros(len(dofs), dtype=np.float32), dofs)
                ent.set_dofs_damping(
                    np.full(len(dofs), PASSIVE_JOINT_DAMPING, dtype=np.float32), dofs
                )
            except Exception as exc:  # pragma: no cover - sim-specific
                logger.warning(
                    "[articulation] %s passivize failed: %s", name, exc
                )
            init = scene_joint_init.get(name, {}) or {}
            self.articulated_joint_init[name] = {
                j: float(init.get(j, 0.0)) for j in dof_map
            }

    def _apply_joint_init(self) -> None:
        """Set each articulated joint to its initial qpos and zero its velocity.

        Zeroing velocity is what makes a frozen joint actually frozen: setting
        only the position leaves any residual velocity (e.g. a limit-constraint
        kick from a prior settle) to coast the joint inward over the episode.
        """
        for name, dof_map in self.articulated_joint_dofs.items():
            ent = self.entity_map.get(name)
            if ent is None:
                continue
            init = self.articulated_joint_init.get(name, {})
            for joint_name, dof in dof_map.items():
                ent.set_dofs_position(
                    np.array([init.get(joint_name, 0.0)], dtype=np.float32), [dof]
                )
            ent.zero_all_dofs_velocity()

    def get_joint_positions(self) -> dict[str, dict[str, float]]:
        """Current qpos of every tracked articulated joint: name -> joint -> qpos."""
        out: dict[str, dict[str, float]] = {}
        for name, dof_map in self.articulated_joint_dofs.items():
            ent = self.entity_map.get(name)
            if ent is None:
                continue
            joints: dict[str, float] = {}
            for joint_name, dof in dof_map.items():
                q = to_numpy(ent.get_dofs_position([dof]))
                joints[joint_name] = float(np.asarray(q).reshape(-1)[0])
            out[name] = joints
        return out

    # ---- Reset ----

    def get_initial_z(self, name: str) -> float:
        """Fallback spawn z when the caller omits it: table_surface + half_height + margin.

        Only used when ``reset`` receives a 2-tuple (no resolved z). Places the
        object so its bottom just touches the table surface (plus a small 2mm
        margin to avoid interpenetration). ``object_heights`` is the
        quaternion-aware world-frame Z extent computed by
        ``PlacedObject.object_height_m``. The primary path is the resolver's
        world z carried through ``reset``; this assumes
        "object spawns on the table", which is wrong for shelved/in-drawer
        starts, so it must remain a fallback only.
        """
        h = self.object_heights.get(name, 0.04)
        return self.table_surface_z + h / 2.0 + 0.002

    def reset(
        self,
        obj_positions: dict[str, tuple[float, float] | tuple[float, float, float]],
        marker_xy: tuple[float, float] | None = None,
        target_marker=None,
        settle_steps: int = 30,
    ) -> dict[str, np.ndarray]:
        """Reset robot home + reposition objects by name → (x, y[, z]).

        ``z`` is the spawn height. When the caller supplies it (3-tuple), it is
        the resolver's already-resolved world Z (single source of truth); the
        sim must not recompute it. Only fall back to the table-relative
        ``get_initial_z`` when z is omitted (2-tuple).
        """
        self._writes += 1
        self.franka.set_dofs_position(HOME_QPOS, self.motors_dof)
        self.franka.control_dofs_position(HOME_QPOS, self.motors_dof)
        self.franka.zero_all_dofs_velocity()

        spawned = {}
        for name, pos in obj_positions.items():
            ent = self.entity_map.get(name)
            if ent is None:
                continue
            x, y = float(pos[0]), float(pos[1])
            z = float(pos[2]) if len(pos) >= 3 else self.get_initial_z(name)
            ent.set_pos(
                torch.tensor([x, y, z], dtype=torch.float32,
                             device=self._device).unsqueeze(0),
                zero_velocity=True, relative=False,
            )
            po = self.placed_map.get(name)
            q_wxyz = po.quaternion if po else [1, 0, 0, 0]
            spawned[name] = ((x, y, z), q_wxyz)
            ent.set_quat(
                torch.tensor(q_wxyz, dtype=torch.float32,
                             device=self._device).unsqueeze(0),
                zero_velocity=True, relative=False,
            )

        if target_marker is not None and marker_xy is not None:
            target_marker.set_pos(torch.tensor(
                [marker_xy[0], marker_xy[1], 0.0025],
                dtype=torch.float32,
                device=self._device).unsqueeze(0))

        self._apply_joint_init()

        for _ in range(settle_steps):
            self.step()

        positions = self.sync_object_states()
        violations = settle_violations(
            spawned, {name: (positions[name], self.object_quats[name]) for name in spawned})
        if violations:
            raise SceneInvalid("; ".join(violations))
        return positions

    def sync_object_states(self) -> dict[str, np.ndarray]:
        """Sync object pose maps from the simulator after physics settling."""
        positions: dict[str, np.ndarray] = {}
        for name, ent in self.entity_map.items():
            positions[name] = to_numpy(ent.get_pos()).astype(np.float32).copy()
            self.object_quats[name] = to_numpy(ent.get_quat()).astype(np.float32).copy()
        return positions

    @property
    def _device(self):
        """Infer Genesis torch device from franka entity."""
        try:
            import genesis as gs
            return gs.device
        except Exception:
            return "cpu"
