"""Genesis simulation environment for data collection.

Encapsulates scene loading, Franka setup, cameras, IK,
and reset logic. No episode/recording concerns.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from robotsmith.sim.franka import (
    JOINT_NAMES, N_DOFS, HOME_QPOS, KP, KV,
    FORCE_LOWER, FORCE_UPPER, to_numpy,
)
from robotsmith.assets.schema import Asset
from robotsmith.grasp.planner import GraspPlanner, resolve_grasp_strategy
from robotsmith.scenes.config import SceneConfig

logger = logging.getLogger(__name__)


TARGET_MARKER_SIZE = (0.06, 0.06, 0.005)

# Viscous damping on passive articulated DOFs (e.g. drawer slide) so the
# drag-release velocity kick bleeds off instead of coasting the joint fully
# closed. Genesis-0.4.5 default (0.3) is far too low on MI300; 20.0 holds >0.1
# open even under a -2 m/s kick.
PASSIVE_JOINT_DAMPING = 20.0


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
    object_scales: dict[str, float] = field(default_factory=dict)
    planner: Optional[GraspPlanner] = None
    table_surface_z: float = 0.0

    articulated_joint_dofs: dict[str, dict[str, int]] = field(default_factory=dict)
    """object name -> {joint name -> local DOF index} for articulated assets."""
    articulated_joint_init: dict[str, dict[str, float]] = field(default_factory=dict)
    """object name -> {joint name -> initial qpos} applied (passively) on reset."""

    _default_finger_open: float = 0.04

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
        grasp_planner: str = "auto",
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
        object_scales = {po.name: po.metric_scale for po in handle.placed}

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

        planner: GraspPlanner
        if grasp_planner not in {"auto", "learned"}:
            raise ValueError(
                "grasp_planner must be one of: auto, learned"
            )

        # Only assets that actually resolve to a learned grasp need a mesh.
        # Articulated assets resolve to "none" (moved by drag_handle, mesh from
        # the URDF). This keeps the planner asset-aware instead of forcing a
        # scene-wide mesh requirement.
        missing_mesh = [
            name for name, asset in asset_map.items()
            if resolve_grasp_strategy(asset, default="learned") == "learned"
            and asset.visual_mesh is None and asset.collision_mesh is None
        ]
        if missing_mesh:
            raise RuntimeError(
                "LearnedGraspPlanner requires a mesh for every learned-grasp object; "
                f"missing visual/collision mesh for {missing_mesh}"
            )
        try:
            from robotsmith.grasp.graspgen_wrapper import GraspGenModel
            from robotsmith.grasp.learned_planner import LearnedGraspPlanner
            cfg_path = os.environ.get("GRASPGEN_CONFIG", None)
            model = GraspGenModel(config_yaml=cfg_path)
            max_az = float(os.environ.get("MAX_APPROACH_Z", "1.0"))
            top_k = int(os.environ.get("GRASP_TOP_K", "100"))
            fixed_ori = os.environ.get("GRASP_FIXED_ORI", "0") == "1"
            planner = LearnedGraspPlanner(
                model, z_offset=table_z, max_approach_z=max_az,
                fixed_orientation=fixed_ori, top_k=top_k,
            )
            logger.debug("[grasp] using LearnedGraspPlanner")
        except Exception as exc:
            raise RuntimeError(
                "LearnedGraspPlanner could not be initialized: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        env = cls(
            scene=handle.scene,
            franka=franka,
            scene_config=scene_config,
            entity_map=entity_map,
            placed_map=placed_map,
            object_heights=object_heights,
            asset_map=asset_map,
            object_quats=object_quats,
            object_scales=object_scales,
            cam_up=cam_up,
            cam_wrist=cam_wrist,
            end_effector=end_effector,
            motors_dof=motors_dof,
            arm_dof=arm_dof,
            finger_dof=finger_dof,
            planner=planner,
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

        Initial qpos = asset metadata default, overlaid by the per-scenario
        ``ObjectPlacement.joint_init`` override (scene override > asset default).
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
            init = dict(getattr(asset.metadata, "joint_init", {}) or {})
            init.update(scene_joint_init.get(name, {}) or {})
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
        world z carried through ``reset`` (docs/design.md §5.1); this assumes
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
        ``get_initial_z`` when z is omitted (2-tuple). See docs/design.md §5.1.
        """
        self.franka.set_dofs_position(HOME_QPOS, self.motors_dof)
        self.franka.control_dofs_position(HOME_QPOS, self.motors_dof)
        self.franka.zero_all_dofs_velocity()

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
            self.scene.step()

        return self.sync_object_states()

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
