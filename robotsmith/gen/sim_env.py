"""Genesis simulation environment for data collection.

Encapsulates scene loading, Franka setup, cameras, IK,
and reset logic. No episode/recording concerns.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from robotsmith.gen.franka import (
    JOINT_NAMES, N_DOFS, HOME_QPOS, KP, KV,
    FORCE_LOWER, FORCE_UPPER, to_numpy,
)
from robotsmith.grasp import TemplateGraspPlanner
from robotsmith.scenes.config import SceneConfig


TARGET_MARKER_SIZE = (0.06, 0.06, 0.005)


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
        print(f"[display] Xvfb started (PID={proc.pid})")


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
    planner: Optional[TemplateGraspPlanner] = None
    table_surface_z: float = 0.0

    _default_quat: Optional[np.ndarray] = None
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
        use_videos: bool = True,
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
        print(f"[scene] {resolved.summary()}")

        handle = load_resolved_scene(
            resolved, gs_module=gs, fps=fps,
            box_box_detection=box_box_detection,
        )

        table_z = (scene_config.table_height
                   + scene_config.table_size[2] / 2.0)

        entity_map: dict[str, object] = {}
        placed_map: dict = {}
        object_heights: dict[str, float] = {}
        for name, entity, po in zip(
            handle.object_names, handle.objects, handle.placed
        ):
            entity_map[name] = entity
            placed_map[name] = po
            object_heights[name] = po.object_height_m

        # Cameras
        cam_up = handle.scene.add_camera(
            res=(640, 480),
            pos=(0.55, 0.55, table_z + 0.55),
            lookat=(0.55, 0.0, table_z + 0.10),
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
        print("[cam] wrist camera attached to franka hand link")

        planner = TemplateGraspPlanner(z_offset=table_z)
        default_template = planner._templates.get("block") or next(
            iter(planner._templates.values()))

        env = cls(
            scene=handle.scene,
            franka=franka,
            scene_config=scene_config,
            entity_map=entity_map,
            placed_map=placed_map,
            object_heights=object_heights,
            cam_up=cam_up,
            cam_wrist=cam_wrist,
            end_effector=end_effector,
            motors_dof=motors_dof,
            arm_dof=arm_dof,
            finger_dof=finger_dof,
            planner=planner,
            table_surface_z=table_z,
            _default_quat=default_template.ee_quat,
            _default_finger_open=default_template.finger_open,
        )

        # Settle
        franka.set_dofs_position(HOME_QPOS, motors_dof)
        franka.control_dofs_position(HOME_QPOS, motors_dof)
        for _ in range(settle_steps):
            handle.scene.step()

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

    def solve_ik(self, pos, quat=None, finger_pos=None) -> np.ndarray:
        if quat is None:
            quat = self._default_quat
        if finger_pos is None:
            finger_pos = self._default_finger_open
        qpos = to_numpy(self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=np.array(pos, dtype=np.float32),
            quat=np.array(quat, dtype=np.float32),
        ))
        target = np.zeros(N_DOFS, dtype=np.float32)
        target[:7] = qpos[:7]
        target[7] = finger_pos
        target[8] = finger_pos
        return target

    # ---- Reset ----

    def get_initial_z(self, name: str) -> float:
        """Spawn z for reset: table_surface + half_height + margin.

        Places the object so its bottom just touches the table surface
        (plus a small 2mm margin to avoid interpenetration).
        ``object_heights`` is the quaternion-aware world-frame Z extent
        computed by ``PlacedObject.object_height_m``.
        """
        h = self.object_heights.get(name, 0.04)
        return self.table_surface_z + h / 2.0 + 0.002

    def reset(
        self,
        obj_positions: dict[str, tuple[float, float]],
        marker_xy: tuple[float, float] | None = None,
        target_marker=None,
        settle_steps: int = 30,
    ):
        """Reset robot home + reposition objects by name → (x, y)."""
        self.franka.set_dofs_position(HOME_QPOS, self.motors_dof)
        self.franka.control_dofs_position(HOME_QPOS, self.motors_dof)
        self.franka.zero_all_dofs_velocity()

        for name, (x, y) in obj_positions.items():
            ent = self.entity_map.get(name)
            if ent is None:
                continue
            z = self.get_initial_z(name)
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

        for _ in range(settle_steps):
            self.scene.step()

    @property
    def _device(self):
        """Infer Genesis torch device from franka entity."""
        try:
            import genesis as gs
            return gs.device
        except Exception:
            return "cpu"
