"""Simulator helper for asset reviewers: one asset on the ground, scenario physics.

    from probe import Probe
    p = Probe("asset")                 # asset directory with model.urdf
    p.drive_joint("drawer_slide", 0.24)
    p.step(100)
    print(p.joints(), p.link_pose("drawer"), p.box_overlaps())

Physics matches the scenario runtime: 30 fps with 8 substeps, box-box contacts, passive
joints (no position spring, the runtime's viscous damping).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robotsmith.assets import derive
from robotsmith.assets.schema import parse_urdf_collision_boxes, parse_urdf_joints
from robotsmith.sim.sim_env import PASSIVE_JOINT_DAMPING


def _quat_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class Probe:
    def __init__(self, asset_dir: str | Path, *, fixed: bool = True, pos=(0.0, 0.0, 0.0),
                 cubes: dict[str, tuple] | None = None) -> None:
        """``cubes``: free test boxes ``{name: ((sx, sy, sz), (x, y, z))}`` in metres, e.g. to
        drop onto a place target; ``mass`` 0.1 kg each."""
        import genesis as gs

        gs.init(backend=gs.gpu, logging_level="warning")
        self.urdf = Path(asset_dir) / "model.urdf"
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1.0 / 30, substeps=8),
            rigid_options=gs.options.RigidOptions(enable_collision=True, enable_joint_limit=True,
                                                  box_box_detection=True),
            show_viewer=False,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.entity = self.scene.add_entity(gs.morphs.URDF(file=str(self.urdf), fixed=fixed, pos=pos))
        self.cubes = {}
        for name, (size, at) in (cubes or {}).items():
            rho = 0.1 / float(np.prod(size))
            self.cubes[name] = self.scene.add_entity(gs.morphs.Box(size=size, pos=at),
                                                     material=gs.materials.Rigid(rho=rho))
        self.scene.build()
        self._dof = {j.name: self.entity.get_joint(j.name).dofs_idx_local[0]
                     for j in parse_urdf_joints(self.urdf) if j.is_movable}
        if self._dof:
            dofs = list(self._dof.values())
            self.entity.set_dofs_kp(np.zeros(len(dofs)), dofs)
            self.entity.set_dofs_damping(np.full(len(dofs), PASSIVE_JOINT_DAMPING), dofs)
        self._boxes = [b for b in parse_urdf_collision_boxes(self.urdf) if b.name]
        # Genesis merges fixed-joint links into their parent; keep each URDF link's
        # offset from the surviving link it was merged into.
        live = {lk.name for lk in self.entity.links}
        rest = derive.link_transforms(self.urdf)
        parent = {j.child: j.parent for j in parse_urdf_joints(self.urdf)}
        self._anchor = {}
        for link in rest:
            host = link
            while host not in live:
                host = parent[host]
            self._anchor[link] = (host, np.linalg.inv(rest[host]) @ rest[link])

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.scene.step()

    def joints(self) -> dict[str, float]:
        q = self.entity.get_dofs_position().cpu().numpy()
        return {name: float(q[i]) for name, i in self._dof.items()}

    def set_joint(self, name: str, q: float, velocity: float = 0.0) -> None:
        """Teleport a joint to ``q`` with joint velocity ``velocity`` (m/s or rad/s)."""
        dof = [self._dof[name]]
        self.entity.set_dofs_position(np.array([q]), dof)
        self.entity.set_dofs_velocity(np.array([velocity]), dof)

    def drive_joint(self, name: str, q: float, steps: int = 60) -> float:
        """Move a joint to ``q`` over ``steps`` frames under a temporary position servo (as a
        gripper dragging it would), release it, and return the reached position."""
        dof = [self._dof[name]]
        self.entity.set_dofs_kp(np.array([2000.0]), dof)
        start = self.joints()[name]
        for t in np.linspace(start, q, steps):
            self.entity.control_dofs_position(np.array([t]), dof)
            self.scene.step()
        self.entity.set_dofs_kp(np.array([0.0]), dof)
        self.entity.control_dofs_force(np.array([0.0]), dof)
        return self.joints()[name]

    def _link_tf(self, link: str) -> np.ndarray:
        host, offset = self._anchor[link]
        lk = self.entity.get_link(host)
        tf = np.eye(4)
        tf[:3, :3] = _quat_matrix(lk.get_quat().cpu().numpy())
        tf[:3, 3] = lk.get_pos().cpu().numpy()
        return tf @ offset

    def link_pose(self, link: str) -> tuple[list[float], list[float]]:
        """World position (m) and rotation matrix rows of a link."""
        tf = self._link_tf(link)
        return tf[:3, 3].round(5).tolist(), tf[:3, :3].round(5).tolist()

    def cube_pos(self, name: str) -> list[float]:
        """World position (m) of a test cube's centre."""
        return self.cubes[name].get_pos().cpu().numpy().round(5).tolist()

    def push(self, link: str, force, steps: int = 1) -> None:
        """Apply a world-frame force (N) at a link's origin for ``steps`` frames."""
        solver = self.scene.sim.rigid_solver
        idx = [self.entity.get_link(self._anchor[link][0]).idx]
        for _ in range(steps):
            solver.apply_links_external_force(force=np.asarray([force], dtype=np.float64), links_idx=idx)
            self.scene.step()

    def contacts(self) -> list[dict]:
        """The asset's current contacts as ``{link_a, link_b, force_n}``: asset link names,
        test cube names, or ``ground``."""
        c = self.entity.get_contacts()
        names = {lk.idx: lk.name for lk in self.entity.links}
        for name, cube in self.cubes.items():
            names |= {lk.idx: name for lk in cube.links}
        force = np.linalg.norm(c["force_a"].cpu().numpy(), axis=-1).tolist()
        return [{"link_a": names.get(a, "ground"), "link_b": names.get(b, "ground"), "force_n": round(f, 3)}
                for a, b, f in zip(c["link_a"].tolist(), c["link_b"].tolist(), force)]

    def box_world(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """World axis-aligned bounds of every named collision box at the live link poses."""
        out = {}
        for b in self._boxes:
            out[b.name] = derive.box_bounds(b, self._link_tf(b.link))
        return out

    def box_overlaps(self, min_depth: float = 0.0) -> list[dict]:
        """Pairs of collision boxes on different links whose bounds overlap, with the
        smallest per-axis overlap (m) as ``depth`` (a negative gap is reported as 0)."""
        boxes = self.box_world()
        link_of = {b.name: b.link for b in self._boxes}
        names = sorted(boxes)
        out = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if link_of[a] == link_of[b]:
                    continue
                (alo, ahi), (blo, bhi) = boxes[a], boxes[b]
                ov = np.minimum(ahi, bhi) - np.maximum(alo, blo)
                if (ov > 0).all() and ov.min() > min_depth:
                    out.append({"a": a, "b": b, "depth": round(float(ov.min()), 4)})
        return out
