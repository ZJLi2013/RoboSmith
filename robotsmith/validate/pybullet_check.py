"""Validate URDF assets by loading them in PyBullet and running basic physics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationResult:
    name: str
    urdf_path: str
    loaded: bool
    num_joints: int = 0
    final_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    stable: bool = False
    error: str = ""

    @property
    def passed(self) -> bool:
        return self.loaded and self.stable


def validate_urdf(
    urdf_path: str | Path,
    name: str = "",
    fixed_base: bool = False,
    sim_steps: int = 480,
    stability_threshold: float = 2.0,
) -> ValidationResult:
    """Load a URDF in PyBullet DIRECT mode and check basic physics stability."""
    import pybullet as p

    urdf_path = str(Path(urdf_path).resolve())
    name = name or Path(urdf_path).parent.name

    cid = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81, physicsClientId=cid)

        plane_id = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=cid)
        p.createMultiBody(0, plane_id, physicsClientId=cid)

        try:
            body_id = p.loadURDF(
                urdf_path,
                basePosition=[0, 0, 0.5],
                useFixedBase=int(fixed_base),
                flags=p.URDF_USE_INERTIA_FROM_FILE,
                physicsClientId=cid,
            )
        except Exception as e:
            return ValidationResult(name=name, urdf_path=urdf_path, loaded=False, error=str(e))

        if body_id < 0:
            return ValidationResult(name=name, urdf_path=urdf_path, loaded=False, error="loadURDF returned -1")

        num_joints = p.getNumJoints(body_id, physicsClientId=cid)

        for _ in range(sim_steps):
            p.stepSimulation(physicsClientId=cid)

        pos, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=cid)

        stable = abs(pos[2]) < stability_threshold and abs(pos[0]) < stability_threshold

        return ValidationResult(
            name=name,
            urdf_path=urdf_path,
            loaded=True,
            num_joints=num_joints,
            final_position=tuple(round(v, 4) for v in pos),
            stable=stable,
        )
    finally:
        p.disconnect(cid)


def validate_all_assets(assets_root: Path, fixed_base: bool = False) -> list[ValidationResult]:
    """Validate all URDF assets under assets_root/objects/."""
    results = []
    objects_dir = assets_root / "objects"
    if not objects_dir.exists():
        return results

    for asset_dir in sorted(objects_dir.iterdir()):
        if not asset_dir.is_dir():
            continue
        urdf = asset_dir / "model.urdf"
        if not urdf.exists():
            continue

        is_static = asset_dir.name in ("plane", "table_simple")
        result = validate_urdf(urdf, name=asset_dir.name, fixed_base=is_static or fixed_base)
        results.append(result)

    return results


def print_validation_report(results: list[ValidationResult]) -> None:
    """Print a formatted validation report."""
    print(f"\n{'='*60}")
    print(f"PyBullet Validation Report ({len(results)} assets)")
    print(f"{'='*60}")
    passed = 0
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        pos_str = f"pos=({r.final_position[0]:.3f}, {r.final_position[1]:.3f}, {r.final_position[2]:.3f})"
        err = f" [{r.error}]" if r.error else ""
        print(f"  {status}  {r.name:20s}  loaded={r.loaded}  joints={r.num_joints}  {pos_str}{err}")
        if r.passed:
            passed += 1
    print(f"\n  {passed}/{len(results)} passed")
    print(f"{'='*60}\n")
