"""Whether the arm's joint PD reaches a target a few millimetres away without overshooting, which is what keeps
``SimEnv.follow``'s 5 mm bound inside a physics step and not only after it.

  python scripts/acceptance/tracking_envelope.py --out exps/r1/evidence/e18/envelope.json

Reads model quantities only, nothing is executed: at hand poses drawn by ``default_rng(0)`` (the position uniform
over ``DEFAULT_WORKSPACE_XY`` and 0.05-0.35 m above the table, pointing down at a uniform yaw, IK from home with
the base joint aimed at the position) the mass matrix and the gravity torque ``sum_l J_l^T m_l g`` of the arm, each
pose once empty-handed and once holding ``die_01`` (its mass at the held object's lever arm). Three conditions, each
must hold at every pose:

- overdamped: ``KV_j / (2 sqrt(KP_j M_jj)) >= 1`` for every arm joint;
- unsaturated: ``KP_j * 1.5 e / L_j + |tau_g,j|`` (the largest error ``follow`` allows, all on one joint) within
  ``FORCE_UPPER_j``;
- sag: ``L @ |tau_g / KP|`` (the steady error gravity leaves) below ``e / 2``, else a point is never reached.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

N_POSES = 20
HAND_ABOVE_TABLE_M = (0.05, 0.35)
DIE_XY = (0.50, 0.00)
GRAVITY = np.array([0.0, 0.0, 9.81])


def _pose_q(env, backend, rng) -> np.ndarray:
    from robotsmith.grasp.transforms import TOP_DOWN_QUAT, quat_wxyz_to_matrix, rotmat_to_quat_wxyz
    from robotsmith.scenes.config import DEFAULT_WORKSPACE_XY
    from robotsmith.sim.franka import HOME_QPOS

    (x0, y0), (x1, y1) = DEFAULT_WORKSPACE_XY
    (z0, z1), table = HAND_ABOVE_TABLE_M, env.table_surface_z
    x, y, z, yaw = rng.uniform([x0, y0, table + z0, -np.pi], [x1, y1, table + z1, np.pi])
    c, s = np.cos(yaw), np.sin(yaw)
    quat = rotmat_to_quat_wxyz(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                               @ quat_wxyz_to_matrix(np.asarray(TOP_DOWN_QUAT, dtype=np.float64)))
    init = np.asarray(HOME_QPOS, dtype=np.float64).copy()
    init[0] = float(np.arctan2(y, x))
    return np.asarray(backend.solve_ik(np.array([x, y, z]), quat, float(HOME_QPOS[7]), init_qpos=init),
                      dtype=np.float64)


def _model(env, q):
    """Mass-matrix diagonal of the arm joints at ``q`` (the larger of Genesis's and the bound
    ``sum_l m_l |J_v|^2 + lambda_max(I_l) |J_w|^2`` plus armature, since ``get_mass_mat`` may hold the last step's
    matrix), and their gravity torque."""
    franka, n = env.franka, env.franka.n_dofs
    with env.posed(q, {}):
        mass = _matrix(franka.get_mass_mat(), (n, n))
        bound, tau = np.zeros(n), np.zeros(n)
        for link in franka.links:
            m = float(_matrix(link.get_mass(), (1,))[0])
            if m <= 0.0:
                continue
            jac = _matrix(franka.get_jacobian(link), (6, n))
            inertia = float(np.linalg.eigvalsh(_matrix(link.inertial_i, (3, 3))).max())
            bound += m * (jac[:3] ** 2).sum(axis=0) + inertia * (jac[3:] ** 2).sum(axis=0)
            tau += jac[:3].T @ (m * GRAVITY)
    if hasattr(franka, "get_dofs_armature"):
        bound += _matrix(franka.get_dofs_armature(), (n,))
    dofs = np.asarray(env.motors_dof)[:7]
    return np.maximum(np.diag(mass)[dofs], bound[dofs]), tau[dofs], float(np.abs(np.diag(mass) - bound)[dofs].max())


def envelope(e_m: float) -> dict:
    from robotsmith.motion.planner import GenesisBackend
    from robotsmith.scenes.config import ObjectPlacement, SceneConfig
    from robotsmith.sim.franka import FORCE_UPPER, KP, KV, to_numpy
    from robotsmith.sim.sim_env import SimEnv

    x, y = DIE_XY
    env = SimEnv.build(SceneConfig(name="tracking_envelope", objects=[
        ObjectPlacement("die_01", position_range=[[x, y, 0.0], [x, y, 0.0]], name_override="die")]), seed=0)
    backend, rng = GenesisBackend(env.solve_ik), np.random.default_rng(0)
    m_die = float(to_numpy(env.entity_map["die"].get_mass()).reshape(-1)[0])
    t_hand_die = np.eye(4)
    t_hand_die[2, 3] = 0.1
    lever = {"empty": env.lever_arms()[:7], "held": env.lever_arms(("die", t_hand_die))[:7]}
    kp, kv, tau_max = (np.asarray(a, dtype=np.float64)[:7] for a in (KP, KV, FORCE_UPPER))
    rows, worst = [], {"zeta_min": np.inf, "saturation_max": 0.0, "sag_max_m": 0.0}
    for k in range(N_POSES):
        q = _pose_q(env, backend, rng)
        m_jj, tau_g, mass_gap = _model(env, q)
        for case, lv in lever.items():
            extra = m_die * lv**2 if case == "held" else 0.0
            tau = np.abs(tau_g) + (m_die * 9.81 * lv if case == "held" else 0.0)
            zeta = kv / (2.0 * np.sqrt(kp * (m_jj + extra)))
            saturation = (kp * 1.5 * e_m / lv + tau) / tau_max
            sag = float(lv @ (tau / kp))
            rows.append({"pose": k, "case": case, "q": np.round(q, 5).tolist(),
                         "zeta": np.round(zeta, 3).tolist(), "saturation": np.round(saturation, 3).tolist(),
                         "sag_m": round(sag, 6), "tau_g": np.round(tau_g, 3).tolist(),
                         "mass_gap": round(mass_gap, 5)})
            worst["zeta_min"] = min(worst["zeta_min"], float(zeta.min()))
            worst["saturation_max"] = max(worst["saturation_max"], float(saturation.max()))
            worst["sag_max_m"] = max(worst["sag_max_m"], sag)
    worst = {k: round(v, 6) for k, v in worst.items()}
    return {"e_m": e_m, "poses": {"empty": N_POSES, "held": N_POSES}, "rng_seed": 0, "die_mass_kg": m_die, **worst,
            "holds": {"overdamped": worst["zeta_min"] >= 1.0, "unsaturated": worst["saturation_max"] <= 1.0,
                      "sag_below_half_e": worst["sag_max_m"] < e_m / 2}, "rows": rows}


def main() -> None:
    from robotsmith.motion.joint_planner import STEP_M

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = envelope(STEP_M)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("zeta_min", "saturation_max", "sag_max_m", "holds")}))


if __name__ == "__main__":
    main()
