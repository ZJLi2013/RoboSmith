# RoboSmith

[中文 README](README.md)

**Synthetic Data Generation for Physical AI / Robotics on AMD ROCm.**

RoboSmith is a **manipulation data pipeline**: you define a task in code, and it builds the scene, plans grasps, executes collision-aware expert trajectories, and exports a ready-to-train LeRobot dataset + video.

It is the **SDG stage** of the ROCm Physical AI toolchain, wired to two sibling repos:

```text
RocRecon   (real2sim: prompt/image -> sim-ready URDF assets)
   │  produces assets
   ▼
RoboSmith  (SDG: assets + tasks -> expert trajectories -> LeRobot data)   <- this repo
   │  calls the motion backend
   ▼
rocRobo    (collision-free IK + avoidance trajopt, motion solving)
```

<p align="center">
  <img src="images/scene_overview.png" width="600" alt="Franka + collision-aware random layout (Genesis, MI300X)">
</p>

## What it generates

Describe "what's on the table, which steps to run, what counts as success" in a bit of Python, and RoboSmith produces one continuous episode with video + a LeRobot dataset + success/failure diagnostics:

- **Rigid pick-and-place**: given objects and a target, it auto-runs "grasp -> transport -> place". Grasp points come from a learned grasp model and are filtered by feasibility — no hand-authored trajectories.
- **Long-horizon multi-step tasks**: chain several actions into one episode, e.g. "open drawer -> pick object -> place into drawer -> close drawer" — the arm avoids the cabinet, switches to controlled straight-line contact when close, and re-anchors to each object's live pose at every step. See [`scenarios/pick_place_into_drawer.py`](scenarios/pick_place_into_drawer.py).
- **Declarative task definition**: a small API to write tasks (success conditions support and/or/not composition) without touching the underlying IK or trajectory generation.

## Quick start

```bash
pip install -e .

# Long-horizon task (open/close-drawer e2e; canonical launcher with rocRobo avoidance sidecar)
bash scripts/setup/run_scenario.sh

# Inner entry: a single scenario
python scripts/datagen/run_generated_scenario.py \
  --scenario scenarios/pick_place_into_drawer.py --smoke

# Static layout render (no rollout)
python scripts/datagen/snapshot_scenario.py \
  --scenario scenarios/pick_place_into_drawer.py
```

How to author a new scenario: see [`scenarios/README.md`](scenarios/README.md).

### Runtime prerequisites

Data generation depends on three **out-of-repo** runtimes (each installed/licensed separately, not distributed here):

- **ROCm runtime image**: Genesis physics sim; pick the image by GPU arch, see [`docker/README.md`](docker/README.md) (MI300/gfx942 uses `docker/Dockerfile.gfx942`).
- **rocRobo motion backend**: collision-free IK + avoidance trajopt, from [rocRobo](https://github.com/ZJLi2013/rocRobo), run as a sidecar (license-friendly, PyRoki/JAX).
- **Learned grasp runtime**: grasp candidate model + weights (**third-party licensed, bring your own, not included in this repo**).

## Repository layout

```text
robotsmith/
├── assets/            # Asset / AssetMetadata / AssetLibrary (exact-id), builtin assets + catalog
├── cap/               # declarative task definition layer (scene/task/success/skill intent)
├── scenes/            # SceneConfig / ObjectPlacement, upright pose, Genesis loader
├── tasks/             # TaskSpec, predicates, task presets
├── scenario_runtime/  # scenario load / materialize / segment-level closed-loop runner
├── grasp/             # grasp planner (learned) + feasibility rerank
├── motion/            # motion planning + rocRobo avoidance backend integration
├── execution/         # executor: expand skills into joint trajectories
├── skills/            # skill primitives (pick/place/open/close) + registry
├── sim/               # Genesis SimEnv + Franka + LeRobot recorder
└── rollout/           # rollout orchestration, episode layout, video export
```

## Acknowledgements

- [LeRobot](https://github.com/huggingface/lerobot) — dataset format and VLA training infrastructure
- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) — GPU-accelerated physics simulator
- [rocRobo](https://github.com/ZJLi2013/rocRobo) — motion solving (collision-free IK + avoidance trajopt)
- [RocRecon](https://github.com/ZJLi2013/RocRecon) — real2sim asset generation (prompt/image -> sim-ready URDF)
- [Objaverse](https://objaverse.allenai.org/) — 3D object dataset

## License

Apache-2.0, see [LICENSE](LICENSE).
