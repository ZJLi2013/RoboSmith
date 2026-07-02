# RoboSmith

[中文](README.md) | English

**Synthetic Data Generation for Physical AI / Robotics on AMD ROCm.**

RoboSmith is **embodied Data Infrastructure (Data Infra, not a VLA training framework)** for interactive manipulation: you declare a long-horizon, articulated task in a few lines of code, and it builds the scene, plans grasps, executes collision-aware expert trajectories, and exports a ready-to-train LeRobot dataset + video. It **stands on the shoulders of the open-source community** (Genesis simulation, LeRobot dataset format, Articraft articulated assets, PyRoki kinematics, …), wiring them together on AMD ROCm — turning one real object on a single MI300 into long-horizon, interactive manipulation data.

It is one of **three already-landed pillars** of a ROCm-native robotics platform (the SDG stage):

| Capability pillar | Scope | Engine | Status |
| --- | --- | --- | --- |
| Motion / dynamics | collision-free IK + avoidance trajopt + segment routing | **rocRobo** | ✅ landed |
| SDG synthetic data | assets → scene → expert trajectories → LeRobot export | **RoboSmith** | ✅ landed |
| real2sim | real/generated object → simulatable asset | **rocRecon** | ✅ landed |

<table align="center">
  <tr>
    <td width="50%">
      <video src="https://github.com/ZJLi2013/RoboSmith/raw/main/videos/pick_place_into_drawer_ep000_up.mp4" controls muted loop width="100%"></video>
    </td>
    <td width="50%">
      <video src="https://github.com/ZJLi2013/RoboSmith/raw/main/videos/pick_place_onto_supporter_ep000_up.mp4" controls muted loop width="100%"></video>
    </td>
  </tr>
  <tr>
    <td align="center"><b>Drawer · long-horizon articulated</b><br>open → pick → place → close</td>
    <td align="center"><b>Two-tier supporter · side-insertion</b><br>die taken from the upper shelf, inserted horizontally into the lower shelf (a top-down descent would be blocked)</td>
  </tr>
</table>

## Why interaction data

The bottleneck for embodied AI is not pixels — it's **interaction**: contact-rich, multi-step, articulated behavior data, exactly what visual-quality-first synthetic pipelines struggle to cover.

- **Simple grasping** (pick up a block) is cheap but low-information: single-step, rigid-body only, the robot barely touches its environment.
- **Long-horizon interaction** (open a drawer → put something inside → close it) is the scarce, genuinely valuable tier: multi-step, articulated, collision avoidance required. Collecting this on real robots is slow, expensive, and unsafe, and you can't scrape "1,000 demos of opening a drawer and placing an object" off the internet.

The upside of SDG is direct: manufacture demos in simulation, with physics / randomization / labels fully controllable and free. What RoboSmith additionally fills is **a gap within the ROCm ecosystem** — running the whole chain end to end, with open-source components, on AMD.

## What it generates (asset × action × predicate)

Describe "what's on the table, which steps to run, what counts as success" in a bit of Python, and RoboSmith produces one continuous episode with video + a LeRobot dataset + success/failure diagnostics. As a synthetic-data generator, it essentially "given an asset's affordance, automatically unfolds the generatable (task, action, predicate)". Current coverage — two classes of **manipulable** assets:

| Asset class | affordance (unlock condition) | semantic action → underlying primitive | success predicate |
| --- | --- | --- | --- |
| **rigid** | graspable surface (mesh/bbox + upright + metric_scale) | pick / place → learned grasp (GraspGen) | object_in_container / object_above / stacked / objects_aligned |
| **articulated** | task_joints + handles (thin semantic annotation) | open / close → drag_handle (prismatic straight line, coaxial reversal) | joint_opened / joint_closed |

- **Declarative task definition (Code-as-Policy)**: a small API to write tasks (success conditions support and/or/not composition); declare "what to do", the engine handles "how" — no touching the underlying IK or trajectory generation.
- **Segment-level motion routing**: free segments (approach/transport) go through rocRobo collision-free planning; contact segments (drag handle / descend / side-insert) use controlled straight lines; grasp/release move fingers only — whether to avoid obstacles depends on whether the segment is a contact segment.
- **Stackable asset sources**: built-in **Objaverse** (current default) + **rocRecon** real2sim reconstruction (text/image/real object) + **Articraft** articulated asset generation (image + text).

Example scenario: [`scenarios/pick_place_into_drawer.py`](scenarios/pick_place_into_drawer.py); authoring guide: [`scenarios/README.md`](scenarios/README.md).

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

### Runtime prerequisites

Data generation depends on three **out-of-repo** runtimes (each installed/licensed separately, not distributed here):

- **ROCm runtime image**: Genesis physics sim; pick the image by GPU arch, see [`docker/README.md`](docker/README.md) (MI300/gfx942 uses `docker/Dockerfile.gfx942`).
- **rocRobo motion backend**: collision-free IK + avoidance trajopt, from [rocRobo](https://github.com/ZJLi2013/rocRobo), run as a sidecar (license-friendly, PyRoki/JAX).
- **Learned grasp runtime (GraspGen)**: grasp candidate model + weights (**NVIDIA research/eval license, bring your own, not included in this repo**).

> Note: JAX (rocRobo) and PyTorch (Genesis/GraspGen) do not coexist cleanly in a single ROCm process, so the motion backend runs as a separate container sidecar, driven across the boundary via `docker exec`.

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

## Roadmap and limitations

The roadmap is **four vertical capability axes + one horizontal platform axis**: morphology/embodiment (single-arm → dual-arm → humanoid), asset physics (rigid → **articulated (where we are)** → deformable → cable), long-horizon orchestration, closed-loop evaluation (highest-priority yet to build), plus converging the sim×motion bridge and rollout into stable APIs.

Current limitations:

1. Every `pick` depends on **GraspGen** (NVIDIA research/eval license, non-commercial): pulled from upstream at build time and baked into a *local* image, not redistributed. The grasp model is isolated behind a single wrapper, so swapping in an open-source / commercially-usable grasper is clean.
2. **pick is currently collision-blind**: stable on an empty table, but once there's a static obstacle beside the object (partition/cabinet), the free segment cuts straight through — making the free segment of pick also avoid obstacles is the highest-priority hole on the roadmap.
3. **Side-insertion under real configurations is not yet fully reliable**: it succeeds in isolated scenes, but during a real upper-shelf pick → lower-shelf place the insertion segment can still fail to plan and degrade to a straight-line fallback.

## References

- 📝 Blog post: [RoboSmith — A ROCm-native synthetic-data pipeline for embodied interactive manipulation](https://andyluo7.github.io/rocm/amd/mi300x/robotics/embodiedai/sdg/2026/07/01/robosmith-rocm-native-synthetic-data-pipeline/)
- [rocRobo](https://github.com/ZJLi2013/rocRobo) — motion solving (collision-free IK + avoidance trajopt)
- [RocRecon](https://github.com/ZJLi2013/RocRecon) — real2sim asset generation (prompt/image → sim-ready URDF)
- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) — GPU-accelerated physics simulator
- [LeRobot](https://github.com/huggingface/lerobot) — dataset format and VLA training infrastructure
- [Articraft](https://github.com/mattzh72/articraft) — image + text → articulated assets
- [PyRoki](https://github.com/chungmin99/pyroki) — kinematics (rocRobo foundation)
- [Objaverse](https://objaverse.allenai.org/) — 3D object dataset
- [GraspGen](https://github.com/NVlabs/GraspGen) — learned grasping (NVlabs · research/eval license)

## License

Apache-2.0, see [LICENSE](LICENSE).
