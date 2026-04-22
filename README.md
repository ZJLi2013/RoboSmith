# RoboSmith

**Embodied data infrastructure**: from 3D assets to robot manipulation data — all on AMD ROCm.

> RoboSmith is a data infrastructure, not a VLA training framework. You define a task, we generate assets and produce expert data. You bring the VLA.

## Why RoboSmith

Training VLA models requires diverse, high-quality manipulation data at scale.
Today's pipelines are fragmented: asset creation, sim setup, and data collection live in separate tools with incompatible formats.
RoboSmith unifies them — a single `TaskSpec` drives asset selection and data production. Eval is handled by [vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness) with RoboSmith's benchmark plugin.

```
Part 1 ── Sim-Ready Assets          Part 2 ── Data Engine + Eval
                                    
 Text/Image → TRELLIS.2-4B           TaskSpec → Scene → Genesis
 → URDF + PBR + collision           
 26 curated + on-demand gen           Data Production Backends:
                                       ├─ IK scripted (default)
                                       ├─ DART (IK + noise)
                                       ├─ DAgger (policy + IK)
                                       └─ Online RL (planned)
                                      → LeRobot dataset
                                    
                                      Eval: vla-eval benchmark plugin
                                       └─ RoboSmithBenchmark (Genesis scene → vla-eval)
                                    
         │                                     │
         └─────────────── TaskSpec ─────────────┘
                     (composable predicates — define once, use everywhere)
```

<p align="center">
  <img src="images/scene_overview.png" width="600" alt="tabletop_simple scene — Franka + random objects">
  <br>
  <em>tabletop_simple: Franka on table + collision-aware random layout (Genesis, MI300X)</em>
</p>

## Current Capabilities

### Part 1: Sim-Ready Asset Pipeline — done

- **26 curated assets** across 10 categories (mug, bowl, block, can, bottle, fruit, figurine, plate, L-block, box), maximizing geometric diversity
- **On-demand generation**: text/image query → [TRELLIS.2-4B](https://github.com/ZJLi2013/TRELLIS.2/tree/rocm) (default, ROCm) or [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) (fallback) → URDF + PBR GLB + collision convex hull
- Quality presets: `--quality fast` (512px, RL batching) / `balanced` (1K, default) / `high` (4K, paper figures)
- Collision-aware scene layout with stable pose sampling

### Part 2: Data Engine — done (IK backend)

TaskSpec-driven data production with pluggable backends. IK scripted backend validated on MI300X:

| Task | Strategy | Success Rate | Frames/ep | Description |
|------|----------|:---:|:---:|------|
| `pick_cube` | `pick` | **100%** (20/20) | 135 | reach → grasp → lift |
| `place_cube` | `pick_and_place` | **100%** (20/20) | 225 | + transport → place → release |
| `stack_blocks` | `stack` (N=3) | **90%** (18/20) | 675 | 3 rounds of pick-and-place |

**Data production backends** (pluggable via `DataBackend` ABC):

| Backend | Method | Use Case | Status |
|---------|--------|----------|:---:|
| **IK scripted** | Open-loop IK waypoints | Initial dataset, seed data | Done |
| **DART** | IK + noise injection + IK re-solve | Recovery supervision, robustness | Planned |
| **DAgger** | Policy rollout + IK expert relabel | Close distribution gap | Planned |
| **Online RL** | Policy exploration + reward signal | Fine-grained manipulation | Planned |

Output: [LeRobot](https://github.com/huggingface/lerobot) v3.0 datasets, directly compatible with VLA training (SmolVLA verified).

### Eval: vla-eval Benchmark Plugin

RoboSmith provides `RoboSmithBenchmark` — a [vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness) `Benchmark` plugin that turns Genesis scenes into eval environments:

- 10+ VLA models auto-available via vla-eval (Pi0, StarVLA, OpenVLA, GR00T...)
- Same TaskSpec + predicates as data collection
- Action: 7D EE delta, Observation: 8D EE state + overhead + wrist cameras

## Quick Start

```bash
pip install -e .

# Browse built-in assets
robotsmith list
robotsmith search "cup"

# Import Objaverse assets (10 categories, 24 variants, ~50 MB)
pip install objaverse
python scripts/part1/import_objaverse.py

# Generate new assets (GPU required, >=24 GB VRAM)
robotsmith generate "red ceramic mug" --image reference.png
robotsmith generate "red ceramic mug"   # no image: auto T2I → 3D

# Collect manipulation data
python scripts/part2/collect_data.py --task pick_cube --n-episodes 100 --save
python scripts/part2/collect_data.py --task stack_blocks --n-episodes 20 --save

# Visualize
pip install -e ".[viz]"
robotsmith view tabletop_simple           # 3D scene preview (viser, browser)
python scripts/part1/browse_assets.py     # HTML asset gallery
```

## Roadmap

| Phase | Focus | Status |
|:---:|-------|:---:|
| 1 | Sim-ready assets (26 objects, 10 categories, TRELLIS.2-4B) | Done |
| 2 | Multi-task IK data (pick/place/stack) + vla-eval benchmark plugin | Done |
| **3** | **Irregular object grasping — Grasp Affordance Layer** (see below) | **Next** |
| 4 | DART data backend (`--dart-sigma`) | Planned |
| 5 | DAgger data backend (policy rollout + IK relabel) | Planned |
| 6 | Articulated + push/slide tasks (drawer, push-to-target) | Planned |

### Phase 3: Irregular Object Grasping (Next Step)

**Problem:** Part 1 generates diverse assets (bowls, mugs, bottles, screwdrivers, toys...), but Part 2 can only manipulate cubes/blocks — all IK grasp parameters (orientation, height, finger width) are hardcoded for small boxes. The objects that *most need* RoboSmith's data pipeline are exactly the ones it cannot handle yet.

**Why it matters:** Without this, RoboSmith remains a demo for primitive shapes. Solving this is the single highest-leverage step to make the project practically useful.

**Approach (progressive):**

| Step | What | Effort |
|------|------|:---:|
| 3a | Per-category `GraspTemplate` — one-time human annotation per category (approach direction, grasp offset, EE orientation, finger width) replacing hardcoded `TrajectoryParams` | Medium |
| 3b | Template-driven `IKStrategy` — read grasp params from `GraspTemplate` matched by asset category, enabling IK data gen for bowls, mugs, bottles, etc. | Medium |
| 3c | Auto grasp prediction — integrate GraspNet/AnyGrasp to predict grasp poses from mesh, removing per-category annotation | Long-term |

See [docs/design.md — Grasp Affordance Gap](docs/design.md#next-stepgrasp-affordance-gap-与演进路线) for detailed technical analysis.

## Project Structure

```
robotsmith/                      # Python package (pip install -e .)
├── assets/                      #   Part 1: asset management (schema, library, search, builtins)
├── gen/                         #   Part 1: 3D generation (TRELLIS.2 / Hunyuan3D backends, mesh_to_urdf)
├── scenes/                      #   Part 2: scene building (SceneConfig, genesis_loader, presets)
├── tasks/                       #   Part 2: task definition (TaskSpec, predicates, IK strategies)
├── eval/                        #   vla-eval benchmark plugin (RoboSmithBenchmark)
├── validate/                    #   physics validation (PyBullet)
├── viz/                         #   visualization (Viser asset browser + scene viewer)
└── cli.py                       #   CLI entry point (robotsmith list/search/generate/view/...)

scripts/
├── part1/                       # Asset generation & management
│   ├── import_objaverse.py      #   import Objaverse curated assets
│   ├── compute_stable_poses.py  #   compute stable placement poses
│   ├── browse_assets.py         #   generate HTML asset gallery
│   └── render_mesh_local.py     #   local mesh → PNG rendering
├── part2/                       # Data collection + eval
│   ├── collect_data.py          #   IK data collection (+ --dart-sigma DART augmentation)
│   ├── snapshot_scene.py        #   scene layout screenshot
│   ├── train_smolvla.py         #   SmolVLA fine-tune (data quality check)
│   └── test_benchmark.py        #   vla-eval benchmark smoke test
└── tools/                       # Utilities
    ├── sync_assets.py           #   remote GPU node asset sync
    └── patch_genesis_rocm.py    #   Genesis ROCm compatibility patch

configs/
└── eval/                        # vla-eval benchmark configs
    ├── robotsmith_pick_cube.yaml
    └── robotsmith_smoke_test.yaml

assets/                          # Asset storage (objects/ + generated/ + catalog.json)
docs/                            # Documentation
tests/                           # Tests
```

## Dependencies

**Core (no GPU):** `trimesh >= 4.0`, `numpy >= 1.24`

**Optional:**
- `genesis-world >= 0.2` — Genesis simulator (Part 2)
- `torch >= 2.0` — 3D generation + sim (ROCm / CUDA)
- [TRELLIS.2](https://github.com/ZJLi2013/TRELLIS.2/tree/rocm) — default 3D backend (ROCm fork)
- [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) — fallback 3D backend
- `viser >= 1.0` — 3D visualization (`pip install -e ".[viz]"`)
- `vla-eval >= 0.1` — VLA evaluation harness (benchmark plugin)
- `pybullet >= 3.2` — physics validation

## Documentation

- [docs/design.md](docs/design.md) — Architecture design (Part 1/2 + core abstractions)
- [docs/study.md](docs/study.md) — Research notes (RoboLab, 3D generation, benchmarks)
- [docs/part1-exp.md](docs/part1-exp.md) — Part 1 experiment results (asset pipeline)
- [docs/part2.md](docs/part2.md) — Part 2 summary (data engine + eval)
- [docs/part2-exp.md](docs/part2-exp.md) — Part 2 experiment results
- [docs/part3.md](docs/part3.md) — Part 3 design: irregular object grasping data generation
- [docs/background.md](docs/background.md) — Technical background (watertight mesh, URDF, convex hull)

## Acknowledgments

RoboSmith builds on top of excellent open-source projects:

- [LeRobot](https://github.com/huggingface/lerobot) — robot learning dataset format and VLA training infrastructure
- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) — GPU-accelerated physics simulator
- [RoboLab](https://github.com/NVLabs/RoboLab) — composable task definition and evaluation design inspiration
- [vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness) — standardized VLA evaluation framework
- [TRELLIS.2](https://github.com/microsoft/TRELLIS) — 3D asset generation backbone
- [Objaverse](https://objaverse.allenai.org/) — curated 3D object dataset
