[EN](./README-en.md) | [CN](./README.md)

# RoboSmith

Sim-ready digital asset library with 3D generation fallback for robot manipulation research.

## What it does

```
User query: "red mug"  (text)
       │
       ▼
 AssetLibrary.search()
       │
   ┌───┴───┐
   hit     miss
   │         │
   ▼         ▼
 Return    FLUX T2I → ref image → Hunyuan3D-2.1 → mesh_to_urdf → auto-catalog
 URDF      (text→image)   (image→3D, 344K verts, 60s)           │
   │         ┌───────────────────────────────────────────────────┘
   ▼         ▼
 Sim-ready Asset (URDF + collision mesh + metadata)
```

> **Note**: All current SOTA 3D generation models (Hunyuan3D-2.1, TRELLIS.2, TripoSG) are single-image-to-3D
> and do not accept pure text input. When a text query misses in the library, a T2I model (FLUX) first generates
> a reference image which is then fed to the 3D generation pipeline.
> See [docs/design.md — T2I bridge analysis](docs/design.md#14-text-to-image-桥接组件).

## Quick start

```bash
pip install -e .

# Bootstrap built-in assets (12 objects)
robotsmith bootstrap

# List all assets
robotsmith list

# Search
robotsmith search "cup"
robotsmith search "red block"

# Resolve a scene preset
robotsmith scene tabletop_simple

# Generate a new asset (requires GPU + Hunyuan3D-2.1)
robotsmith generate "red ceramic mug" --image reference.png   # with ref image
robotsmith generate "red ceramic mug"                          # no image: auto T2I → I2-3D (planned)

# Validate all assets with PyBullet
pip install pybullet
robotsmith validate
```

## Built-in assets

12 primitive URDF objects, ready for physics simulation:

| Object | Tags | Mass |
|--------|------|------|
| mug_red | mug, cup, red, container | 0.25 kg |
| bowl_white | bowl, white, container | 0.30 kg |
| plate_round | plate, dish, round | 0.35 kg |
| block_red | block, cube, red, stackable | 0.05 kg |
| block_blue | block, cube, blue, stackable | 0.05 kg |
| block_green | block, cube, green, stackable | 0.05 kg |
| bottle_tall | bottle, tall, green, container | 0.40 kg |
| can_soda | can, soda, red, cylinder | 0.35 kg |
| fork_silver | fork, silver, utensil | 0.04 kg |
| spoon_silver | spoon, silver, utensil | 0.04 kg |
| table_simple | table, furniture, surface | 15.0 kg |
| plane | plane, ground, floor, static | 0.0 kg |

## 3D generation backend

Pluggable multi-backend architecture (`GenBackend` ABC + registry). Default is Hunyuan3D-2.1 (verified on ROCm).
All current SOTA 3D models are image-to-3D (no pure text input), so a T2I bridge generates a reference image when only text is provided:

```python
# With reference image — direct image-to-3D
asset = lib.generate("red ceramic mug", image_path="mug.png")

# Text only — auto T2I → image → 3D (T2I bridge, planned)
asset = lib.generate("red ceramic mug")  # auto-generates ref image via FLUX
```

| Backend | Model | PBR | VRAM | ROCm | Status |
|---------|-------|:---:|------|------|--------|
| **`hunyuan3d`** | [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) 3.3B | Yes | ≥10 GB | **Verified** (MI300X) | **Default, ready** |
| `trellis2` | [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) 4B | Yes | ≥24 GB | Blocked | Stub (cumesh/flex_gemm CUDA-only) |
| `triposg` | [TripoSG](https://github.com/VAST-AI-Research/TripoSG) 1.5B | No | ≥6 GB | Likely | Stub |

> Model survey details: [docs/design.md](docs/design.md). Background concepts: [docs/background.md](docs/background.md).

## Scene presets

1 built-in scene configuration, resolved via `ProgrammaticSceneBackend`:

- **tabletop_simple** — table + mug + bowl + 3 blocks

Scene configs are Python dataclasses with randomized object placement. An abstract `SceneBackend` interface allows future integration of AI scene generators (e.g. SceneSmith).

## Scene visualization

Web-based 3D scene preview powered by [viser](https://viser.studio). Works on headless remote GPU nodes via SSH port forwarding.

**Python API (recommended for scripting workflows):**

```python
from robotsmith.assets.library import AssetLibrary
from robotsmith.scenes.backend import ProgrammaticSceneBackend
from robotsmith.scenes.presets.tabletop_simple import tabletop_simple
from robotsmith.viz.scene_viewer import SceneViewer

lib = AssetLibrary("assets")
backend = ProgrammaticSceneBackend(seed=42)
scene = backend.resolve(tabletop_simple, lib)

viewer = SceneViewer(port=8080)
viewer.show_resolved_scene(scene)       # load all objects + table + ground
viewer.add_robot_urdf("franka.urdf")    # optional: load robot arm
viewer.run()                            # open browser at http://localhost:8080
```

**CLI (quick scene checks):**

```bash
pip install -e ".[viz]"
robotsmith view tabletop_simple
robotsmith view --asset mug_red
```

> Supports: URDF primitives (box/cylinder/sphere), OBJ/GLB meshes (Hunyuan3D generated), robot URDFs (Franka, multi-joint).

## Package structure

```
robotsmith/
  pyproject.toml
  robotsmith/
    __init__.py
    assets/
      schema.py          # Asset + AssetMetadata dataclasses
      library.py         # AssetLibrary: search, add, list, get, generate
      search.py          # Tag-based search with Chinese/English aliases
      builtin.py         # Bootstrap 12 primitive URDF assets
    gen/
      backend.py           # GenBackend ABC + registry
      hunyuan3d_backend.py # Hunyuan3D-2.1 backend (verified, default)
      trellis2_backend.py  # TRELLIS.2 backend (stub)
      triposg_backend.py   # TripoSG backend (stub)
      mesh_to_urdf.py      # trimesh → URDF (collision hull + inertia)
      catalog.py           # Auto-tag from prompt + metadata generation
      generate.py          # Orchestrator: registry → gen → convert → catalog
    scenes/
      config.py          # SceneConfig + ObjectPlacement dataclasses
      backend.py         # SceneBackend ABC + ProgrammaticSceneBackend
      presets/            # tabletop_simple
    viz/
      scene_viewer.py    # viser-based 3D scene viewer (API + CLI)
    validate/
      pybullet_check.py  # Load URDF in PyBullet DIRECT, run physics
    cli.py               # CLI entry points (incl. view subcommand)
  assets/                # Generated asset files (URDF + mesh + metadata)
  tests/                 # 30 tests (all passing)
  experiments.md         # Experiment log (Hunyuan3D / e2e verification)
  docs/design.md         # Original design document (Part 1 + Part 2 vision)
```

## Sim-ready maturity

> Background on watertight meshes, trimesh, convex hull approximation, and URDF: see [docs/background.md](docs/background.md).

A sim-ready asset needs to work across 4 levels. Current status:

```
Level   What                                      Status
─────   ────                                      ──────
L0      Loads in simulator, no crash               ✅  Done
L1      Collision works (no pass-through)           ⚠️  Partial — convex hull loses
                                                        concavities (mug handle, bowl interior)
L2      Material-accurate physics                   ❌  Not yet
L3      Visually realistic (PBR, texture, color)    ❌  Not yet
```

### What's missing and why it matters

**Visual (L3)** — critical for vision-based policy training:

| Gap | Current | Needed | Impact |
|-----|---------|--------|--------|
| Color / material | URDF `<visual>` has no `<material>` tag | Per-object RGBA or PBR material | Camera sees gray blobs → policy can't distinguish objects |
| Texture | Shape gen has no texture output | Hunyuan3D-2.1 PBR texture stage ready to enable | Large sim-to-real domain gap |
| ~~Mesh quality~~ | ~~Early shap-e was coarse~~ | **Hunyuan3D-2.1 integrated (344K verts)** | ✅ Resolved |

**Physics (L2)** — critical for realistic contact behavior:

| Gap | Current | Needed | Impact |
|-----|---------|--------|--------|
| Density | Fixed 800 kg/m³ (wood) for all objects | Per-material: ceramic ~2400, metal ~7800, glass ~2500 | Wrong mass → unrealistic dynamics |
| Friction | Fixed 0.5 for all | Ceramic ~0.3, rubber ~0.8, metal ~0.2 | Wrong friction → grasps slip or stick unrealistically |
| Collision geometry | Single convex hull | Convex decomposition (V-HACD / CoACD) | Mug handle, bowl cavity lost → grasp fails on concave shapes |
| Contact params | Not set | Stiffness, damping (Genesis supports these) | Bouncy or mushy contacts |

### Path forward

| Priority | Task | Approach |
|----------|------|----------|
| P0 | Material-aware density + friction | Infer material from prompt ("ceramic mug" → density=2400, friction=0.3), add lookup table |
| P1 | URDF color from prompt | Extract color keyword → write `<material><color rgba="..."/>` into URDF |
| P1 | Better collision geometry | Integrate V-HACD / CoACD for convex decomposition instead of single convex hull |
| ~~P2~~ | ~~Higher-quality gen model~~ | ✅ **Done** — Hunyuan3D-2.1 (3.3B, 344K verts) integrated as default |
| P2 | Texture preservation | Hunyuan3D-2.1 PBR texture stage ready to enable (custom_rasterizer built OK) |
| P3 | Genesis material properties | Write contact stiffness/damping into URDF or Genesis-specific config |

## Test results

```
30 passed in 1.35s
  - test_library.py:     13 tests (bootstrap, search, add, get)
  - test_mesh_to_urdf.py: 7 tests (box, sphere, scaling, mass, catalog)
  - test_scenes.py:       6 tests (presets, backend, determinism, stub)
```

Remote end-to-end test on MI300X: search hit + search miss → Hunyuan3D-2.1 gen → URDF → PyBullet validation — all PASS.

## Dependencies

Core (local, no GPU):
- `trimesh >= 4.0`
- `numpy >= 1.24`

Optional:
- `viser >= 1.0` — 3D scene visualization (`pip install -e ".[viz]"`)
- `pybullet >= 3.2` — physics validation
- `torch >= 2.0` — 3D generation (GPU), ROCm / CUDA
- [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) or [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) — default 3D gen backend (clone separately)
- [FLUX.1-dev](https://github.com/black-forest-labs/flux) — T2I bridge (auto-generates ref images for text queries, planned)
- `genesis-world >= 0.2` — Genesis sim (Part 2)

## Demo: end-to-end reproduction

The `demo/` directory provides scripts to reproduce the full pipeline from scratch:

```bash
# On a remote GPU node (MI300X / MI308X + ROCm 6.4)
bash demo/setup_env.sh           # 1. Install Hunyuan3D-2.1 + RoboSmith
python demo/run_pipeline.py      # 2. image → 3D mesh → URDF → catalog → visualize
```

See [demo/README.md](demo/README.md) for details.

## Design document

The original design covering both Part 1 (asset library) and Part 2 (sim-to-policy pipeline) is at [docs/design.md](docs/design.md).
