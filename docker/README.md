# RoboSmith Docker

One image per GPU architecture, plus one for GPU rendering through hipRT_render. Build from the repository root:

| Image | Genesis | Renders with | Command |
|---|---|---|---|
| MI300 (gfx942), default | 1.4.6 | Genesis camera, or hipRT_render on the GPU | `docker build -f docker/Dockerfile.hiprt -t robotsmith:hiprt-rocm7.2.4-genesis1.4.6 .` |
| MI350 (gfx950) | 1.4.6 (`[genesis]` extra), not yet run on gfx950 | Genesis camera (CPU raster under Xvfb) | `docker build -f docker/Dockerfile.gfx950 -t robotsmith:rocm-headless .` |
| MI300 (gfx942), retired | 0.4.5 | Genesis camera (CPU raster under Xvfb) | `docker build -f docker/Dockerfile.gfx942 -t robotsmith:gfx942-rocm6.4.3-genesis0.4.5 .` |

The hipRT image needs nothing that is not public; without a hipRT_render checkout it renders with Genesis' camera.
The 0.4.5 image is kept only to reproduce earlier results and is no longer maintained: Genesis 1.4 reports an
object's pose at its base link origin where 0.4.5 reported its URDF inertial origin, and the code now reads the 1.4
convention, so on 0.4.5 grasp and place heights are off by the inertial offset of any asset whose inertial origin
is not at its link origin (12 mm for `block_01`).

Per instance against the 0.4.5 image (MI325X; F0 block and mug, F1, F2 playground 0–9, F5 test 100–109):

| Family | Same result | Differs |
|---|---|---|
| F0 die, F0 mug, F5 | 10/10 each | — |
| F2 | 9/10 | seed 8 fails on both, `execution_failed` on 0.4.5, `task_failed` on 1.4.6 |
| F1 | 9/10 | seed 3 certifies on 0.4.5, `task_failed` on 1.4.6 |
| F0 block | 4/10 certified on 1.4.6, 9/10 on 0.4.5 | the block slips in the fingers on the way to the mat and touches it while held |

Every instance that differs, with the fields that differ among `certified`, `success`, `outcome`, `stopped` and
`first_violation`:

| Family | Seed | 0.4.5 | 1.4.6 | Differs in |
|---|---|---|---|---|
| F0 block | 0 | certified | violation | `certified`, `outcome`, `first_violation` |
| F0 block | 1 | certified | violation | `certified`, `outcome`, `first_violation` |
| F0 block | 2 | certified | violation | `certified`, `success`, `outcome`, `first_violation` |
| F0 block | 3 | certified | violation | `certified`, `outcome`, `first_violation` |
| F0 block | 6 | violation | violation | `success`, `first_violation` |
| F0 block | 8 | certified | violation | `certified`, `success`, `outcome`, `first_violation` |
| F1 | 3 | certified | task_failed | `certified`, `success`, `outcome` |
| F2 | 8 | execution_failed | task_failed | `outcome`, `stopped` |

[`docs/quick-start.md`](../docs/quick-start.md) covers the first run.

Headless rendering policy:

- Use `Xvfb`/GLX for Genesis camera rendering; the entrypoint starts Xvfb.
- Do not set `PYOPENGL_PLATFORM=egl`.
- The retired 0.4.5 image keeps `HSA_OVERRIDE_GFX_VERSION=9.4.2` set; the hipRT and gfx950 images unset it.
- Asset build files are not in git; run `python -m robotsmith.assets.fetch` before mounting the repository.

## GPU rendering with hipRT_render (gfx942)

`Dockerfile.hiprt` is ROCm 7.2.4, Python 3.12, torch 2.11 and Genesis 1.4.6 with Quadrants 1.4.0 (Genesis 1.4.6 needs
torch 2.11 on ROCm: earlier builds lack a device property its rigid solver reads). hipRT_render is
not public yet, so it is not in the image: mount a checkout and build HIPRT into it once.

```bash
docker run -it --rm --device /dev/kfd --device /dev/dri --group-add video --ipc host --shm-size 32g \
  --security-opt seccomp=unconfined -v "$PWD":/workspace/robotsmith -v /path/to/hipRT_render:/opt/hipRT_render \
  robotsmith:hiprt-rocm7.2.4-genesis1.4.6

bash scripts/setup/hiprt.sh /opt/hipRT_render   # first run builds HIPRT from source; later runs reuse it
source /opt/hipRT_render/robosmith-hiprt.env
```

`hiprt.sh` writes only the checkout's `third_party/`, `.torch_ext/` and `robosmith-hiprt.env`, so give each container
its own checkout. It clones HIPRT at `e3c01fc`, applies hipRT_render's patches and builds it; it reuses the build while
the HIPRT commit, the patches and the ROCm version are unchanged, and refuses a `third_party/HIPRT` at another commit.
The env file keeps the HIPRT source tree on the path because hiprtc reads its kernel sources at run time, and keeps the
compiled bridge in `.torch_ext/`, so only the first `bind()` in a checkout spends about two minutes compiling.

With the env file sourced, a Genesis scene renders through `instinct_render.genesis.HiprtRenderer(...).bind(scene)`.
One frame of a scenario's overview camera, with its time per frame:

```bash
python scripts/acceptance/render_frame.py --scenario scenarios/acceptance/f0_open.yaml --seed 0 --renderer hiprt --out f0.png
```

`--renderer genesis` uses Genesis' own camera instead, which needs no hipRT_render.
