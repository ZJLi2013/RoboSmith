# RoboSmith Docker

One image per GPU architecture, plus one for GPU rendering through hipRT_render. Build from the repository root:

| Image | Genesis | Renders with | Command |
|---|---|---|---|
| MI300 (gfx942) | 0.4.5 | Genesis camera (CPU raster under Xvfb) | `docker build -f docker/Dockerfile.gfx942 -t robotsmith:gfx942-rocm6.4.3-genesis0.4.5 .` |
| MI350 (gfx950) | `[genesis]` extra, unpinned | Genesis camera (CPU raster under Xvfb) | `docker build -f docker/Dockerfile.gfx950 -t robotsmith:rocm-headless .` |
| hipRT (gfx942) | 1.4.6 | Genesis camera, or hipRT_render on the GPU | `docker build -f docker/Dockerfile.hiprt -t robotsmith:hiprt-rocm7.2.4-genesis1.4.6 .` |

The first two images are the default: certify was measured on them, and they need nothing that is not public.

[`docs/quick-start.md`](../docs/quick-start.md) covers the first run.

Headless rendering policy:

- Use `Xvfb`/GLX for Genesis camera rendering; the entrypoint starts Xvfb.
- Do not set `PYOPENGL_PLATFORM=egl`.
- The gfx942 image keeps `HSA_OVERRIDE_GFX_VERSION=9.4.2` set; the gfx950 and hipRT images unset it.
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
Without hipRT_render the image still renders through Genesis' own camera.
