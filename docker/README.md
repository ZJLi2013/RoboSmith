# RoboSmith Docker

One image per GPU architecture. Build from the repository root:

| GPU | Command |
|---|---|
| MI300 (gfx942) | `docker build -f docker/Dockerfile.gfx942 -t robotsmith:gfx942-rocm6.4.3-genesis0.4.5 .` |
| MI350 (gfx950) | `docker build -f docker/Dockerfile.gfx950 -t robotsmith:rocm-headless .` |

The gfx942 image includes the GraspGen dependencies; the gfx950 image needs the GraspGen layer described in
[`docs/quick-start.md`](../docs/quick-start.md), which also covers weights, the rocRobo sidecar, and the first run.

Headless rendering policy:

- Use `Xvfb`/GLX for Genesis camera rendering; the entrypoint starts Xvfb.
- Do not set `PYOPENGL_PLATFORM=egl`.
- The gfx942 image keeps `HSA_OVERRIDE_GFX_VERSION=9.4.2` set; the gfx950 image unsets it.
- Asset build files are not in git; run `python -m robotsmith.assets.fetch` before mounting the repository.

For GraspGen on ROCm, install `spconv_rocm` from the ROCm branch and use the repo `torch_scatter` shim rather than PyG ROCm wheels.
