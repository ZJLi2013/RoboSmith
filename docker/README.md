# RoboSmith Docker

One image per GPU architecture. Build from the repository root:

| GPU | Command |
|---|---|
| MI300 (gfx942) | `docker build -f docker/Dockerfile.gfx942 -t robotsmith:gfx942-rocm6.4.3-genesis0.4.5 .` |
| MI350 (gfx950) | `docker build -f docker/Dockerfile.gfx950 -t robotsmith:rocm-headless .` |

[`docs/quick-start.md`](../docs/quick-start.md) covers the first run.

Headless rendering policy:

- Use `Xvfb`/GLX for Genesis camera rendering; the entrypoint starts Xvfb.
- Do not set `PYOPENGL_PLATFORM=egl`.
- The gfx942 image keeps `HSA_OVERRIDE_GFX_VERSION=9.4.2` set; the gfx950 image unsets it.
- Asset build files are not in git; run `python -m robotsmith.assets.fetch` before mounting the repository.

## Genesis 1.4 and hipRT on gfx942 (not yet the default)

Genesis 1.4.6 with Quadrants 1.4.0 builds and steps the F0 acceptance scene on MI300X without code changes, and
hipRT_render's Genesis front end renders its up camera (640×480, direct lighting) in about 8 ms. Certify has not yet
been compared instance by instance against Genesis 0.4.5, so the images above stay the default.

The environment is a venv on top of a ROCm 7.2.4 image with Python 3.12 and torch 2.11 in `/opt/venv` (measured on
`zhengjli/hiprt-robotwin:rocm724-t3`, a locally built hipRT_render image):

```bash
docker run -d --name robotsmith-g14 --device /dev/kfd --device /dev/dri --group-add video --ipc host --shm-size 32g \
  --security-opt seccomp=unconfined -v /data:/data zhengjli/hiprt-robotwin:rocm724-t3
python -m venv /data/robosmith/venv
echo /opt/venv/lib/python3.12/site-packages > /data/robosmith/venv/lib/python3.12/site-packages/_opt_venv.pth
/data/robosmith/venv/bin/pip install genesis-world==1.4.6 jsonschema imageio
/data/robosmith/venv/bin/pip install --no-deps -e /data/robosmith/RoboSmith
```

- `/opt/venv` is itself a venv, so `--system-site-packages` does not see its torch; the `.pth` line does. Installing
  torch from PyPI instead would replace the ROCm build.
- hipRT_render needs both `third_party/hiprt_install` and the HIPRT source tree `third_party/HIPRT` (HIPRT `e3c01fc`
  with the repository's patches): hiprtc reads the BVH kernel sources at run time, and without them `bind()` fails
  with `hiprtBuildGeometries: hiprtError=2`.
- The first `bind()` compiles the bridge and the kernel (about 60 s); set `TORCH_EXTENSIONS_DIR` to a persistent
  directory to keep the build.
