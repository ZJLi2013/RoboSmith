# RoboSmith Docker

Build from the repository root:

```bash
docker build -f docker/Dockerfile -t robotsmith:rocm-headless .
```

Run on an AMD GPU node:

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size=24g \
  -e HIP_VISIBLE_DEVICES=0 \
  -v "$PWD":/workspace/robotsmith \
  -v /data/GraspGen:/workspace/GraspGen:ro \
  -v /data/GraspGenModels:/workspace/GraspGenModels:ro \
  robotsmith:rocm-headless \
  bash
```

Headless rendering policy:

- Use `Xvfb`/GLX for Genesis camera rendering.
- Do not set `PYOPENGL_PLATFORM=egl`.
- Do not set `HSA_OVERRIDE_GFX_VERSION` unless a specific node requires it and the override has been validated.
- Keep large asset meshes outside git; copy or mount them into `assets/objects/...` when running experiments.

For GraspGen on ROCm, install `spconv_rocm` from the ROCm branch and use the repo `torch_scatter` shim rather than PyG ROCm wheels.
