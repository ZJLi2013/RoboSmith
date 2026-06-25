# syntax=docker/dockerfile:1.7
#
# Interaction-Ready SDG workshop image (CDNA3 / gfx942) — fully self-contained.
#
# Goal: a single image the teacher builds once and ships to the cloud, with the
# repo, demo assets, GraspGen source + checkpoints, the rocRobo compare video and
# all runtime config BAKED IN. The student container runs with NO -v code/data
# mounts and NO host-path assumptions (only the GPU devices and docker.sock).
#
# This does NOT replace docker/Dockerfile.gfx942 — it layers on top of it, so the
# base runtime image (ROCm 6.4.3 + Genesis 0.4.5 pinned stack) stays lean and
# reusable. See docker/Dockerfile.gfx942 for the base stack rationale.
#
# Build from the RoboSmith repo root (BuildKit named contexts pull in the sibling
# repos; adjust the paths to your checkout):
#
#   docker build -f workshop/Dockerfile \
#     -t robotsmith:workshop-gfx942 \
#     --build-context graspgen=/home/zhengjli/robot/GraspGen \
#     --build-context graspgenmodels=/home/zhengjli/robot/GraspGenModels \
#     --build-context rocrobo=/home/zhengjli/robot/rocRobo \
#     .
#
# GraspGenModels must already contain the spconv_rocm-converted *_rocm.pth weights
# (scripts/setup/_convert_spconv_ckpt.py) and graspgen_franka_panda.yml must point
# at them — same contract as docs runtime.md.
ARG BASE_IMAGE=robotsmith:gfx942-rocm6.4.3-genesis0.4.5
FROM ${BASE_IMAGE}

WORKDIR /workspace/robotsmith

# Workshop-needed repo dirs that the base runtime image does not bake.
COPY scenarios ./scenarios
COPY assets ./assets
COPY workshop ./workshop

# GraspGen source + (rocm-converted) checkpoints — baked, not mounted.
COPY --from=graspgen . /workspace/GraspGen
COPY --from=graspgenmodels . /workspace/GraspGenModels

# rocRobo collision-blind-vs-aware compare video used by notebook §4.
COPY --from=rocrobo rocRobo/samples/media/demo_compare.mp4 \
     ./workshop/videos/rocrobo_compare.mp4

# Jupyter (not a base dep) + docker CLI so the notebook can `docker exec` the
# rocRobo sidecar without mounting the host docker binary.
RUN python -m pip install --no-cache-dir notebook ipykernel \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker.io \
    && rm -rf /var/lib/apt/lists/*

# All runtime config baked in (cloud deploy: no -e / -v / host-path needed).
ENV PYTHONPATH=/workspace/GraspGen:/workspace/robotsmith \
    GRASPGEN_CONFIG=/workspace/GraspGenModels/checkpoints/graspgen_franka_panda.yml \
    ROCROBO_BACKEND=1 \
    ROCROBO_CONTAINER=rocrobo_dev \
    ROCROBO_WORKDIR=/rocrobo \
    ROCROBO_PYTHONPATH=/rocrobo/rocRobo/core \
    ROCROBO_ASSETS=/rocrobo/pyroki/examples/assets

EXPOSE 8888

# ENTRYPOINT robotsmith-headless (xvfb wrapper) is inherited from the base image.
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
