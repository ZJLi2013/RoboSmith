#!/usr/bin/env bash
set -eu

# Parametrized rocRobo feature1 e2e launcher (RoboSmith + rocRobo sidecar).
# Docker flags / sidecar startup are documented once in docs/runtime.md §E2E;
# this script is the canonical launcher that fills in a scenario payload.
#
# Default planner is learned `auto` on BOTH arches (MI300 + MI350) — GraspGen/
# spconv_rocm is in both images, and drawer e2e learned die pick is verified on
# banff. Only the IMAGE tag differs per GPU arch. Defaults target wbb3 (MI350);
# for banff (MI300 / gfx942) just swap the image:
#   NAME=f1_m3 \
#     IMAGE=robotsmith:gfx942-rocm6.4.3-genesis0.4.5 \
#     bash scripts/setup/run_scenario.sh
# (PLANNER=template remains available only as a deterministic baseline / smoke.)
#
# `--smoke` is the only implemented rollout path in run_generated_scenario.py
# (it still runs the full episode + video + LeRobot dataset; the name is legacy).
# Requires the rocrobo_dev serve sidecar to be running (docs/runtime.md §E2E).

NAME="${NAME:-f1_m1}"
SCENARIO="${SCENARIO:-scenarios/pick_place_into_drawer.py}"
PLANNER="${PLANNER:-auto}"
IMAGE="${IMAGE:-robotsmith:rocm-headless-graspgen}"
SEED="${SEED:-42}"
EPISODES="${EPISODES:-1}"
REPO="${REPO:-/home/zhengjli/robot/RoboSmith}"

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc=host --shm-size=24g \
  -e HIP_VISIBLE_DEVICES=0 \
  -e PYTHONPATH=/workspace/GraspGen:/workspace/robotsmith \
  -e GRASPGEN_CONFIG=/workspace/GraspGenModels/checkpoints/graspgen_franka_panda.yml \
  -e ROCROBO_BACKEND=1 \
  -v "$REPO":/workspace/robotsmith \
  -v /home/zhengjli/robot/GraspGen:/workspace/GraspGen:ro \
  -v /home/zhengjli/robot/GraspGenModels:/workspace/GraspGenModels \
  -v /home/zhengjli/.cache/huggingface:/root/.cache/huggingface \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /usr/bin/docker:/usr/bin/docker:ro \
  "$IMAGE" \
  bash -lc "cd /workspace/robotsmith && python scripts/datagen/run_generated_scenario.py --scenario $SCENARIO --output-dir output/$NAME --n-episodes $EPISODES --seed $SEED --grasp-planner $PLANNER --smoke"

echo "STARTED $NAME"
