#!/usr/bin/env bash
set -eu

# Parametrized e2e launcher: runs one scenario in the RoboSmith image.
# This script is the canonical launcher that fills in a scenario payload.
#
# Only the IMAGE tag differs per GPU arch. Defaults target MI300 / gfx942; for MI350 (gfx950) swap the image:
#   NAME=block SCENARIO=scenarios/pick_place_block_on_mat.yaml IMAGE=robotsmith:rocm-headless bash scripts/setup/run_scenario.sh
#
# `--smoke` is the only implemented rollout path in run_generated_scenario.py
# (it still runs the full episode + video + LeRobot dataset; the name is legacy).

NAME="${NAME:-mug_on_mat}"
SCENARIO="${SCENARIO:-scenarios/pick_place_mug_on_mat.yaml}"
IMAGE="${IMAGE:-robotsmith:gfx942-rocm6.4.3-genesis0.4.5}"
SEED="${SEED:-0}"
EPISODES="${EPISODES:-1}"
REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc=host --shm-size=24g \
  -e HIP_VISIBLE_DEVICES=0 \
  -e PYTHONPATH=/workspace/robotsmith \
  -v "$REPO":/workspace/robotsmith \
  -v "$HF_CACHE":/root/.cache/huggingface \
  "$IMAGE" \
  bash -lc "cd /workspace/robotsmith && python -m robotsmith.assets.fetch && python scripts/datagen/run_generated_scenario.py --scenario $SCENARIO --output-dir output/$NAME --n-episodes $EPISODES --seed $SEED --smoke"

echo "STARTED $NAME"
