#!/usr/bin/env bash
set -eu

WORKDIR="/tmp/robotsmith-triposr"
LOGDIR="${WORKDIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="${LOGDIR}/triposr_verify_${TIMESTAMP}.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGFILE"; }

mkdir -p "$LOGDIR" "$WORKDIR/outputs"
log "=== START: TripoSR ROCm verification ==="
log "Node: $(hostname)"

CONTAINER_NAME="robotsmith_triposr_verify"
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

log "Step 1: Starting Docker container..."
docker run --rm -d \
  --name "$CONTAINER_NAME" \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --security-opt seccomp=unconfined \
  -v /tmp:/tmp \
  -e HIP_VISIBLE_DEVICES=0 \
  -e PYTORCH_ROCM_ARCH=gfx942 \
  -w "$WORKDIR" \
  rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0 \
  sleep infinity

rc() { docker exec "$CONTAINER_NAME" bash -c "$1"; }

log "Step 2: Install TripoSR..."
rc "pip install -q tsr trimesh rembg pillow 2>&1 | tail -10" 2>&1 | tee -a "$LOGFILE"

log "Step 2b: If tsr not found, try git install..."
rc "python -c 'import tsr; print(\"tsr OK\")' 2>/dev/null || pip install -q git+https://github.com/VAST-AI-Research/TripoSR.git trimesh rembg pillow 2>&1 | tail -10" 2>&1 | tee -a "$LOGFILE"

log "Step 3: Check import..."
rc "python -c \"
import torch
print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')
try:
    from tsr.system import TSR
    print('TripoSR import OK')
except Exception as e:
    print(f'TripoSR import FAIL: {e}')
\"" 2>&1 | tee -a "$LOGFILE"

docker stop "$CONTAINER_NAME" 2>/dev/null || true
log "=== END: $(date) ==="
