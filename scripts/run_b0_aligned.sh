#!/bin/bash
set -euo pipefail

# Stage 1 - B0 Aligned: Reproduce MI308 V6 results
#
# Optimized config based on I/O benchmark (2025-04-10):
#   - Video format (SVT-AV1) instead of PNG → 15.7× smaller, faster training I/O
#   - num_workers=4 → 5.1× faster training vs PNG+nw=0
#   - All data on /datasets NVMe (not root partition)
#
# Expected: unseen ~40%, training ~50%
# Expected time: data gen ~47 min, training ~17 min, eval ~10 min → total ~75 min

REPO=local/franka-pick-vision-100ep
DATA_ROOT=/datasets/robotsmith
OUT=${DATA_ROOT}/stage1_b0_aligned
EXP_TAG="B0-aligned"

export HF_HOME=/datasets/hf_cache
HF_LEROBOT="${HF_HOME}/lerobot/${REPO}"

echo "=== Stage 1 ${EXP_TAG}: V6a Reproduction (Video + nw=4) ==="
echo "  GPU: AMD Instinct MI308X"
echo "  Data: 100 episodes, vision-only (9D state + images, Video/SVT-AV1)"
echo "  Training: 2000 steps, batch 4, num_workers=4"
echo "  Storage: /datasets NVMe"
echo ""

# ---- 0. Deps + AMD workarounds + headless display ----
pip install -q 'transformers==5.3.0' 'accelerate' 'sentencepiece' 'protobuf' 2>&1 | tail -3
apt-get update -qq && apt-get install -y -qq xvfb > /dev/null 2>&1 || true

export TORCH_SDPA_BACKEND=MATH
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# Headless rendering: start Xvfb for pyglet/OpenGL
if [ -z "${DISPLAY:-}" ]; then
    Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX &
    export DISPLAY=:99
    sleep 1
    echo "[env] Xvfb started on :99"
fi

python -c "import transformers; print(f'transformers=={transformers.__version__}')"

mkdir -p "$OUT"

# ---- 1. Clean stale cache (if any) ----
if [ -d "$HF_LEROBOT" ]; then
    echo "[${EXP_TAG}] Removing stale HF cache: $HF_LEROBOT"
    rm -rf "$HF_LEROBOT"
fi

# ---- 2. Data Generation (100ep, vision-only, VIDEO format) ----
echo ""
echo "[${EXP_TAG}] Step 1: Data generation (100 episodes, vision-only, video)..."
t0=$SECONDS
python pipeline/collect_data.py \
  --n-episodes 100 \
  --repo-id "$REPO" \
  --save "$OUT" \
  --seed 42 \
  --no-bbox-detection \
  --task "Pick up the red cube."
DATAGEN_TIME=$((SECONDS - t0))
echo "[RESULT] Data gen: ${DATAGEN_TIME}s (~$((DATAGEN_TIME / 60)) min)"

du -sh "$HF_LEROBOT" 2>/dev/null || true

# ---- 3. SmolVLA Training (2K steps, batch 4, num_workers=4) ----
echo ""
echo "[${EXP_TAG}] Step 2: SmolVLA training (2000 steps, batch=4, num_workers=4)..."
t0=$SECONDS
python pipeline/train_smolvla.py \
  --dataset-id "$REPO" \
  --n-steps 2000 \
  --batch-size 4 \
  --log-every 50 \
  --save-every 500 \
  --num-workers 4 \
  --save-dir "$OUT/outputs/smolvla_b0"
TRAIN_TIME=$((SECONDS - t0))
echo "[RESULT] Training: ${TRAIN_TIME}s (~$((TRAIN_TIME / 60)) min)"

# ---- 4. Eval: unseen (seed=99) ----
echo ""
echo "[${EXP_TAG}] Step 3: Eval seed=99 (unseen)..."
python pipeline/eval_policy.py \
  --policy-type smolvla \
  --checkpoint "$OUT/outputs/smolvla_b0/final" \
  --dataset-id "$REPO" \
  --n-episodes 10 --max-steps 150 \
  --action-horizon 10 \
  --seed 99 \
  --no-bbox-detection \
  --task "Pick up the red cube." \
  --save "$OUT/eval_unseen"

# ---- 5. Sanity: training (seed=42) ----
echo ""
echo "[${EXP_TAG}] Step 4: Sanity seed=42 (training)..."
python pipeline/eval_policy.py \
  --policy-type smolvla \
  --checkpoint "$OUT/outputs/smolvla_b0/final" \
  --dataset-id "$REPO" \
  --n-episodes 10 --max-steps 150 \
  --action-horizon 10 \
  --seed 42 \
  --no-bbox-detection \
  --task "Pick up the red cube." \
  --save "$OUT/eval_train"

# ---- 6. Results ----
echo ""
echo "=== ${EXP_TAG} RESULTS ==="
echo "  Data gen wall time: ${DATAGEN_TIME}s (~$((DATAGEN_TIME / 60)) min)"
echo "  Training wall time: ${TRAIN_TIME}s (~$((TRAIN_TIME / 60)) min)"
python -c "
import json
r99 = json.load(open('$OUT/eval_unseen/eval_summary.json'))
r42 = json.load(open('$OUT/eval_train/eval_summary.json'))
print(f'  seed=99 (unseen):   {r99[\"n_success\"]}/{r99[\"n_episodes\"]} = {r99[\"success_rate\"]:.0%}')
print(f'  seed=42 (training): {r42[\"n_success\"]}/{r42[\"n_episodes\"]} = {r42[\"success_rate\"]:.0%}')
print()
print('Per-episode (seed=99):')
for e in r99['results']:
    s = 'OK' if e['success'] else 'FAIL'
    print(f'  ep{e[\"episode\"]:02d} cube=({e[\"cube_xy\"][0]:.3f},{e[\"cube_xy\"][1]:.3f}) {s} lift={e[\"max_lift_m\"]:.4f}m')
print()
print('Per-episode (seed=42):')
for e in r42['results']:
    s = 'OK' if e['success'] else 'FAIL'
    print(f'  ep{e[\"episode\"]:02d} cube=({e[\"cube_xy\"][0]:.3f},{e[\"cube_xy\"][1]:.3f}) {s} lift={e[\"max_lift_m\"]:.4f}m')
"

echo ""
echo "=== ${EXP_TAG} COMPLETE ==="
