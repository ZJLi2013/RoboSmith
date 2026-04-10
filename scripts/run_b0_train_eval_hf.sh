#!/bin/bash
set -euo pipefail

# Stage 1 - B0: Train + Eval using RDNA4-generated data from HuggingFace
#
# Data: lidavidsh/franka-pick-100ep-genesis (100ep, video, RDNA4)
# Training: 2K steps, batch 4, num_workers=4, MI308X
# Eval: Genesis sim on MI308X (CPU rendering, ~10 min)

DATASET=lidavidsh/franka-pick-100ep-genesis
OUT=/datasets/robotsmith/stage1_b0
EXP_TAG="B0-hf"

export HF_HOME=/datasets/hf_cache
export TORCH_SDPA_BACKEND=MATH
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "=== Stage 1 ${EXP_TAG}: Train + Eval (RDNA4 data from HF) ==="
echo "  Dataset: ${DATASET}"
echo "  GPU: AMD Instinct MI308X"
echo "  Training: 2000 steps, batch 4, num_workers=4"
echo ""

# ---- 0. Deps + headless display ----
pip install -q 'transformers==5.3.0' 'accelerate' 'sentencepiece' 'protobuf' 2>&1 | tail -3
apt-get update -qq && apt-get install -y -qq xvfb > /dev/null 2>&1 || true

if [ -z "${DISPLAY:-}" ]; then
    Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX &
    export DISPLAY=:99
    sleep 1
    echo "[env] Xvfb started on :99"
fi

python -c "import transformers; print(f'transformers=={transformers.__version__}')"
mkdir -p "$OUT"

# ---- 1. SmolVLA Training (2K steps, batch 4, num_workers=4) ----
echo ""
echo "[${EXP_TAG}] Step 1: SmolVLA training (2000 steps, batch=4, num_workers=4)..."
echo "  Dataset will be downloaded from HF on first run."
t0=$SECONDS
python pipeline/train_smolvla.py \
  --dataset-id "$DATASET" \
  --n-steps 2000 \
  --batch-size 4 \
  --log-every 50 \
  --save-every 500 \
  --num-workers 4 \
  --save-dir "$OUT/outputs/smolvla_b0"
TRAIN_TIME=$((SECONDS - t0))
echo "[RESULT] Training: ${TRAIN_TIME}s (~$((TRAIN_TIME / 60)) min)"

# ---- 2. Eval: unseen (seed=99) ----
echo ""
echo "[${EXP_TAG}] Step 2: Eval seed=99 (unseen)..."
t0=$SECONDS
python pipeline/eval_policy.py \
  --policy-type smolvla \
  --checkpoint "$OUT/outputs/smolvla_b0/final" \
  --dataset-id "$DATASET" \
  --n-episodes 10 --max-steps 150 \
  --action-horizon 10 \
  --seed 99 \
  --no-bbox-detection \
  --task "Pick up the cube." \
  --save "$OUT/eval_unseen"
EVAL1_TIME=$((SECONDS - t0))
echo "[RESULT] Eval unseen: ${EVAL1_TIME}s"

# ---- 3. Sanity: training (seed=42) ----
echo ""
echo "[${EXP_TAG}] Step 3: Sanity seed=42 (training)..."
t0=$SECONDS
python pipeline/eval_policy.py \
  --policy-type smolvla \
  --checkpoint "$OUT/outputs/smolvla_b0/final" \
  --dataset-id "$DATASET" \
  --n-episodes 10 --max-steps 150 \
  --action-horizon 10 \
  --seed 42 \
  --no-bbox-detection \
  --task "Pick up the cube." \
  --save "$OUT/eval_train"
EVAL2_TIME=$((SECONDS - t0))
echo "[RESULT] Eval training: ${EVAL2_TIME}s"

# ---- 4. Results ----
echo ""
echo "=== ${EXP_TAG} RESULTS ==="
echo "  Training wall time: ${TRAIN_TIME}s (~$((TRAIN_TIME / 60)) min)"
echo "  Eval unseen time:   ${EVAL1_TIME}s"
echo "  Eval training time: ${EVAL2_TIME}s"
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
