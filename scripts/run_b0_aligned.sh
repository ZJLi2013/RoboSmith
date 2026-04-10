#!/bin/bash
set -euo pipefail

# Stage 1 - B0 Aligned: Reproduce MI308 V6 results with num_workers=4
#
# Changes vs original B0:
#   1. num_workers=4 (was 0) — PNG format doesn't use torchcodec, safe on ROCm
#   2. Clean HF cache first to avoid stale data
#
# Expected: unseen ~40%, training ~50%, training time <10 min (was 78 min)

REPO=local/franka-pick-vision-100ep
OUT=/output/stage1_b0_aligned
EXP_TAG="B0-aligned"
HF_CACHE="/root/.cache/huggingface/lerobot/${REPO}"

echo "=== Stage 1 ${EXP_TAG}: V6a Reproduction + num_workers=4 ==="
echo "  GPU: AMD Instinct MI308X"
echo "  Data: 100 episodes, vision-only (9D state + images, PNG)"
echo "  Training: 2000 steps, batch 4, num_workers=4"
echo ""

# ---- 0. Deps + AMD workarounds ----
pip install -q 'transformers==5.3.0' 'accelerate' 'sentencepiece' 'protobuf' 2>&1 | tail -3

export TORCH_SDPA_BACKEND=MATH
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python -c "import transformers; print(f'transformers=={transformers.__version__}')"

# ---- 1. Clean stale cache ----
if [ -d "$HF_CACHE" ]; then
    echo "[${EXP_TAG}] Removing stale HF cache: $HF_CACHE"
    rm -rf "$HF_CACHE"
fi

# ---- 2. Data Generation (100ep, vision-only, PNG) ----
echo ""
echo "[${EXP_TAG}] Step 1: Data generation (100 episodes, vision-only)..."
t0=$SECONDS
python pipeline/collect_data.py \
  --n-episodes 100 \
  --repo-id "$REPO" \
  --save "$OUT" \
  --seed 42 \
  --no-bbox-detection \
  --no-videos \
  --task "Pick up the red cube."
echo "[${EXP_TAG}] Data gen done in $((SECONDS - t0))s"

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
echo "[${EXP_TAG}] Training done in ${TRAIN_TIME}s (~$((TRAIN_TIME / 60)) min)"

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
