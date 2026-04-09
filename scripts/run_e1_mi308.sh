#!/bin/bash
set -euo pipefail

# Stage 1 - E1: Increased training volume (10K steps, batch 16)
# Uses SAME data as B0 (already generated), only retrains + evals

REPO=local/franka-pick-vision-100ep
OUT=/output/stage1_e1
EXP_TAG="E1"

echo "=== Stage 1 ${EXP_TAG}: Increased Training Volume ==="
echo "  Data: B0 same (100ep, vision-only)"
echo "  Training: 10000 steps, batch 16 (~11.9 epochs)"

# ---- 0. Deps ----
pip install -q 'transformers==5.3.0' 'accelerate' 'sentencepiece' 'protobuf' 2>&1 | tail -3

export TORCH_SDPA_BACKEND=MATH
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python -c "import transformers; print(f'transformers=={transformers.__version__}')"

# ---- 1. SmolVLA Training (10K steps, batch 16) ----
echo ""
echo "[${EXP_TAG}] Step 1: SmolVLA training (10000 steps, batch=16)..."
python pipeline/train_smolvla.py \
  --dataset-id "$REPO" \
  --n-steps 10000 \
  --batch-size 16 \
  --log-every 100 \
  --save-every 2000 \
  --num-workers 0 \
  --save-dir "$OUT/outputs/smolvla_e1"

# ---- 2. Eval: unseen (seed=99) ----
echo ""
echo "[${EXP_TAG}] Step 2: Eval seed=99 (unseen)..."
python pipeline/eval_policy.py \
  --policy-type smolvla \
  --checkpoint "$OUT/outputs/smolvla_e1/final" \
  --dataset-id "$REPO" \
  --n-episodes 10 --max-steps 150 \
  --action-horizon 10 \
  --seed 99 \
  --no-bbox-detection \
  --task "Pick up the red cube." \
  --save "$OUT/eval_unseen"

# ---- 3. Sanity: training (seed=42) ----
echo ""
echo "[${EXP_TAG}] Step 3: Sanity seed=42 (training)..."
python pipeline/eval_policy.py \
  --policy-type smolvla \
  --checkpoint "$OUT/outputs/smolvla_e1/final" \
  --dataset-id "$REPO" \
  --n-episodes 10 --max-steps 150 \
  --action-horizon 10 \
  --seed 42 \
  --no-bbox-detection \
  --task "Pick up the red cube." \
  --save "$OUT/eval_train"

# ---- 4. Results ----
echo ""
echo "=== ${EXP_TAG} RESULTS ==="
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
