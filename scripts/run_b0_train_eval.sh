#!/bin/bash
set -euo pipefail

# Stage 1 - B0: Training + Eval only (data already generated)
# Fix: pin transformers==5.3.0 to avoid lerobot groot dataclass breakage

REPO=local/franka-pick-vision-100ep
OUT=/output/stage1_b0
EXP_TAG="B0"

echo "=== ${EXP_TAG}: Training + Eval (data already exists) ==="

# ---- 0. Install deps (pinned versions) ----
echo "[${EXP_TAG}] Step 0: Installing dependencies (pinned)..."
pip install -q 'transformers==5.3.0' 'accelerate' 'sentencepiece' 'protobuf' 2>&1 | tail -3

export TORCH_SDPA_BACKEND=MATH
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
echo "[${EXP_TAG}] TORCH_SDPA_BACKEND=$TORCH_SDPA_BACKEND"

python -c "import transformers; print(f'transformers=={transformers.__version__}')"

# ---- 2. SmolVLA Training (2000 steps, batch 4) ----
echo ""
echo "[${EXP_TAG}] Step 2: SmolVLA training (2000 steps, batch=4)..."
python pipeline/train_smolvla.py \
  --dataset-id "$REPO" \
  --n-steps 2000 \
  --batch-size 4 \
  --log-every 50 \
  --save-every 500 \
  --num-workers 0 \
  --save-dir "$OUT/outputs/smolvla_b0"

# ---- 3. Eval: unseen positions (seed=99) ----
echo ""
echo "[${EXP_TAG}] Step 3: Eval seed=99 (unseen cube positions)..."
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

# ---- 4. Sanity: training positions (seed=42) ----
echo ""
echo "[${EXP_TAG}] Step 4: Sanity seed=42 (training positions)..."
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

# ---- 5. Results ----
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
