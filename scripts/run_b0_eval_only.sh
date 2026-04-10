#!/bin/bash
set -euo pipefail

DATASET=lidavidsh/franka-pick-100ep-genesis
DATASET_ROOT=/datasets/robotsmith/franka-pick-100ep-genesis
OUT=/datasets/robotsmith/stage1_b0
CKPT=$OUT/outputs/smolvla_b0/final

export HF_HOME=/datasets/hf_cache
export TORCH_SDPA_BACKEND=MATH
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

apt-get update -qq && apt-get install -y -qq xvfb > /dev/null 2>&1 || true
if [ -z "${DISPLAY:-}" ]; then
    Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX &
    export DISPLAY=:99
    sleep 1
    echo "[env] Xvfb started on :99"
fi

echo "=== B0 Eval Only (model from training) ==="
echo "  Checkpoint: $CKPT"
ls "$CKPT/model.safetensors" && echo "  Model exists."

# ---- 1. Eval: unseen (seed=99) ----
echo ""
echo "[B0] Eval seed=99 (unseen)..."
t0=$SECONDS
python pipeline/eval_policy.py \
  --policy-type smolvla \
  --checkpoint "$CKPT" \
  --dataset-id "$DATASET" \
  --dataset-root "$DATASET_ROOT" \
  --n-episodes 10 --max-steps 150 \
  --action-horizon 10 \
  --seed 99 \
  --no-bbox-detection \
  --task "Pick up the cube." \
  --save "$OUT/eval_unseen"
EVAL1_TIME=$((SECONDS - t0))
echo "[RESULT] Eval unseen: ${EVAL1_TIME}s"

# ---- 2. Sanity: training (seed=42) ----
echo ""
echo "[B0] Eval seed=42 (training)..."
t0=$SECONDS
python pipeline/eval_policy.py \
  --policy-type smolvla \
  --checkpoint "$CKPT" \
  --dataset-id "$DATASET" \
  --dataset-root "$DATASET_ROOT" \
  --n-episodes 10 --max-steps 150 \
  --action-horizon 10 \
  --seed 42 \
  --no-bbox-detection \
  --task "Pick up the cube." \
  --save "$OUT/eval_train"
EVAL2_TIME=$((SECONDS - t0))
echo "[RESULT] Eval training: ${EVAL2_TIME}s"

# ---- 3. Results ----
echo ""
echo "=== B0 EVAL RESULTS ==="
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

echo "=== B0 EVAL COMPLETE ==="
