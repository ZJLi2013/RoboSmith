#!/bin/bash
set -euo pipefail

# Stage 1 - B0: V6a baseline reproduction on AMD MI308
# 100ep vision-only, 2K steps, batch 4
#
# Run inside Docker:
#   docker run --rm \
#     --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
#     -e PYOPENGL_PLATFORM=egl \
#     -v ~/github/robotsmith:/workspace/robotsmith \
#     -v ~/outputs:/output \
#     -v ~/.hf_cache:/root/.cache/huggingface \
#     -w /workspace/robotsmith \
#     genesis-amd-official-lerobot:rocm643-v043-np1 \
#     bash scripts/run_b0_mi308.sh

REPO=local/franka-pick-vision-100ep
OUT=/output/stage1_b0
EXP_TAG="B0"

echo "=== Stage 1 ${EXP_TAG}: V6a Baseline Reproduction ==="
echo "  GPU: AMD Instinct MI308X"
echo "  Data: 100 episodes, vision-only (9D state + images)"
echo "  Training: 2000 steps, batch 4"
echo "  Model: SmolVLA (frozen vision + trainable expert)"
echo ""

# ---- 0. Install missing deps + AMD workarounds ----
echo "[${EXP_TAG}] Step 0: Installing missing dependencies..."
pip install -q 'transformers==5.3.0' 'accelerate' 'sentencepiece' 'protobuf' 2>&1 | tail -3

export TORCH_SDPA_BACKEND=MATH
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
echo "[${EXP_TAG}] TORCH_SDPA_BACKEND=$TORCH_SDPA_BACKEND"

# ---- 1. Data Generation (100 episodes, vision-only) ----
echo ""
echo "[${EXP_TAG}] Step 1: Data generation (100 episodes, vision-only)..."
python pipeline/collect_data.py \
  --n-episodes 100 \
  --repo-id "$REPO" \
  --save "$OUT" \
  --seed 42 \
  --no-bbox-detection \
  --no-videos \
  --task "Pick up the red cube."

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
"

echo ""
echo "=== ${EXP_TAG} COMPLETE ==="
