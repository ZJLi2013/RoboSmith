#!/bin/bash
set -euo pipefail

# Quick I/O benchmark: PNG vs video, num_workers sweep
# All data on /datasets (1.7T empty NVMe)
# 20 episodes only — just enough to measure performance

OUT=/output/benchmark_io
mkdir -p "$OUT"

pip install -q 'transformers==5.3.0' 'accelerate' 'sentencepiece' 'protobuf' 2>&1 | tail -3
export TORCH_SDPA_BACKEND=MATH
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "  I/O Benchmark: PNG vs Video, num_workers sweep"
echo "============================================================"
echo ""

HF_BASE="/root/.cache/huggingface/lerobot"

# ============================================================
# Part 1: Data Gen — PNG vs Video (20 ep each)
# ============================================================

# --- 1a. PNG format ---
REPO_PNG="local/bench-png-20ep"
rm -rf "${HF_BASE}/${REPO_PNG}" 2>/dev/null || true

echo "=== DATA GEN: PNG format (20 ep) ==="
t0=$SECONDS
python pipeline/collect_data.py \
  --n-episodes 20 --repo-id "$REPO_PNG" \
  --save "$OUT/datagen_png" --seed 42 \
  --no-bbox-detection --no-videos \
  --task "Pick up the red cube."
PNG_TIME=$((SECONDS - t0))
PNG_SIZE=$(du -sm "${HF_BASE}/${REPO_PNG}" 2>/dev/null | cut -f1)
echo "[RESULT] PNG: ${PNG_TIME}s, ${PNG_SIZE}MB on disk"
echo ""

# --- 1b. Video format ---
REPO_VID="local/bench-video-20ep"
rm -rf "${HF_BASE}/${REPO_VID}" 2>/dev/null || true

echo "=== DATA GEN: Video format (20 ep) ==="
t0=$SECONDS
python pipeline/collect_data.py \
  --n-episodes 20 --repo-id "$REPO_VID" \
  --save "$OUT/datagen_video" --seed 42 \
  --no-bbox-detection \
  --task "Pick up the red cube."
VID_TIME=$((SECONDS - t0))
VID_SIZE=$(du -sm "${HF_BASE}/${REPO_VID}" 2>/dev/null | cut -f1)
echo "[RESULT] Video: ${VID_TIME}s, ${VID_SIZE}MB on disk"
echo ""

# ============================================================
# Part 2: Training — num_workers sweep (200 steps, batch 4)
# Use PNG data (always works), then video if no crash
# ============================================================

echo "=== TRAINING: num_workers sweep on PNG data (200 steps) ==="
for NW in 0 1 4; do
  echo ""
  echo "--- num_workers=${NW} (PNG) ---"
  SAVE_DIR="$OUT/train_png_nw${NW}"
  rm -rf "$SAVE_DIR" 2>/dev/null || true
  t0=$SECONDS
  python pipeline/train_smolvla.py \
    --dataset-id "$REPO_PNG" \
    --n-steps 200 --batch-size 4 \
    --log-every 50 --save-every 9999 \
    --num-workers "$NW" \
    --save-dir "$SAVE_DIR" 2>&1 | grep -E "step.*loss|dataloader|done"
  NW_TIME=$((SECONDS - t0))
  echo "[RESULT] PNG nw=${NW}: ${NW_TIME}s total (200 steps)"
done

echo ""
echo "=== TRAINING: num_workers sweep on Video data (200 steps) ==="
for NW in 0 1 4; do
  echo ""
  echo "--- num_workers=${NW} (Video) ---"
  SAVE_DIR="$OUT/train_vid_nw${NW}"
  rm -rf "$SAVE_DIR" 2>/dev/null || true
  t0=$SECONDS
  timeout 300 python pipeline/train_smolvla.py \
    --dataset-id "$REPO_VID" \
    --n-steps 200 --batch-size 4 \
    --log-every 50 --save-every 9999 \
    --num-workers "$NW" \
    --save-dir "$SAVE_DIR" 2>&1 | grep -E "step.*loss|dataloader|done|Error|crash" || true
  NW_TIME=$((SECONDS - t0))
  echo "[RESULT] Video nw=${NW}: ${NW_TIME}s total (200 steps)"
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "  BENCHMARK SUMMARY"
echo "============================================================"
echo ""
echo "Data Gen (20 ep):"
echo "  PNG:   ${PNG_TIME}s (~$((PNG_TIME * 5))s for 100ep), ${PNG_SIZE}MB"
echo "  Video: ${VID_TIME}s (~$((VID_TIME * 5))s for 100ep), ${VID_SIZE}MB"
echo ""
echo "Training times logged above."
echo "============================================================"
