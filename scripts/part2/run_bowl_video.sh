#!/bin/bash
set -e

echo "=== Installing dependencies ==="
export PYOPENGL_PLATFORM=egl
export HF_HOME=/workspace/.hf_cache
pip install genesis-world==0.4.5 lerobot scipy --quiet 2>&1 | tail -5
pip install -e /workspace --quiet 2>&1 | tail -3

echo
echo "=== Checking versions ==="
python -c "import genesis; print('Genesis:', genesis.__version__)"
python -c "from robotsmith.tasks import TASK_PRESETS; print('pick_bowl:', 'pick_bowl' in TASK_PRESETS)"

echo
echo "=========================================="
echo "=== pick_bowl 3ep WITH VIDEOS         ==="
echo "=========================================="
python scripts/part2/collect_data.py \
  --task pick_bowl \
  --n-episodes 3 \
  --save /workspace/output \
  --repo-id local/bowl-video-demo \
  --cpu \
  --seed 42

echo
echo "=== Finding video files ==="
find /workspace/.hf_cache -name "*.mp4" 2>/dev/null
find /workspace/output -name "*.mp4" 2>/dev/null

echo
echo "=== DONE ==="
