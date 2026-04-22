#!/bin/bash
set -e

echo "=== Installing dependencies ==="
export PYOPENGL_PLATFORM=egl
export HF_HOME=/workspace/.hf_cache
pip install genesis-world==0.4.5 lerobot scipy trimesh --quiet 2>&1 | tail -5
pip install -e /workspace --quiet 2>&1 | tail -3

echo
echo "=== Checking versions ==="
python -c "import genesis; print('Genesis:', genesis.__version__)"
python -c "from robotsmith.scenes.presets import SCENE_PRESETS; print('Scene presets:', list(SCENE_PRESETS.keys()))"

echo
echo "=========================================="
echo "=== pick_cube 10ep (scene pipeline)    ==="
echo "=========================================="
python scripts/part2/collect_data.py \
  --task pick_cube \
  --n-episodes 10 \
  --save /workspace/output \
  --repo-id local/scene-regr-pick-cube \
  --cpu --no-videos --seed 42

echo
echo "=========================================="
echo "=== place_cube 10ep (scene pipeline)   ==="
echo "=========================================="
python scripts/part2/collect_data.py \
  --task place_cube \
  --n-episodes 10 \
  --save /workspace/output \
  --repo-id local/scene-regr-place-cube \
  --cpu --no-videos --seed 42

echo
echo "=========================================="
echo "=== stack_blocks 10ep (scene pipeline) ==="
echo "=========================================="
python scripts/part2/collect_data.py \
  --task stack_blocks \
  --n-episodes 10 \
  --save /workspace/output \
  --repo-id local/scene-regr-stack-blocks \
  --cpu --no-videos --seed 42

echo
echo "=== ALL DONE ==="
