#!/bin/bash
set -e

echo === Installing dependencies ===
export PYOPENGL_PLATFORM=egl
pip install genesis-world==0.4.5 lerobot scipy --quiet 2>&1 | tail -5
pip install -e /workspace --quiet 2>&1 | tail -3

echo
echo === Checking versions ===
python -c "import genesis; print('Genesis:', genesis.__version__)"
python -c "from robotsmith.grasp import GRASP_TEMPLATES; print('Bowl template:', 'bowl' in GRASP_TEMPLATES)"
python -c "from robotsmith.tasks import TASK_PRESETS; print('pick_bowl:', 'pick_bowl' in TASK_PRESETS)"

SAVE_DIR=/workspace/output

echo
echo ==========================================
echo === TEST: pick_bowl 10 episodes        ===
echo ==========================================
python scripts/part2/collect_data.py \
  --task pick_bowl \
  --n-episodes 10 \
  --save $SAVE_DIR \
  --repo-id local/regression-bowl \
  --cpu \
  --no-videos \
  --seed 42

echo
echo ==========================================
echo === BOWL TEST COMPLETE                 ===
echo ==========================================
