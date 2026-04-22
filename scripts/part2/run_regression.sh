#!/bin/bash
set -e

echo === Installing dependencies ===
export PYOPENGL_PLATFORM=egl
pip install genesis-world==0.4.5 lerobot scipy --quiet 2>&1 | tail -5
pip install -e /workspace --quiet 2>&1 | tail -3

echo
echo === Checking versions ===
python -c "import genesis; print('Genesis:', genesis.__version__)"
python -c "import lerobot; print('LeRobot:', lerobot.__version__)"
python -c "from robotsmith.grasp import TemplateGraspPlanner; print('GraspPlanner: OK')"
python -c "from robotsmith.motion import MotionExecutor; print('MotionExecutor: OK')"
python -c "from robotsmith.orchestration import Skill, run_skills; print('Orchestration: OK')"

SAVE_DIR=/workspace/output

echo
echo ==========================================
echo === TEST 1: pick_cube 10 episodes      ===
echo ==========================================
python scripts/part2/collect_data.py \
  --task pick_cube \
  --n-episodes 10 \
  --save $SAVE_DIR \
  --repo-id local/regression-pick \
  --cpu \
  --no-videos \
  --seed 42

echo
echo ==========================================
echo === TEST 2: place_cube 10 episodes     ===
echo ==========================================
python scripts/part2/collect_data.py \
  --task place_cube \
  --n-episodes 10 \
  --save $SAVE_DIR \
  --repo-id local/regression-place \
  --cpu \
  --no-videos \
  --seed 42

echo
echo ==========================================
echo === TEST 3: stack_blocks 10 episodes   ===
echo ==========================================
python scripts/part2/collect_data.py \
  --task stack_blocks \
  --n-episodes 10 \
  --save $SAVE_DIR \
  --repo-id local/regression-stack \
  --cpu \
  --no-videos \
  --seed 42

echo
echo ==========================================
echo === ALL TESTS COMPLETE                 ===
echo ==========================================
