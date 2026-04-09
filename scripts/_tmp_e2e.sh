#!/bin/bash
set -eu
cd /data/robotsmith
export HUNYUAN3D_REPO_PATH=/data/shared/Hunyuan3D-2.1
export PYTHONPATH=/data/shared/Hunyuan3D-2.1
echo "=== E2E: T2I -> Hunyuan3D Shape -> mesh_cleanup -> URDF ==="
python demo/run_pipeline.py --prompt "red ceramic mug" --use-t2i --size 0.1 --no-texture
echo "=== Generated assets ==="
find /data/robotsmith/assets/generated/ -type f 2>/dev/null
echo "=== DONE ==="
