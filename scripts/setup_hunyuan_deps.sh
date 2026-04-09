#!/bin/bash
set -e
cd /data/Hunyuan3D-2.1

EXCLUDE_PKGS="torch|torchvision|torchaudio|cupy|bpy|flash.attn|triton"
grep -vEi "^(${EXCLUDE_PKGS})([<>=!~;[:space:]]|$)" requirements.txt > req_clean.txt
echo "=== Filtered requirements ==="
cat req_clean.txt
echo "=== Installing ==="
pip install -r req_clean.txt 2>&1 | tail -10
pip install trimesh pybullet 2>&1 | tail -3
echo "=== DEPS_DONE ==="
