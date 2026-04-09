#!/usr/bin/env bash
set -euo pipefail

# RoboSmith Demo — Environment Setup
# Run inside a ROCm Docker container on an AMD GPU node.
#
# Usage:
#   docker exec -it robotsmith_demo bash
#   bash demo/setup_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
HUNYUAN_DIR="${REPO_ROOT}/Hunyuan3D-2.1"

echo "============================================"
echo "  RoboSmith Demo — Environment Setup"
echo "============================================"
echo "  REPO_ROOT:   ${REPO_ROOT}"
echo "  HUNYUAN_DIR: ${HUNYUAN_DIR}"
echo ""

# --- Step 1: Install RoboSmith + T2I dependencies ---
echo "[1/5] Installing RoboSmith + T2I bridge..."
cd "${REPO_ROOT}"
pip install -e . --quiet 2>/dev/null || pip install -e .
pip install trimesh numpy --quiet
pip install diffusers transformers accelerate sentencepiece protobuf --quiet 2>/dev/null || \
    pip install diffusers transformers accelerate sentencepiece protobuf
echo "  ✓ RoboSmith + T2I (SDXL-Turbo) installed"

# --- Step 2: Clone Hunyuan3D-2.1 ---
if [ -d "${HUNYUAN_DIR}/hy3dshape" ]; then
    echo "[2/5] Hunyuan3D-2.1 already cloned, skipping..."
else
    echo "[2/5] Cloning Hunyuan3D-2.1..."
    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 "${HUNYUAN_DIR}"
    echo "  ✓ Cloned"
fi

# --- Step 3: Install Hunyuan3D dependencies ---
echo "[3/5] Installing Hunyuan3D-2.1 dependencies..."
cd "${HUNYUAN_DIR}"

EXCLUDE_PKGS="torch|torchvision|torchaudio|cupy|bpy|flash.attn|triton|numpy"
grep -vEi "^(${EXCLUDE_PKGS})([<>=!~;\t ]|$)" requirements.txt > /tmp/req_clean.txt || true
pip install -r /tmp/req_clean.txt --quiet 2>/dev/null || pip install -r /tmp/req_clean.txt
echo "  ✓ Python dependencies installed"

# --- Step 4: Compile custom_rasterizer ---
echo "[4/5] Compiling custom_rasterizer..."
cd "${HUNYUAN_DIR}/hy3dpaint/custom_rasterizer"

export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx942}"
pip install -e . --no-build-isolation --quiet 2>/dev/null || \
    pip install -e . --no-build-isolation
echo "  ✓ custom_rasterizer built"

# --- Step 5: Compile DifferentiableRenderer ---
echo "[5/5] Compiling DifferentiableRenderer..."
cd "${HUNYUAN_DIR}/hy3dpaint/DifferentiableRenderer"
pip install pybind11 --quiet 2>/dev/null || true
bash compile_mesh_painter.sh
echo "  ✓ DifferentiableRenderer built"

# --- Set environment ---
cd "${REPO_ROOT}"
export HUNYUAN3D_REPO_PATH="${HUNYUAN_DIR}"

echo ""
echo "============================================"
echo "  ✓ Setup complete!"
echo ""
echo "  Next steps:"
echo "    export HUNYUAN3D_REPO_PATH=${HUNYUAN_DIR}"
echo "    python demo/run_pipeline.py"
echo "============================================"
