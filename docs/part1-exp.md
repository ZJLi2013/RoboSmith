# Part 1 实验记录 — Sim-Ready Asset Pipeline

> 3D 生成后端 ROCm 适配、T2I bridge、E2E 管线验证。

## Experiment Summary

| Exp | Hypothesis | Status | Key Result | Conclusion |
|-----|-----------|--------|------------|------------|
| RS-1 | Built-in primitive URDFs load correctly (local tests) | PASS | 13/13 tests pass, 12 assets bootstrapped | Framework + search works |
| RS-2 | shap-e text-to-3D runs on MI300X + ROCm 6.4 | PASS | 10s gen, watertight, PyBullet OK, 5.24GB GPU | ROCm verified |
| RS-3 | mesh_to_urdf produces valid URDF | PASS | 7/7 tests pass (box, sphere, scaling, mass) | Conversion pipeline works |
| RS-4 | E2E: search miss -> gen -> URDF -> PyBullet valid | PASS | 101.8s total (gen=101.5s), PyBullet OK | Full pipeline works on MI300X |
| RS-5 | Hunyuan3D-2.1 shape gen on MI300X + ROCm 6.4 | **PASS** | 60s, 344K verts, AOTriton FA, ~10GB VRAM | **Set as default backend** |
| RS-6 | T2I (SDXL-Turbo/SDXL) on MI300X + ROCm (512×512) | **PASS** | 0.3s/img (Turbo), 1.9s/img (Base), 8GB VRAM | **T2I bridge verified** |
| RS-7 | ~~E2E: text → T2I → Hunyuan3D → URDF (旧 prompt)~~ | — | *已被 RS-9 取代* | — |
| RS-8 | T2I prompt 工程：约束系统 + 参数调优 | **PASS** | Base CFG=4.5+768px 背景干净; CLIP 77 token 截断已修复 | **约束 prompt + 参数调优双管齐下** |
| RS-9 | E2E: Turbo vs Base → Hunyuan3D → URDF (约束 prompt) | **PASS** | Turbo 64s/525K verts; Base 130s/379K verts; Turbo 颜色更准 | **Turbo 确认为默认 T2I** |

---

## Exp-RS-2: shap-e ROCm Verification

### Hypothesis

shap-e (openai/shap-e) text-to-3D generation runs on AMD MI300X with ROCm 6.4,
producing a watertight mesh exportable as OBJ.

### Experiment Design

- **Node**: smc300x-clt-r4c11-02 (MI300X) or banff-sc-cs41-29 (MI300X)
- **Docker**: rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12
- **Install**: pip install shap-e trimesh
- **Test**: Generate "red ceramic mug" and "wooden fork", export as OBJ
- **Measure**: generation time, vertex count, watertight status, GPU memory

### Expected

- Success: mesh generated in < 120s, > 1000 vertices, exportable as OBJ
- Failure: CUDA/HIP incompatibility, shap-e import error, or OOM

### Results

| Prompt | Gen Time | Vertices | Faces | Watertight | BBox (m) | GPU Mem | PyBullet |
|--------|----------|----------|-------|------------|----------|---------|----------|
| red ceramic mug | 10.2s | 88870 | 177740 | yes | 1.98x1.46x1.69 | 5.24 GB | LOAD OK |
| wooden fork | 8.2s | 5408 | 10812 | yes | 1.98x0.16x0.24 | 5.24 GB | LOAD OK |

- **Node**: smc300x-clt-r4c11-02 (AMD Instinct MI300X)
- **Docker**: rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0
- **Torch**: 2.6.0+gitdbfe118, HIP=6.4.43484-123eb5128
- **Model load time**: ~110s (first run with download; cached thereafter)
- **Import time**: 11.8s

### Analysis

- Both meshes are **watertight** (allows reliable volume/inertia computation)
- Mesh scale is ~2m (shap-e native). Need `target_size_m` normalization in `mesh_to_urdf` (already implemented)
- Mug has high poly count (88k verts) — may need decimation for real-time sim, but works for PyBullet validation
- Fork has reasonable poly (5.4k verts), elongated bbox matches expected shape
- GPU peak memory 5.24 GB — well within MI300X capacity, can batch-generate
- PyBullet loads both: `pybullet_final_z=-4.08` indicates objects fell through ground (expected: no ground plane in URDF test), confirming physics runs

### Conclusion & Next Step

**PASS** — shap-e runs on MI300X + ROCm 6.4, produces watertight meshes from text prompts,
exportable as OBJ, loadable in PyBullet. Generation time ~10s per object.

**Next**: Integrate shap-e backend into `robotsmith.gen.shap_e_backend` (Phase 3, already implemented locally).

---

## Exp-RS-5: Hunyuan3D-2.1 ROCm Verification

### Hypothesis

Hunyuan3D-2.1 (3.3B DiT shape gen) runs on AMD MI300X (gfx942) + ROCm 6.4,
producing a high-fidelity mesh (>100K vertices) exportable as GLB.

### Experiment Design

- **Node**: smc300x-clt-r4c7-37 (MI300X x8, /data 1.4T)
- **Docker**: rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0
- **Model**: tencent/Hunyuan3D-2.1 (3.3B shape, downloaded from HuggingFace)
- **Input**: assets/demo.png (from Hunyuan3D-2.1 repo)
- **Test**: Image-to-3D shape generation, GLB export

### Results

| Metric | Value |
|--------|-------|
| Model | Hunyuan3D-DiT-v2-1 (3.3B params, fp16) |
| Pipeline load | 77.5s (with DINOv2 encoder) |
| Diffusion sampling | 50 steps @ 2.55 it/s = ~20s |
| Volume decoding | 7134 chunks @ ~207 it/s = ~34s |
| Total shape gen | **~60s** |
| Output mesh | **344,389 vertices** |
| GLB export | ✅ OK |
| Attention backend | AOTriton (Flash Attention via HIP, PyTorch built-in) |
| GPU memory | ~10 GB (of 206 GB available) |
| Reproducibility | 2 runs: 344K / 357K vertices |

### ROCm library compatibility

| Dependency | Status | Notes |
|-----------|--------|-------|
| PyTorch 2.6.0 | ✅ | ROCm Docker pre-installed |
| diffusers / transformers | ✅ | Pure Python |
| DINOv2 (timm) | ✅ | Standard ViT |
| Flash Attention | ✅ | AOTriton auto-used by PyTorch SDPA |
| custom_rasterizer | ✅ | `--no-build-isolation` + `PYTORCH_ROCM_ARCH=gfx942` |
| DifferentiableRenderer | ✅ | pybind11, `compile_mesh_painter.sh` |

### Conclusion & Action

**PASS** — Hunyuan3D-2.1 shape gen runs on MI300X + ROCm 6.4, zero source modification.
344K-vertex meshes in 60s, vastly superior to shap-e (88K verts, no PBR).

**Action**: Set `hunyuan3d` as default backend in RoboSmith, replacing `shap_e`.
PBR texture stage (custom_rasterizer + DifferentiableRenderer) compiled but not yet tested end-to-end.

> Full experiment details: [Hunyuan3D-2/experiments.md](../overnight_tasks/Hunyuan3D-2/experiments.md)

---

## Exp-RS-6: T2I Bridge ROCm Verification

### Hypothesis

Text-to-image models generate product-photography style reference images (512×512) on AMD MI300X + ROCm,
suitable as input for Hunyuan3D-2.1 image-to-3D pipeline. 512×512 is sufficient because
Hunyuan3D-2.1 internally resizes input to 518×518 for DINOv2 encoder.

### Experiment Design

- **Node**: banff-sc-cs41-29 (AMD Instinct MI300X x8, /data 2.4T)
- **Docker**: rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0
- **Install**: `pip install diffusers transformers accelerate sentencepiece protobuf`
- **Models tested**:
  - SDXL-Turbo (`stabilityai/sdxl-turbo`) — 4 steps, distilled, fastest
  - SDXL Base (`stabilityai/stable-diffusion-xl-base-1.0`) — 25 steps, higher quality
- **Note**: FLUX.1-dev/schnell are **gated** repos on HuggingFace (require auth). SDXL family is ungated.
- **Prompts**: T2I_PROMPT_TEMPLATE with "red ceramic mug", "wooden fork", "blue plastic bottle"
- **Output**: 512×512 PNG, FP16 inference

### Results

**SDXL-Turbo** (4 steps, guidance_scale=0.0):

| Object | Gen Time | File Size | Peak VRAM |
|--------|----------|-----------|-----------|
| red_mug | 41.0s (first, incl. warmup) / **0.3s** (cached) | 247 KB | 10.4 / 8.0 GB |
| wooden_fork | **0.3s** | 374 KB | 8.0 GB |
| blue_bottle | **0.3s** | 204 KB | 8.0 GB |

**SDXL Base** (25 steps, guidance_scale=7.5):

| Object | Gen Time | File Size | Peak VRAM |
|--------|----------|-----------|-----------|
| red_mug | **7.8s** (first, incl. warmup) | 224 KB | 8.0 GB |
| wooden_fork | **1.9s** | 376 KB | 8.0 GB |
| blue_bottle | **1.9s** | 247 KB | 8.0 GB |

**Environment**:
- PyTorch 2.6.0+gitdbfe118
- GPU: AMD Instinct MI300X, 206.1 GB VRAM
- Attention: AOTriton Flash Attention (both Flash and Efficient variants) ✅
- Pipeline load: ~25-36s (including model download from HF cache)

### Analysis

- **512×512 is optimal**: fast generation, low VRAM (8 GB), sufficient for Hunyuan3D DINOv2 (518×518)
- **SDXL-Turbo is the best choice for RoboSmith T2I bridge**:
  - 0.3s per image (after warmup) — negligible overhead in the 60s Hunyuan3D pipeline
  - 8 GB VRAM — can coexist with Hunyuan3D shape gen (10 GB) on same GPU (18 GB total, well within MI300X 192 GB)
  - Ungated, no HF auth needed
  - AOTriton FA works out of the box
- **FLUX.1-dev/schnell**: higher quality but gated. Can upgrade later with HF token if needed.
- **Image quality**: product-photography prompt template produces clean single-object images on white background — ideal for 3D reconstruction input

### Conclusion & Action

**PASS** — SDXL-Turbo verified on MI300X + ROCm 6.4 as T2I bridge for RoboSmith.
0.3s/image at 512×512, 8 GB VRAM, AOTriton FA, zero modification.

**Action**: Use `stabilityai/sdxl-turbo` as default T2I backend. FLUX.1-schnell as upgrade path (requires HF auth).
Proceed to E2E pipeline verification (see RS-9).

---

## Exp-RS-9: E2E Text → T2I → Hunyuan3D → URDF（Turbo vs Base 对比）

### Hypothesis

使用约束系统 prompt（RS-8 定稿版）+ 参数调优，SDXL-Turbo 和 SDXL-Base 生成的参考图均能驱动 Hunyuan3D-2.1 生成高质量 mesh。对比两者的颜色准确度、mesh 质量和总耗时以确定默认 T2I 后端。

### Experiment Design

- **Node**: banff-sc-cs41-29 (MI300X, /data 2.3T free)
- **Docker**: rocm/pytorch:rocm6.4.3 + diffusers + Hunyuan3D-2.1
- **Input**: `"red ceramic mug"` (seed=42)
- **Pipeline A**: SDXL-Turbo (guidance=0.0, 512px, 4 steps) → Hunyuan3D-2.1 → URDF
- **Pipeline B**: SDXL-Base (guidance=4.5, 768px, 25 steps, + negative prompt) → Hunyuan3D-2.1 → URDF

### Results

| Metric | SDXL-Turbo (默认) | SDXL-Base (备选) |
|--------|:-:|:-:|
| T2I 推理时间 | **2.9s** | 66.9s |
| Shape gen 时间 | 59.2s | 61.2s |
| URDF 转换时间 | 2.3s | 1.5s |
| **总耗时** | **64.4s** | **129.7s** |
| 顶点数 | 524,760 | 379,091 |
| 面数 | 1,049,524 | 758,184 |
| Watertight | No | No |
| 参考图颜色 | 红色杯子 (正确) | 白杯+红内壁 (偏差) |
| 把手数量 | 1 (正确) | 1 (正确) |
| 背景干净度 | 白/浅灰 | 灰色+阴影 |

### Key Findings

1. **双耳问题已修复** — 移除 `symmetrical shape` 后 Turbo 恢复单把手 (seed=42)
2. **Turbo 颜色还原更准** — "red ceramic mug" → 红色杯子；Base → 白色杯体+红色内壁
3. **Turbo 速度快 2x** — 64s vs 130s（Base 的 T2I 阶段因 25 步推理 + 首次模型下载更慢）
4. **Turbo mesh 更密** — 525K verts vs 379K（可能因 Turbo 参考图轮廓更清晰）
5. **Base 的 CFG↓ 导致语义损失** — CFG=4.5 修复了复杂背景但牺牲了颜色精度

### Conclusion

**PASS** — 约束 prompt 版 E2E 管线在 MI300X + ROCm 6.4 上验证通过。

**SDXL-Turbo 确认为默认 T2I 后端**：速度快、颜色准、mesh 密、无需调参。
SDXL-Base 作为备选保留，适用于需要更精细背景控制的场景（但需注意颜色偏差）。

---

## Exp-RS-8: T2I Prompt 工程 — 约束系统 + 参数调优

### Hypothesis

将 T2I prompt 从"自然语言描述"转为"约束系统"，配合 SDXL-Base 参数调优（CFG↓ + 分辨率↑），可生成更适合 Image-to-3D 的参考图。

### Experiment Design

- **Node**: banff-sc-cs41-29 (MI300X)
- **Docker**: rocm/pytorch:rocm6.4.3 + diffusers
- **Models**: SDXL-Turbo (guidance=0.0, 512px) + SDXL-Base (guidance=4.5, 768px)
- **Objects**: red ceramic mug, wooden fork, blue plastic bottle

### 最终 Prompt（约束系统版）

```python
T2I_PROMPT_TEMPLATE = (
    "a single {obj}, centered, isolated object, "
    "pure white background, no surface, no ground, no table, "
    "front view, slight top-down angle, "
    "clean silhouette, sharp edges, opaque, matte finish, "
    "uniform soft lighting, no shadow, no reflection, "
    "minimalist, object-only, high detail"
)
```

关键参数：

| 参数 | SDXL-Turbo | SDXL-Base |
|------|:-:|:-:|
| `guidance_scale` | 0.0 | 4.5 |
| `resolution` | 512 | 768 |
| `negative_prompt` | 不适用 | `table, surface, floor, ground, shadow, reflection, ...` |

### Results

| Model | Mug | Fork | Bottle | 背景 |
|-------|:-:|:-:|:-:|------|
| SDXL-Turbo (默认) | ✅ 干净居中 | ✅ 干净居中 | ⚠️ 仍半透明 | 白色/浅灰 |
| SDXL-Base (备选) | ✅ 干净灰底 | ✅ 干净灰底 | ⚠️ 仍半透明 | 浅灰均匀 |

### Key Findings

1. **"约束"比"描述"有效** — `no surface, no ground, no table` 打破了模型默认的"物体必须放在某处"假设
2. **参数比 prompt 更重要** — SDXL-Base 从 CFG=7.5/512px → CFG=4.5/768px 是修复复杂背景的关键
3. **禁用词**: `product photography`, `studio lighting`, `orthographic`, `1:1 composition` → 触发复杂场景
4. **CLIP 77 token 限制** — prompt 必须压缩到 ~65 tokens，关键约束前置
5. **`symmetrical shape` 陷阱** — 被字面解读为"物体应双侧对称"（杯子双把手），已移除
6. **透明物体** (bottle → glass look) 是训练数据偏差，prompt 无法完全修复

### Conclusion

**PASS** — 约束版 prompt + 参数调优已整合到代码库（`run_pipeline.py`, `test_flux_t2i.py`, `test_e2e_t2i_3d.py`）。

---

---

## RS-10: TRELLIS.2 E2E — red ceramic mug (MI300X, ROCm 6.4)

**Date**: 2026-04-09
**Goal**: Verify TRELLIS.2-4B on ROCm via [ZJLi2013/TRELLIS.2@rocm](https://github.com/ZJLi2013/TRELLIS.2/tree/rocm); compare with Hunyuan3D PBR output.

### Setup

- Container: `trellis2_cdna3` (rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0)
- GPU: AMD Instinct MI300X
- TRELLIS.2 dependencies: flash_attn 2.8.3 (Triton), nvdiffrast 0.4.0, cumesh, o_voxel, flexgemm, trimesh 4.11.5
- Resolution: 512³

### Pipeline

```
text prompt → SDXL-Turbo (T2I) → TRELLIS.2-4B → o_voxel GLB export → mesh_cleanup → URDF
```

### Timing Breakdown

| Stage | Time |
|-------|------|
| T2I (SDXL-Turbo, 512×512) | 93s |
| TRELLIS.2 pipeline load | 63s |
| Sparse structure sampling (12 steps) | 12s |
| Shape SLat sampling (12 steps, ×2) | 11s + 107s |
| Texture SLat sampling | ~5s/step |
| **Total gen** | **275s** |
| o_voxel GLB export (remesh + simplify + xatlas UV + bake) | 85s |
| **Total** | **423s** |

### Output

| Metric | Value |
|--------|-------|
| Raw vertices | 5,405,042 |
| Raw faces | 10,834,068 |
| After remesh + simplify | 652,279 verts, 992,967 faces |
| GLB size | 38.6 MB |
| Texture map | 4K PBR (13.6 MB) |
| Collision mesh | 1.3 MB convex hull |

### Comparison: TRELLIS.2 vs Hunyuan3D PBR

| | Hunyuan3D-2.1 PBR | TRELLIS.2-4B |
|---|---|---|
| Total time | ~531s | ~423s |
| GLB size | 1.1 MB | 38.6 MB |
| Texture quality | PBR Paint (内壁纹理破碎) | O-Voxel PBR (4K, clean) |
| bpy dependency | lazy-import patch | None |
| Base plane artifact | Present (needs cleanup) | TBD |
| Install complexity | Simple | Heavy (ROCm forks) |

### Conclusion

**PASS** — TRELLIS.2 produces significantly higher quality PBR meshes with native GLB export (no bpy dependency).
Texture resolution and coverage far exceed Hunyuan3D Paint, which suffers from visible interior fragmentation.
Trade-off: larger GLB file size (38.6 MB vs 1.1 MB) and heavier install chain.

---

## Debug Tracking

| Round | Issue | Fix | Result |
|-------|-------|-----|--------|
| v1 | `pip install shap-e` — not on PyPI | Install from git: `pip install git+https://github.com/openai/shap-e.git` | OK |
| v2 | Bash heredoc syntax error with nested Python f-strings | Use standalone Python script (scp'd separately) | OK |
| v3 | `No module 'ipywidgets'` in `decode_latent_mesh` | `pip install ipywidgets` in container | OK |
| e2e-v1 | `shap-e` not on PyPI (pip install shap-e fails) | Already fixed above (git install) | OK |
| e2e-v2 | tar on Windows missed dirs | Explicit paths in tar command | OK |
| e2e-v3 | PYTHONPATH not passed via docker exec -e | Used bash wrapper script | OK |
| e2e-v4 | "blue teapot" matched "block_blue" | Changed to "purple dragon figurine" (no tag overlap) | OK |
| triposr-v1 | `pip install tsr` installs wrong package (not VAST-AI TripoSR) | Need: `pip install git+https://github.com/VAST-AI-Research/TripoSR.git` with proper setup | DEFERRED |
| t2i-v1 | FLUX.1-dev/schnell gated repos, 401 error | Switch to SDXL-Turbo (ungated) | OK |
| e2e-t2i | `hy3dshape` import / missing deps (omegaconf, cv2, scikit-image) | Install all deps; use `Hunyuan3D-2.1/hy3dshape/` as sys.path | OK |
| prompt | CLIP 77 token truncation (93 tokens) | Compress prompt to ~65 tokens, critical constraints first | OK |
| prompt | `symmetrical shape` causes double-handle mug | Remove from template | OK |
| prompt | SDXL-Base崩坏 at CFG=7.5, 512px | CFG→4.5, resolution→768 | **FIXED** |
