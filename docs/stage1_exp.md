# Stage 1 实验记录

> 单物体 pick-cube (vision-only)，Franka + Genesis，目标 unseen 80%+。
> 实验计划见 [stage-1.md](stage-1.md)。
>
> 全部实验均为 **vision-only**（无 `--add-goal`，9D state + images），对齐 Stage 2 视觉泛化设定。

---

## 环境

### MI308X (Training + Eval)

| 项目 | 值 |
|------|-----|
| GPU | AMD Instinct MI308X (192GB VRAM × 8) |
| ROCm | 7.1.1 (torch 2.9.1+rocm7.1.1) |
| Genesis | 0.4.5 (+ ROCm cuda.bindings patch) |
| LeRobot | 0.4.4 |
| SmolVLA weights | `lerobot/smolvla_base` (450M params, 99.9M trainable) |
| transformers | 4.57.6 |
| Video backend | **pyav** (避免 torchcodec CUDA 依赖) |
| Python | 3.12 |
| Docker | `rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1` |
| Node | `banff-cyxtera-s71-4.ctr.dcgpu` |

### RDNA4 (Data Generation)

| 项目 | 值 |
|------|-----|
| GPU | AMD Radeon AI PRO R9700 × 4 (RDNA4, gfx1201, 16GB) |
| ROCm | 7.2.0 (torch 2.9.1+rocm7.2.0) |
| Genesis | 0.4.5 |
| 渲染 | EGL + Mesa radeonsi (GPU 硬件光栅化) |
| Node | `10.161.176.9` |

## 实验结果汇总

| Exp | 数据管线 | Episodes | 训练 Steps | Batch | Unseen (seed=99) | Training (seed=42) | 备注 |
|-----|---------|:--------:|:----------:|:-----:|:----------------:|:-----------------:|------|
| B0-old | 开环 IK (MI308X data gen + train, PNG nw=0) | 100 | 2000 | 4 | **1/10 = 10%** | **0/10 = 0%** | ~0.6 epochs, 训练 81 min |
| **B0** | **开环 IK (RDNA4 data gen + MI308X train, video nw=4)** | **100** | **2000** | **4** | **4/10 = 40%** | **3/10 = 30%** | **V6a 对齐！训练 10 min** |
| B0-4k | 开环 IK (RDNA4 全流程, video nw=4) | 100 | 4000 | 4 | 3/10 = 30% | 2/10 = 20% | ~1.2 epochs, loss↓ eval↓ (见 rdna4_exp.md) |
| **E2** | **开环 IK 200ep (RDNA4 全流程)** | **200** | **8000** | **4** | | | **~1.2 epochs, 测试位姿覆盖** |
| E3 | DART σ=0.005 100ep (RDNA4 全流程) | 100 | 4000 | 4 | | | ~1.2 epochs, 测试 DART 数据质量 |
| E4 | DART σ=0.005 200ep (RDNA4 全流程) | 200 | 8000 | 4 | | | ~1.2 epochs, DART + 位姿覆盖 |



## Genesis AMD 渲染后端调研

> 2025-04-10 调研。基于 [Genesis AMD issues](https://github.com/Genesis-Embodied-AI/Genesis/issues?q=AMD)、
> PR [#2393](https://github.com/Genesis-Embodied-AI/Genesis/pull/2393)、
> PR [#2680](https://github.com/Genesis-Embodied-AI/Genesis/pull/2680)、
> 以及本地 MI308X 实测。

### 当前后端确认

在 MI308X Docker 容器内执行 `gs.init(backend=gs.gpu)` 输出：

```
Running on [AMD Instinct MI308X] with backend gs.amdgpu. Device memory: 191.98 GB.
```

**结论**：Genesis 0.4.3 在 MI308X 上自动解析 `gs.gpu` → **`gs.amdgpu`**（非 OpenGL / Vulkan / CPU）。

### Genesis 后端架构

Genesis 有 5 种后端（`genesis/constants.py`）：

| 后端 | 编号 | 引擎 | 用途 |
|------|:----:|------|------|
| `cpu` | 0 | Quadrants CPU | 开发调试 |
| `gpu` | 1 | 自动解析 → cuda / amdgpu / metal | 默认推荐 |
| `cuda` | 2 | Quadrants CUDA | NVIDIA GPU 物理计算 |
| `amdgpu` | 3 | Quadrants HIP (ROCm) | AMD GPU 物理计算 |
| `metal` | 4 | Quadrants Metal | macOS |

`gs.gpu` 的解析顺序：CUDA → AMDGPU → Metal → CPU fallback。
MI308X 上 ROCm/HIP 可用 → 解析为 `amdgpu`。

**物理引擎**：通过 [Quadrants](https://github.com/Genesis-Embodied-AI/quadrants)（Taichi fork）的 AMDGPU 后端，使用 HIP kernel 做物理仿真，直接跑在 GPU 上。

**渲染引擎**：Genesis 的相机渲染使用 OpenGL/EGL（pyrender）。MI308X 是 CDNA 架构（纯计算卡，无光栅化硬件），渲染走 Mesa EGL → **llvmpipe（CPU 软件光栅化）**，这是 data gen ~28s/ep 的主要瓶颈。相比之下，NVIDIA 4090 有硬件光栅化 + EGL 加速。

### 社区 MI300X 性能基准（PR #2680, v01dXYZ, 2025-04-09）

v01dXYZ 在 MI300X 上运行了 Genesis 官方 benchmark，与 RTX Pro 6000 对比：

**物理仿真 FPS（batch_size=30000, 纯物理无渲染）**：

| 场景 | MI300X (amdgpu) | RTX 6000 (cuda) | MI300X / RTX 6000 |
|------|----------------:|-----------------:|:------------------:|
| franka | 3.12M | 5.43M | 57% |
| franka_free | 4.55M | 1.83M | **249%** |
| franka_random (gjk=off) | 2.55M | 6.74M | 38% |
| franka_random (gjk=on) | 1.98M | 6.69M | 30% |
| anymal_random | 1.33M | 6.92M | 19% |
| duck_in_box_easy | 3.04M | 6.95M | 44% |

**分析**：
- 无碰撞检测场景（`franka_free`）MI300X **大幅领先**（2.5×），得益于 HBM 带宽优势
- 有碰撞检测场景 MI300X 约为 RTX 6000 的 **30-57%**，碰撞检测 kernel 尚未针对 CDNA 优化
- 编译时间 MI300X ~50-80s vs RTX 6000 ~27-45s（Quadrants JIT 编译开销更大）

### 已知 AMD 兼容性问题

| Issue | 状态 | 影响 | 描述 |
|-------|------|------|------|
| [#2570](https://github.com/Genesis-Embodied-AI/Genesis/issues/2570) | **已修复** | **MI308X** | LLVM ISel 失败：convex collision kernel 在 CDNA3 (gfx942) 上无法编译。**我们提交的 issue**。Genesis ≤0.4.3 + Quadrants ≤0.4.4 触发；**Genesis 0.4.5 + Quadrants 0.5.2 已不再复现**（[v01dXYZ 验证](https://github.com/Genesis-Embodied-AI/Genesis/issues/2570#issuecomment-4221044595) + 本地 MI308X 确认）。`--no-bbox-detection` workaround 不再需要。 |
| [#2434](https://github.com/Genesis-Embodied-AI/Genesis/issues/2434) | Open | RDNA3 | hipMemset >1GB 数组触发 `hipErrorInvalidValue`。Workaround: `device_memory_GB=20` |
| [#2669](https://github.com/Genesis-Embodied-AI/Genesis/issues/2669) | Open | gfx1150 | `hipErrorInvalidKernelFile`：ISA 不兼容，需要 `HSA_OVERRIDE_GFX_VERSION` |
| [#2680](https://github.com/Genesis-Embodied-AI/Genesis/pull/2680) | Open PR | MI300X | 修复 `rocm-smi` fallback 和 KFD sysfs 内存报告，已含 benchmark 数据 |
| [#2679](https://github.com/Genesis-Embodied-AI/Genesis/issues/2679) | Open | CI | 提议将 MI300X 加入 GitHub CI runner（Hot Aisle $1.99/hr） |
| [#2393](https://github.com/Genesis-Embodied-AI/Genesis/pull/2393) | Merged | 全平台 | 用 `amdgpu` 后端替换了有 bug 的 `vulkan` 后端 |

### 对 RoboSmith 的影响与建议

| 环节 | 当前状态 | 瓶颈 | 可能优化 |
|------|---------|------|---------|
| **物理仿真** | `amdgpu` 后端，GPU 加速 | 碰撞检测 kernel 未优化（30-57% of CUDA） | 等待 Quadrants CDNA3 优化；简单场景影响不大 |
| **相机渲染** | Mesa EGL → **llvmpipe (CPU 软件光栅化)** | **主要瓶颈**：CDNA 无光栅化硬件，~28s/ep 中大部分是 CPU 渲染开销 | 1. 减少相机分辨率 2. 减少渲染帧数 3. 未来考虑 Vulkan compute-based renderer |
| **碰撞检测** | `box_box_detection=True` **已可用** | [#2570](https://github.com/Genesis-Embodied-AI/Genesis/issues/2570) 已修复：Genesis 0.4.5 + Quadrants 0.5.2 不再触发 LLVM ISel 崩溃。300 步物理仿真通过验证，cube 位置正确 | `--no-bbox-detection` workaround 可移除，E1+ 实验启用 box_box_detection=True |
| **并行环境** | batch_size=1（当前） | 未利用 GPU 并行 | Data gen 可考虑 batch scene 加速 |

> **✅ 更新 (2025-04-10)**：`pipeline/collect_data_dart.py` 已支持 `--no-bbox-detection` 参数。
> 但 Genesis 0.4.5 + Quadrants 0.5.2 已修复 #2570，`box_box_detection=True` 可正常使用，该 workaround 不再必要。

### 关键结论

1. **物理仿真走 `amdgpu` GPU 后端**，不是 CPU fallback
2. **相机渲染走 CPU（llvmpipe）**——CDNA 是纯计算架构无光栅化硬件，这是 data gen ~28s/ep 的主要瓶颈（NVIDIA 4090 有硬件光栅化 + EGL 加速，快得多）
3. **#2570 box_box_detection 已修复**：Genesis 0.4.5 + Quadrants 0.5.2 不再触发 LLVM ISel 崩溃（[v01dXYZ 确认](https://github.com/Genesis-Embodied-AI/Genesis/issues/2570#issuecomment-4221044595)）。本地 MI308X 验证 `box_box_detection=True` 通过 300 步物理仿真，cube 稳定落在桌面 (z=0.0195)。**`--no-bbox-detection` workaround 可移除**。注：@duburcqa 指出 Quadrants 底层 LLVM bug 未真正修复，只是 Genesis 源码变化使其不再触发
4. **Genesis AMD 生态正在快速演进**：PR #2393（2月）加入 amdgpu 后端，PR #2680（4月）修复兼容性并提供首批 MI300X benchmark
5. **box_box_detection 对实验的影响**：
   - **单物体 pick (Stage 1)**：影响较小，True/False 差异不大
   - **多物体交互 (Stage 2+)**：**必须启用**——box-box 专用碰撞检测对堆叠、推挤等场景的物理稳定性至关重要
   - **数据质量**：True 生成的 demonstration 物理行为更真实，VLA 策略更 robust
   - **E1+ 建议**：统一启用 `box_box_detection=True`

### Genesis 渲染技术栈：CDNA3 vs RDNA4

> CDNA3 (MI300X/MI308X) = 纯计算卡，无图形流水线硬件。
> RDNA4 (RX 9070 XT) = 完整图形+计算架构，有光栅化/RT 硬件。

#### 系统级图形驱动支持

| 技术栈 | CDNA3 (gfx942) | RDNA4 (gfx1200) |
|--------|:--------------:|:---------------:|
| **OpenGL (radeonsi)** | **不支持** — 无光栅化硬件 | **支持** — OpenGL 4.6, Mesa radeonsi |
| **Vulkan (RADV)** | **显式拒绝** — Mesa 25.0 起 bail out ([Phoronix](https://www.phoronix.com/news/Mesa-25.0-RADV-No-CDNA)) | **支持** — Vulkan 1.4 + RT, Mesa RADV |
| **EGL headless** | Mesa EGL → **llvmpipe (CPU)** | Mesa EGL → **radeonsi (GPU 硬件加速)** |
| **ROCm compute (HIP)** | **完整支持** — 主要用途 | **支持** — ROCm 6.4.1+ ([参考](https://kaeru.my/notes/amd-radeon-9070-xt-on-ubuntu-linux-25-04-with-rocm-6-4-1-and-mesa-25)) |

#### Genesis 渲染后端适配

| Genesis 后端 | 底层技术 | CDNA3 (MI308X) | RDNA4 (RX 9070 XT) | NVIDIA (4090) |
|-------------|---------|:--------------:|:-------------------:|:-------------:|
| **Rasterizer** (默认) | OpenGL / EGL | llvmpipe (**CPU**, ~5 FPS) | radeonsi (**GPU**, ~100+ FPS) | **GPU**, ~200+ FPS |
| **RayTracer** | LuisaRender + OptiX | **不可用** (需 CUDA + OptiX) | **不可用** (LuisaCompute 无 HIP 后端) | **可用** |
| **BatchRenderer** | gs-madrona (CUDA) | **不可用** (CUDA-only) | **不可用** (CUDA-only) | **可用** |

#### CDNA3 上的渲染优化选项

| 方案 | 预计收益 | 复杂度 | 可行性 |
|------|---------|--------|--------|
| 降低分辨率 (640→320) | ~2-4× | 低 | **推荐**：像素量减 75%，llvmpipe 近线性加速 |
| 跳帧渲染 (每 N 步) | ~2-3× | 中 | 需改 data collection 逻辑 |
| HIP compute rasterizer | ~10-20× (理论) | 极高 | 用 HIP 实现软件光栅化，工程量巨大，非标准方案 |

#### RDNA4 上的渲染加速路径

| 方案 | 预计收益 vs CDNA3 llvmpipe | 复杂度 | 可行性 |
|------|:-------------------------:|--------|--------|
| **EGL + radeonsi** (Genesis Rasterizer 默认路径) | **20-60×** | **零改动** | **直接可用**：Genesis 检测到 RDNA GPU 自动走硬件 OpenGL |
| RADV Vulkan renderer (如有 Genesis 支持) | ~20-60× | 高 | Genesis 暂无 Vulkan 渲染后端 |
| LuisaRender HIP 后端 | ~20-50× | 极高 | LuisaCompute 目前仅 CUDA/DX12/Metal，无 HIP |
| gs-madrona ROCm 移植 | 待定 | 高 | gs-madrona 是 CUDA-only，需社区移植 |

#### RDNA4 作为渲染加速卡的可行性

**混合架构方案**：MI308X (CDNA3) 做物理仿真 + VLA 训练，RDNA4 (RX 9070 XT) 做渲染。

| 考量 | 评估 |
|------|------|
| **硬件可用性** | RX 9070 XT 是消费级卡 (~$550)，PCIe 插槽需要确认服务器兼容性 |
| **Genesis 支持** | Rasterizer 后端走 EGL + radeonsi，**零代码改动**即可 GPU 加速渲染 |
| **ROCm 兼容** | ROCm 6.4.1 已支持 RDNA4 ([参考](https://kaeru.my/notes/amd-radeon-9070-xt-on-ubuntu-linux-25-04-with-rocm-6-4-1-and-mesa-25)) |
| **Headless 渲染** | 需确认 RDNA4 无显示器时 EGL headless 是否正常（RDNA3 W7900 已验证） |
| **多 GPU 路由** | Genesis 物理仿真和渲染可能需要在不同 GPU 上，需验证 `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` 路由 |
| **Data gen 提速** | 渲染从 CPU ~5 FPS → GPU ~100+ FPS，**data gen 每 ep 可能从 28s 降到 3-5s** |

#### 总结

| | CDNA3 (MI308X) | RDNA4 (RX 9070 XT) | NVIDIA (4090) |
|-|:-:|:-:|:-:|
| 物理仿真 | GPU (amdgpu) | GPU (amdgpu) | GPU (cuda) |
| 渲染 | **CPU** (llvmpipe) | **GPU** (radeonsi) | **GPU** (OpenGL/EGL) |
| RayTracer | 不可用 | 不可用 | 可用 (OptiX) |
| BatchRenderer | 不可用 | 不可用 | 可用 (CUDA) |
| VLA 训练 | **最优** (192GB HBM3) | 受限 (16GB GDDR6) | 受限 (24GB GDDR6X) |

**建议**：
- **短期 (Stage 1)**：继续在 MI308X + CPU 渲染，data gen 慢但仅需跑一次，训练迭代瓶颈已解决
- **中期 (Stage 2-3)**：评估混合架构——MI308X 训练 + RDNA4 渲染加速，data gen 可提速 5-10×
- **长期**：关注 Genesis 社区 gs-madrona ROCm 移植 和 LuisaCompute HIP 后端进展

### #2570 box_box_detection 修复验证（2025-04-10）

> 背景：[#2570](https://github.com/Genesis-Embodied-AI/Genesis/issues/2570) 是我们提交的 issue。v01dXYZ [回复](https://github.com/Genesis-Embodied-AI/Genesis/issues/2570#issuecomment-4221044595)
> 在 MI300X VF 上测试，Genesis 0.4.4+ (Quadrants 0.4.5+) 无法复现，0.4.3 及以下可复现。

**本地 MI308X 验证**：

| 测试项 | 结果 |
|--------|------|
| Genesis 版本 | 0.4.5 |
| Quadrants 版本 | 0.5.2 |
| `box_box_detection=True` + Franka + Box | **PASS** — scene.build() 成功，无 LLVM Fatal Error |
| 物理仿真 300 步 | **PASS** — cube z=0.0195 (预期 ~0.02)，稳定落在桌面 |
| scene.build() 耗时 | 27.6s（含 JIT 编译） |
| 运行 FPS | ~45-47 FPS |

**box_box_detection 功能说明**：

| | `box_box_detection=True` | `box_box_detection=False` |
|--|--|--|
| 碰撞算法 | 专用 `func_narrow_phase_convex_specializations` kernel | 通用 convex 碰撞回退 |
| Cube-Table 接触 | 精确 box-plane 接触面 | 通用凸包近似 |
| Gripper-Cube 夹持 | 精确 finger-box 侧面接触力 | 近似接触，夹持力可能不精确 |
| 多 Box 堆叠 (Stage 2) | **稳定**——精确 box-box 接触 | 可能穿模或堆叠不稳 |
| Stage 1 影响 | 较小 | 较小 |
| Stage 2+ 影响 | **必须启用** | 碰撞退化 |

**结论**：`--no-bbox-detection` workaround 不再需要。E1+ 实验及后续 Stage 2 均应启用 `box_box_detection=True`。

> ⚠ @duburcqa [指出](https://github.com/Genesis-Embodied-AI/Genesis/issues/2570#issuecomment-4221044595)：
> Quadrants 底层 LLVM ISel bug 并未真正修复，只是 Genesis 源码变化使其不再触发。
> 若未来 Genesis 更新导致该 kernel 路径变化，可能重新触发。建议持续跟踪。

---

## I/O 性能 Benchmark（MI308X）

> 2025-04-10，`banff-cyxtera-s71-4`，存储路径 `/datasets`（空闲 NVMe `nvme2n1`）。
> 目的：定位 B0 训练 78 min（2K steps）的瓶颈，确定最优存储格式与 DataLoader 配置。

### 背景

B0 使用 `--no-videos`（PNG 存储）+ `num_workers=0`，训练 2K steps 耗时 78 min，而 4090 参考值约 6 min。
原始 `--no-videos` 是为了规避 SVT-AV1 编码在根分区（99% full）上 hang 的问题。

### 数据生成（20 episodes）

| 格式 | 耗时 | 磁盘占用 | 每 episode | 备注 |
|------|------|---------|-----------|------|
| PNG | 615s (10.3 min) | 235 MB | ~30.8s | 数千小文件 |
| Video (SVT-AV1) | 562s (9.4 min) | **15 MB** | ~28.1s | 顺序写入，15.7× 更紧凑 |

**结论**：Data gen 瓶颈是 Genesis GPU 渲染（~28s/ep），与存储格式无关。
SVT-AV1 在空闲 NVMe 上完全正常，之前 hang 是根分区满导致。

### 训练（200 steps, batch 4）

| 格式 \ num_workers | nw=0 | nw=1 | nw=4 |
|--------------------|------|------|------|
| **PNG** | 521s | 480s | 159s |
| **Video** | 190s | 141s | **103s** |

参考：GPU 计算 ~0.22s/step → 200 steps 理论下限 ~44s。

### 关键发现

1. **Video 格式大幅降低 I/O 开销**：Video nw=0 (190s) 已比 PNG nw=0 (521s) 快 **2.7×**——顺序读取 video 比打开数千个 PNG 文件高效得多
2. **num_workers 对 PNG 效果巨大**：521s → 159s (**3.3×**)；对 Video 边际收益较小：190s → 103s (1.8×)
3. **最优组合：Video + nw=4 = 103s**，比原始 PNG + nw=0 快 **5.1×**

### 推算 2K steps 训练耗时

| 配置 | 预估 2K steps | vs 4090 参考 (6 min) |
|------|-------------|---------------------|
| PNG nw=0（B0 实际） | ~87 min | 14.5× 慢 |
| PNG nw=4 | ~26 min | 4.3× 慢 |
| Video nw=0 | ~32 min | 5.3× 慢 |
| **Video nw=4（推荐）** | **~17 min** | 2.8× 慢 |
| GPU 计算理论下限 | ~7.3 min | 1.2× 慢 |

### 推荐配置（后续实验统一采用）

| 阶段 | 配置 |
|------|------|
| Data gen | **去掉 `--no-videos`**（使用 video 格式），存储到 `/datasets` NVMe |
| Training | **`num_workers=4`**，自动读取 video 格式数据 |
| 存储路径 | `/datasets/zhengjli/` — 避免使用根分区 `~/.hf_cache` |

---

## B0：V6a 复现 — Hybrid Workflow（RDNA4 Data Gen + MI308X Train/Eval）

**目的**：使用 RDNA4 硬件加速渲染生成数据，MI308X 训练和评估，对齐 V6a ~40%。

### 数据采集（RDNA4 RX 9070 XT）

数据集已上传 HuggingFace: [`lidavidsh/franka-pick-100ep-genesis`](https://huggingface.co/datasets/lidavidsh/franka-pick-100ep-genesis)

| 指标 | 值 |
|------|-----|
| GPU | AMD Radeon AI PRO R9700 (RDNA4, gfx1201) |
| 渲染 | EGL + Mesa radeonsi (GPU 硬件光栅化) |
| 成功 episodes | 100/100 (100%) |
| 采集耗时 | 629s (~10 min), **6.3s/ep** |
| 格式 | Video (SVT-AV1), ~80 MB |
| 数据集 | 13,500 frames (135 × 100 ep), 2 cameras (640×480) |

### 训练（MI308X）

| 指标 | 值 |
|------|-----|
| 环境 | ROCm 7.1.1 + PyTorch 2.9.1 + lerobot 0.4.4 + transformers 4.57.6 |
| 视频后端 | **pyav** (避免 torchcodec CUDA 依赖) |
| num_workers | 4 |
| 训练耗时 | **606s (~10 min)**, 0.30 s/step |
| Loss start→end | 0.967 → 0.027 |
| Peak VRAM | 2.37 GB |
| Epochs | 2000 × 4 / 13500 ≈ 0.6 |

### 评估（MI308X, Genesis CPU 渲染）

| Seed | 成功 / 总数 | 成功率 | 耗时 |
|:----:|:----------:|:-----:|:----:|
| 99 (unseen) | **4/10** | **40%** | 352s |
| 42 (training) | **3/10** | **30%** | 252s |

Per-episode (seed=99 unseen):

| Episode | Cube XY | Result | Max Lift |
|:-------:|---------|:------:|:--------:|
| 0 | (0.521, -0.120) | OK | 0.156m |
| 1 | (0.454, -0.101) | OK | 0.162m |
| 2 | (0.628, -0.100) | FAIL | 0.017m |
| 3 | (0.515, 0.074) | FAIL | 0.000m |
| 4 | (0.562, 0.175) | FAIL | 0.007m |
| 5 | (0.547, -0.031) | FAIL | 0.000m |
| 6 | (0.584, -0.113) | OK | 0.156m |
| 7 | (0.671, -0.050) | FAIL | 0.005m |
| 8 | (0.516, 0.073) | OK | 0.155m |
| 9 | (0.446, 0.064) | FAIL | 0.000m |

### 全流程耗时

| 阶段 | 耗时 | 硬件 |
|------|:----:|:----:|
| Data gen (100 ep) | ~10 min | RDNA4 |
| Training (2K steps) | ~10 min | MI308X |
| Eval unseen (10 ep) | ~6 min | MI308X |
| Eval training (10 ep) | ~4 min | MI308X |
| **Total** | **~30 min** | |

### 分析

1. **V6a baseline 对齐成功**：unseen 40% 与 lerobot V6a MI308 参考值一致
2. **训练速度大幅提升**：从 B0-old 的 81 min 降至 10 min (**8× 加速**)，关键因素：
   - Video 格式 + pyav backend (避免 torchcodec CUDA 依赖)
   - num_workers=4 并行加载
   - /datasets NVMe 存储
3. **Hybrid workflow 验证成功**：RDNA4 data gen + MI308X train/eval 的异构方案完全可行
4. **B0-old vs B0 差异**：B0-old 仅 10% unseen，B0 达到 40%，可能原因：
   - B0-old 使用 PNG 格式 + nw=0，数据加载极慢导致训练质量差异
   - 不同 transformers 版本 (5.3.0 vs 4.57.6) 可能影响 SmolVLA 行为
5. **training seed 30%**：首次在 seed=42 上看到非零成功率，说明模型泛化能力有改善

### Next Step

- [x] V6a baseline 对齐 (unseen 40%) → 进入 E1
- [ ] E1: 增加训练量 (10K steps, batch 16)，预期 unseen > 60%
- [ ] 后续实验统一使用 hybrid workflow + video/pyav/nw=4

---

## B0-old：V6a 复现（首次尝试，MI308X 全流程）

**目的**：确认 pipeline 迁移后功能正确，复现 lerobot V6a ~40%。
**结论**：因 PNG/nw=0 导致训练质量受损，仅达到 10%。后被 B0 hybrid workflow 替代。

V6a 原始参数：100 ep, vision-only (无 `--add-goal`), 9D state + images, 2000 steps, batch 4。

### 数据采集

```bash
python pipeline/collect_data.py \
    --n-episodes 100 \
    --repo-id local/franka-pick-vision-100ep \
    --seed 42 \
    --save \
    --task "Pick up the red cube."
    # 注意：不加 --add-goal（vision-only）
```

| 指标 | 值 |
|------|-----|
| 成功 episodes | 100/100 (100%) |
| 采集耗时 | ~40 min |
| 数据集大小 | 13500 frames (135 frames × 100 ep) |

### 训练

```bash
python pipeline/train_smolvla.py \
    --dataset-id local/franka-pick-vision-100ep \
    --n-steps 2000 \
    --batch-size 4 \
    --save-dir outputs/stage1/B0
```

| 指标 | 值 |
|------|-----|
| Final loss | 0.016 |
| 训练耗时 | 4862s (~81 min), 0.23s/step (GPU) + ~2.3s data loading |
| Peak VRAM | (single GPU, ~4GB used) |
| Epochs | 2000 × 4 / 13500 ≈ 0.6 epochs |

### 评估

```bash
python pipeline/eval_policy.py \
    --policy-type smolvla \
    --checkpoint outputs/stage1/B0/final \
    --n-episodes 10 \
    --seed 99 \
    --task "Pick up the red cube."

# Sanity check (training positions)
python pipeline/eval_policy.py \
    --policy-type smolvla \
    --checkpoint outputs/stage1/B0/final \
    --n-episodes 10 \
    --seed 42 \
    --task "Pick up the red cube."
```

| Seed | 成功 / 总数 | 成功率 |
|:----:|:----------:|:-----:|
| 99 (unseen) | 1/10 | 10% |
| 42 (training) | 0/10 | 0% |

V6a 参考值：unseen 4/10 = 40%, training 0/10 = 0%

### 现象

1. **unseen 1/10 = 10%**，显著低于 V6a 参考值 40%。
2. **training 0/10 = 0%**，与 V6a 参考一致（training seed 也是 0%）。
3. 仅 ep07 (cube=(0.501,0.044)) 成功，lift=0.103m；其余 9 个 episode lift 接近 0（-0.0003m），说明手臂几乎没有运动。
4. seed=42 training 位置 10 个 episode 全部 FAIL，lift 最高仅 0.007m。
5. 训练 loss 从 0.416 降至 0.016，收敛正常。

### 分析

1. **极度 under-training**：2000 steps × batch 4 = 8000 samples seen，而数据集有 13500 frames → 仅 **0.6 epochs**。模型还没看完一遍数据就停了。
2. **V6a 参考值差异**：lerobot 原始 V6a 在 NV 4090 上跑的，本次在 MI308 上跑。虽然之前验证 MI308 V6 和 NV4090 V6a 结果接近（40%），但那是在 `lerobot_from_zero_to_expert` 的原始管线上跑的。本次是迁移后的 `robotsmith/pipeline/` 脚本，可能存在细微差异。
3. **但更可能的原因**：0.6 epochs 太少。V6a 原始也只有 ~1.6 epochs (2000 × 4 / 5000)... 等等，V6a 原始数据集是 100ep × 135frames = 13500 frames → 也是 0.6 epochs？那为什么 V6a 能到 40%？
   - 可能是 V6a 原始评估用的是不同的 action_horizon 或其他参数差异。需要仔细对比。
4. **管线基本正确**：数据采集 100%，训练 loss 正常收敛，评估流程跑通。差距主要在成功率。

### Next Step 判断

- [x] 管线正确（数据采集 100%，训练 loss 收敛，评估流程跑通）→ 进入 E1
- [ ] unseen 10% < V6a 40%，差距较大。E1 增加训练量（10K steps, batch 16 → ~11.9 epochs）应能显著提升
- [ ] 需要排查与 V6a 原始的参数差异（action_horizon=10 vs 默认值？）

---

## E1：增加训练量（V6a+ 方案）

**目的**：V6a 分析指出 2K steps + batch 4 严重 under-training（training 0%）。
验证增大训练量（10K steps, batch 16）的增量收益。100ep × 135 frames = 13500 samples, 10K × 16 / 13500 ≈ 11.9 epochs。

### 训练

```bash
python pipeline/train_smolvla.py \
    --dataset-id local/franka-pick-vision-100ep \
    --n-steps 10000 \
    --batch-size 16 \
    --save-dir outputs/stage1/E1
```

### 评估

| Seed | 成功 / 总数 | 成功率 | vs B0 |
|:----:|:----------:|:-----:|:-----:|
| 99 (unseen) | /10 | % | |
| 42 (training) | /10 | % | |

### 现象


### 分析


### Next Step 判断

- [ ] unseen 成功率提升（vs B0）且 training 成功率 > 50% → 验证训练量收益，进入 E2
- [ ] unseen 成功率无明显提升，或 training 成功率仍低 → 重新评估训练参数或数据质量

---

## E2：闭环 DART 数据

**目的**：在充分训练的前提下（10K steps），验证闭环 DART 数据 vs 开环 IK 数据的增量收益。

### 数据采集

```bash
python pipeline/collect_data_dart.py \
    --n-episodes 100 \
    --repo-id local/franka-dart-vision-100ep \
    --perturb-sigma 0.005 \
    --seed 42 \
    --save
    # 注意：不加 --add-goal（vision-only）
```

### 训练

```bash
python pipeline/train_smolvla.py \
    --dataset-id local/franka-dart-vision-100ep \
    --n-steps 10000 \
    --batch-size 16 \
    --save-dir outputs/stage1/E2
```

### 评估

| Seed | 成功 / 总数 | 成功率 | vs E1 |
|:----:|:----------:|:-----:|:-----:|
| 99 (unseen) | /10 | % | |
| 42 (training) | /10 | % | |

### 现象


### 分析


### Next Step 判断

- [ ] unseen 成功率进一步提升 → 验证 DART 收益，进入 E3
- [ ] unseen 成功率无明显提升 → 重新评估 DART 参数（σ）或数据量是否仍不足

---

## E3：增加数据量

**目的**：验证 200 ep vs 100 ep 的增量收益。200ep × 135 frames = 27000 samples, 10K × 32 / 27000 ≈ 11.9 epochs。

### 数据采集

```bash
python pipeline/collect_data_dart.py \
    --n-episodes 200 \
    --repo-id local/franka-dart-vision-200ep \
    --perturb-sigma 0.005 \
    --seed 42 \
    --save
    # 注意：不加 --add-goal（vision-only）
```

### 训练

```bash
python pipeline/train_smolvla.py \
    --dataset-id local/franka-dart-vision-200ep \
    --n-steps 10000 \
    --batch-size 32 \
    --save-dir outputs/stage1/E3
```

### 评估

| Seed | 成功 / 总数 | 成功率 | vs E2 |
|:----:|:----------:|:-----:|:-----:|
| 99 (unseen) | /10 | % | |
| 42 (training) | /10 | % | |

### 现象


### 分析


### Next Step 判断

- [ ] unseen 成功率进一步提升 → 验证数据量收益，达成 Stage 1 目标
- [ ] unseen 成功率无明显提升 → 重新评估瓶颈（VLA 模型、任务复杂度、Sim-to-Real gap）

---

## 结论

| 改进轴 | 增量效果 | 结论 |
|--------|---------|------|
| 训练量 ↑ (B0→E1) | | V6a 最大瓶颈：2K steps under-training |
| 闭环 DART (E1→E2) | | |
| 数据量 ↑ (E2→E3) | | |

**是否达到 Stage 2 入口标准？**

- [ ] Vision-only unseen: 80%+
- [ ] Vision-only training: 90%+ (sanity check)

---

## Exp-Layout-1: 多物体桌面场景布局验证

### 假设

通过修正 table 位置偏移和相机参数，`snapshot_scene.py` 渲染的 `tabletop_simple` 场景中所有物体应稳定地落在桌面上，且从 overview/top-down 两个视角可完整观察到 table + objects + Franka。

### 问题诊断

Table URDF origin 在 (0,0,0)，tabletop 范围 x∈[-0.4,0.4], y∈[-0.3,0.3]。
但 workspace_xy=[[0.35,-0.20],[0.65,0.20]] 对应 Franka 正前方操作区域（x=0.35~0.65），
大部分采样点 x>0.4 → 超出桌面 → 物理 settle 后物体掉到地面。

| 原因 | 说明 | 修复 |
|------|------|------|
| Table 位于世界原点 | 桌面 x∈[-0.4,0.4] 与 workspace x∈[0.35,0.65] 大部分不重叠 | 在 genesis_loader 中将 table 偏移到 pos=(0.5, 0, 0) |
| 相机离桌面太近 | overview 被桌面遮挡大半 | 调整相机位置和 lookat |
| 多 seed / 多视角冗余 | 验证 layout 只需 1 seed × 2 视角 | 简化为 overview + top-down |

### 实验方案

1. `genesis_loader.py`: table URDF 添加 `pos=(table_offset_x, 0, 0)`, 其中 `table_offset_x` 从 SceneConfig 推导
2. `snapshot_scene.py`: 改为 2 视角 (overview + top-down)，调整相机位置
3. 远端 `--seeds 1` 重新渲染

### 预期

- 所有 5 个物体稳定在桌面上，settle 后 z 无显著下降
- overview: 能看到完整 table + objects + Franka
- top-down: 所有物体在桌面矩形范围内

### 结果

| 轮次 | 修改 | 结果 |
|------|------|------|
| r0 (baseline) | 无 | 物体掉落，相机角度不对 |
| r1 | table offset + camera fix | （待验证） |

### 结论与 Next Step

（待实验）
