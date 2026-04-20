# Stage 1 实验记录

> 单物体 pick-cube (vision-only)，Franka + Genesis，目标 unseen 80%+。
> 实验计划见 [stage-1.md](stage-1.md)。
>
> 全部实验均为 **vision-only**（无 `--add-goal`，9D state + images），对齐 Stage 2 视觉泛化设定。

---

## 环境

### MI300X (Training + Eval)

| 项目 | 值 |
|------|-----|
| GPU | AMD Instinct MI300X (192GB VRAM × 8) |
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
| B0-old | 开环 IK (MI300X data gen + train, PNG nw=0) | 100 | 2000 | 4 | **1/10 = 10%** | **0/10 = 0%** | ~0.6 epochs, 训练 81 min |
| **B0** | **开环 IK (RDNA4 data gen + MI300X train, video nw=4)** | **100** | **2000** | **4** | **4/10 = 40%** | **3/10 = 30%** | **V6a 对齐！训练 10 min** |
| B0-4k | 开环 IK (RDNA4 全流程, video nw=4) | 100 | 4000 | 4 | 3/10 = 30% | 2/10 = 20% | ~1.2 epochs, loss↓ eval↓ (见 rdna4_exp.md) |
| **E2** | **开环 IK 200ep (RDNA4 全流程)** | **200** | **8000** | **4** | | | **~1.2 epochs, 测试位姿覆盖** |
| E3 | DART σ=0.005 100ep (RDNA4 全流程) | 100 | 4000 | 4 | | | ~1.2 epochs, 测试 DART 数据质量 |
| E4 | DART σ=0.005 200ep (RDNA4 全流程) | 200 | 8000 | 4 | | | ~1.2 epochs, DART + 位姿覆盖 |



> Genesis AMD 渲染后端调研已移至 [study.md §9](study.md#9-genesis-amd-渲染后端调研)。

---

## I/O 性能 Benchmark（MI300X）

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

## B0：V6a 复现 — Hybrid Workflow（RDNA4 Data Gen + MI300X Train/Eval）

**目的**：使用 RDNA4 硬件加速渲染生成数据，MI300X 训练和评估，对齐 V6a ~40%。

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

### 训练（MI300X）

| 指标 | 值 |
|------|-----|
| 环境 | ROCm 7.1.1 + PyTorch 2.9.1 + lerobot 0.4.4 + transformers 4.57.6 |
| 视频后端 | **pyav** (避免 torchcodec CUDA 依赖) |
| num_workers | 4 |
| 训练耗时 | **606s (~10 min)**, 0.30 s/step |
| Loss start→end | 0.967 → 0.027 |
| Peak VRAM | 2.37 GB |
| Epochs | 2000 × 4 / 13500 ≈ 0.6 |

### 评估（MI300X, Genesis CPU 渲染）

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
| Training (2K steps) | ~10 min | MI300X |
| Eval unseen (10 ep) | ~6 min | MI300X |
| Eval training (10 ep) | ~4 min | MI300X |
| **Total** | **~30 min** | |

### 分析

1. **V6a baseline 对齐成功**：unseen 40% 与 lerobot V6a MI300 参考值一致
2. **训练速度大幅提升**：从 B0-old 的 81 min 降至 10 min (**8× 加速**)，关键因素：
   - Video 格式 + pyav backend (避免 torchcodec CUDA 依赖)
   - num_workers=4 并行加载
   - /datasets NVMe 存储
3. **Hybrid workflow 验证成功**：RDNA4 data gen + MI300X train/eval 的异构方案完全可行
4. **B0-old vs B0 差异**：B0-old 仅 10% unseen，B0 达到 40%，可能原因：
   - B0-old 使用 PNG 格式 + nw=0，数据加载极慢导致训练质量差异
   - 不同 transformers 版本 (5.3.0 vs 4.57.6) 可能影响 SmolVLA 行为
5. **training seed 30%**：首次在 seed=42 上看到非零成功率，说明模型泛化能力有改善

---


## Exp-Layout-1: 多物体桌面场景布局验证 ✅

**问题**：table URDF origin 在 (0,0,0)，workspace_xy 在 x=0.35~0.65 → 物体超出桌面掉落。

**修复（4 轮迭代）**：

| 修改 | 结果 |
|------|------|
| genesis_loader: table 偏移到 workspace 中心 | 物体在桌面，但 Franka 不在画面 |
| 相机 lookat 调整 | Franka 可见，但 base 在地面 |
| **Franka `pos=(0, 0, table_surface_z)`** | **✅ 场景完整：5 物体在桌面 + Franka base 在桌面高度** |

**结论**：`tabletop_simple` 场景布局可用（MI300X, Genesis 0.4.6 验证）。

---

## S1.4: TaskSpec + Composable Predicates + IK Strategy 框架

### Phase 0 确认

- 观测完备性: 本实验为**基础设施重构**（非训练实验），不涉及 policy 学习。验证目标是功能等价性 — 重构后的 `collect_data.py --task pick_cube` 应产出与旧版 hardcoded 完全一致的数据。
- 随机化变量: cube (x,y) 随机化，与旧版一致。
- 不需要训练/评估，只需确认数采管线功能不退化。

### 假设

将 `collect_data.py` 中 hardcoded 的 pick-cube 逻辑重构为 TaskSpec + PREDICATE_REGISTRY + IK_STRATEGIES 三层抽象后，**数采功能与旧版完全等价**（20ep 快速验证全部成功，轨迹长度一致）。

### 实验方案

**实现范围**：创建 `robotsmith/tasks/` 模块

| 组件 | 文件 | 说明 |
|------|------|------|
| `TaskSpec` dataclass | `robotsmith/tasks/task_spec.py` | 声明式任务定义（design.md §2.2 Phase 1） |
| `PREDICATE_REGISTRY` | `robotsmith/tasks/predicates.py` | 可组合谓词注册表，首个谓词 `object_above` |
| `IK_STRATEGIES` | `robotsmith/tasks/ik_strategies.py` | IK waypoint 生成策略注册表，首个策略 `pick` |
| Task 预设 | `robotsmith/tasks/presets.py` | `pick_cube` TaskSpec 实例 |
| 重构 collect_data.py | `scripts/part2/collect_data.py` | 用 TaskSpec dispatch 替代 hardcoded 逻辑 |

**对照组**：旧版 `collect_data.py`（当前 hardcoded pick-cube，B0 实验已验证 100% expert 成功率）

**变量**：代码架构变更（TaskSpec + registry），数据采集行为不变

### 预期

- 假设成立: `--task pick_cube` 20ep 全部成功（100%），轨迹长度 = 135 frames（与旧版一致）
- 假设不成立: 成功率下降或脚本报错 → IK 参数提取有误，需逐步对比

### 结果

**Phase 1: 本地验证（无 GPU）**

| 测试项 | 结果 |
|--------|:----:|
| Module import (`robotsmith.tasks`) | ✅ |
| TASK_PRESETS 注册 (3 tasks) | ✅ `pick_cube`, `mug_in_bowl`, `stack_blocks` |
| PREDICATE_REGISTRY 注册 (3 predicates) | ✅ `object_above`, `object_in_container`, `stacked` |
| IK_STRATEGIES 注册 (2 strategies) | ✅ `pick`, `pick_and_place` (stub) |
| TaskSpec JSON roundtrip (`to_dict` → `from_dict`) | ✅ |
| `object_above` predicate: lifted 0.10m > margin 0.05 | ✅ True |
| `object_above` predicate: lifted 0.02m < margin 0.05 | ✅ False |
| TrajectoryParams defaults match legacy collect_data.py | ✅ |

**Phase 2: 远端 GPU 端到端（待执行）**

```bash
python scripts/part2/collect_data.py --task pick_cube --n-episodes 20 --no-videos --save /output
```

| 指标 | 预期 | 实际 |
|------|------|------|
| 脚本无 crash | ✅ | （待执行） |
| 轨迹长度 = 135 frames | 135 | （待执行） |
| 20/20 episodes success | 100% | （待执行） |
| predicate vs heuristic 一致 | ✓ | （待执行） |

### 分析
（待 Phase 2 远端验证）

### 结论与 Next Step
（待 Phase 2 远端验证）
