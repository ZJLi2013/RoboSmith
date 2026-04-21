# Part 2 实验记录 — Data Engine

> TaskSpec-driven 多任务 IK 数采、SmolVLA 训练验证、I/O 性能优化。
> 实验计划见 [part2.md](part2.md)。

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

将 `collect_data.py` 中 hardcoded pick-cube 逻辑重构为 TaskSpec + PREDICATE_REGISTRY + IK_STRATEGIES 三层抽象，MI300X 20ep 验证功能等价性。

**实现**：`robotsmith/tasks/` 模块 — `task_spec.py`, `predicates.py`, `ik_strategies.py`, `presets.py`

**结果: 20/20 = 100%，与旧版完全等价**

| 指标 | 值 |
|------|:---:|
| 成功率 | **20/20 = 100%** |
| 轨迹长度 | 135 frames（与旧版一致） |
| predicate vs heuristic | 全部一致 (`pred=✓`) |
| 平均 lift 高度 | ~0.1625m（阈值 0.05m） |
| 耗时 | 502s / 20ep ≈ 25s/ep |

---

## S1.5: PickAndPlaceStrategy — place_cube 端到端验证

`PickAndPlaceStrategy`（8 阶段：approach → descend → grasp → lift → transport → place-descend → release → retreat），MI300X 20ep 验证。

**结果: 20/20 = 100% 成功率**

| 指标 | 值 |
|------|:---:|
| 成功率 | **20/20 = 100%** |
| 轨迹长度 | 225 frames |
| xy_err | 0.0037 ~ 0.0060m (avg ~4.7mm) |
| 放置精度 | avg 4.7mm（threshold 60mm），IK transport + place 精度远超需求 |
| 耗时 | 777s / 20ep ≈ 39s/ep |

纯开环 IK 对 pick_and_place 完全足够，place_z=0.15m 安全余量充足，release 后 cube 平稳落地无弹跳。

---

## S1.6: StackStrategy — 3-block 堆叠端到端验证

`StackStrategy`（3 轮 `PickAndPlaceStrategy`，每轮 place_z 递增 0.04m），MI300X 20ep 验证。

**结果: 18/20 = 90% 成功率**

| 指标 | 值 |
|------|:---:|
| 成功率 | **18/20 = 90%** |
| 轨迹长度 | 675 frames (3 × 225) |
| 成功 block_zs | `[0.019, 0.058~0.059, 0.097~0.098]` |
| 堆叠精度 | ~0.039m 间距（理论 0.04m），IK 放置精度 < 1mm |
| 失败模式 | ep 13, 19: round 1 pick 失败（block_red 未拾起），非堆叠对齐问题 |

**Phase 1 "抓握三件套" 全部验证通过：**

| 任务 | 成功率 | 轨迹长度 |
|------|:---:|:---:|
| `pick_cube` | 100% (20/20) | 135 frames |
| `place_cube` | 100% (20/20) | 225 frames |
| `stack_blocks` | 90% (18/20) | 675 frames |

Next Step: DART 噪声集成（`--dart-sigma`），从 pick_cube 开始验证。

---

## S2: Action Space + Camera 对齐

### S2.1 EE Delta Action Space

Action space 从 9D joint position 切换到 7D EE delta，observation state 从 9D 切换到 8D EE state。
详见 [study.md — §1.5](study.md#15-action-space-选型ee-delta-vs-joint-position)。

| | 旧 (retired) | 新 (默认) |
|---|---|---|
| action | 9D joint position `[j1..j7, f1, f2]` | 7D EE delta `[Δx, Δy, Δz, Δrx, Δry, Δrz, grip]` |
| state | 9D joint position | 8D `[eef_pos3, axangle3, gripper2]` |

MI300X 验证: 2ep pick_cube, 100% 成功率, action pos-delta norms ∈ [0.000006, 0.009]m (30fps).

### S2.2 Wrist Camera (eye-in-hand)

Camera 从固定 side cam 切换到 eye-in-hand wrist cam，attached to hand link。
参数基于 D040 验证配置（workshop overhead+wrist 实验 2.1x 优于 overhead+side）。

| Camera | 参数 |
|--------|------|
| overhead (固定) | `pos=(0.55, 0.55, 0.55)`, `lookat=(0.55, 0, 0.10)`, `fov=45` |
| wrist (eye-in-hand) | `pos=(0.05, 0, -0.08)`, `lookat=(0, 0, 0.10)`, `up=(0, 0, -1)`, `fov=65`, hand link |

MI300X 验证: benchmark smoke test + collect_data 2ep, 均通过。

### S2.3 vla-eval Benchmark Plugin

`RoboSmithBenchmark` 实现 vla-eval `Benchmark` ABC，MI300X smoke test 通过:
- Genesis scene 构建 + Franka + cube + overhead + wrist cameras: ✅
- EE delta action → IK → joint control pipeline: ✅
- 8D EE state observation: ✅
- Predicate-based success detection: ✅

---

## VLA 兼容性分析

Action/obs space 已与主流 VLA (Pi0, StarVLA, OpenVLA) 对齐:

| 维度 | LIBERO (主流 VLA 训练源) | RoboSmith |
|------|---|---|
| action | 7D EE delta | 7D EE delta ✅ |
| state | 8D `[eef_pos3, axangle3, gripper2]` | 8D `[eef_pos3, axangle3, gripper2]` ✅ |
| camera | overhead + wrist (eye-in-hand) | overhead + wrist (eye-in-hand) ✅ |
| 数据格式 | LeRobot | LeRobot ✅ |

Cross-sim distribution gap (渲染风格、物理引擎差异) 仍然存在，但 fine-tune 路径下模型会适应。
StarVLA 在 SimplerEnv 跨 sim 迁移可达 64.6%，RoboSmith pick_cube (简单任务) 预期 zero-shot 非 0%。
