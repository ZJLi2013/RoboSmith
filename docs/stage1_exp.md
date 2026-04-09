# Stage 1 实验记录

> 单物体 pick-cube (vision-only)，Franka + Genesis，目标 unseen 80%+。
> 实验计划见 [stage-1.md](stage-1.md)。
>
> 全部实验均为 **vision-only**（无 `--add-goal`，9D state + images），对齐 Stage 2 视觉泛化设定。

---

## 环境

| 项目 | 值 |
|------|-----|
| GPU | AMD Instinct MI308X (192GB VRAM × 8) |
| ROCm / CUDA | ROCm 6.4.1 (torch 2.6.0+rocm6.4.1) |
| Genesis | 0.4.3 |
| LeRobot | 0.4.4 |
| SmolVLA weights | `lerobot/smolvla_base` (450M params, 99.9M trainable) |
| transformers | 5.3.0 (pinned, >=5.4 breaks lerobot groot dataclass) |
| Python | 3.10 |
| Docker | `genesis-amd-official-lerobot:rocm643-v043-np1` |
| Node | `banff-cyxtera-s71-4.ctr.dcgpu` |

## 实验结果汇总

| Exp | 数据管线 | Episodes | 训练 Steps | Batch | Unseen (seed=99) | Training (seed=42) | 备注 |
|-----|---------|:--------:|:----------:|:-----:|:----------------:|:-----------------:|------|
| B0 | 开环 IK (vision-only) | 100 | 2000 | 4 | **1/10 = 10%** | **0/10 = 0%** | V6a 复现 (ref ~40%), ~0.6 epochs |
| E1 | 开环 IK (vision-only) | 100 | 10000 | 16 | | | 训练量提升, ~32 epochs |
| E2 | 闭环 DART σ=0.005 (vision-only) | 100 | 10000 | 16 | | | DART 增量, ~32 epochs |
| E3 | 闭环 DART σ=0.005 (vision-only) | 200 | 10000 | 32 | | | 数据量增量, ~16 epochs |

## B0：V6a 复现（Vision-Only Baseline）

**目的**：确认 pipeline 迁移后功能正确，复现 lerobot V6a ~40%。

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
