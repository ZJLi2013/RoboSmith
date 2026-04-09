# Stage 1 实验记录

> 单物体 pick-cube (vision-only)，Franka + Genesis，目标 unseen 80%+。
> 实验计划见 [stage-1.md](stage-1.md)。
>
> 全部实验均为 **vision-only**（无 `--add-goal`，9D state + images），对齐 Stage 2 视觉泛化设定。

---

## 环境

| 项目 | 值 |
|------|-----|
| GPU | |
| ROCm / CUDA | |
| Genesis | |
| LeRobot | |
| SmolVLA weights | `lerobot/smolvla_base` |
| Python | |

## 实验结果汇总

| Exp | 数据管线 | Episodes | 训练 Steps | Batch | Unseen (seed=99) | Training (seed=42) | 备注 |
|-----|---------|:--------:|:----------:|:-----:|:----------------:|:-----------------:|------|
| B0 | 开环 IK (vision-only) | 100 | 2000 | 4 | | | V6a 复现 (~40%), ~1.6 epochs |
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
| 成功 episodes | /100 |
| 采集耗时 | |
| 数据集大小 | |

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
| Final loss | |
| 训练耗时 | |
| Peak VRAM | |

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
| 99 (unseen) | /10 | % |
| 42 (training) | /10 | % |

V6a 参考值：unseen 4/10 = 40%, training 0/10 = 0%

### 现象

（复现结果记录）

### 分析

（观察到的现象、失败 case 特征等）

### Next Step 判断

- [ ] 复现成功（unseen ~40%, training ~0%）→ 管线正确，进入 E1
- [ ] 复现失败 → 排查迁移 bug，不进入 E1

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
