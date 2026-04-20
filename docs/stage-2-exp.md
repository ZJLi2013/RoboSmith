# Stage 2 实验记录

> 多物体泛化：gen2sim 物品变体 → SmolVLA post-training → unseen 物品泛化。
> 实验计划见 [stage-2.md](stage-2.md)。
>
> 前置条件：Stage 1 unseen ≥ 80%（单物体 pick-cube vision-only baseline）。

---

## 环境

| 项目 | 值 |
|------|-----|
| GPU | AMD Instinct MI300X |
| Docker | `genesis-amd-official-lerobot:rocm643-v043-np1` |
| Genesis | 0.4.3 |
| SmolVLA | `lerobot/smolvla_base` (450M) |
| 3D Gen | TRELLIS.2-4B |
| Node | `banff-cyxtera-s71-4.ctr.dcgpu` |

## 物品清单

3 类 × 5 变体 = 15 个。每类 3 train + 2 test。

| 类别 | Variant | 来源 | T2I Prompt | URDF 状态 | Split |
|------|---------|------|-----------|:---------:|:-----:|
| mug | tall_ceramic | TRELLIS.2 | "a tall narrow ceramic coffee mug with handle, beige" | 📋 | train |
| mug | short_metal | TRELLIS.2 | "a short wide metallic travel mug with handle, silver" | 📋 | train |
| mug | handleless_white | TRELLIS.2 | "a handleless straight white tea cup, minimalist" | 📋 | train |
| mug | wide_glass | TRELLIS.2 | "a wide-mouth short glass mug, transparent blue" | 📋 | test |
| mug | tiny_red | TRELLIS.2 | "a tiny espresso cup with small handle, bright red" | 📋 | test |
| bottle | slim_green | TRELLIS.2 | "a slim tall green glass bottle" | 📋 | train |
| bottle | fat_brown | TRELLIS.2 | "a fat short brown ceramic bottle" | 📋 | train |
| bottle | ribbed_clear | TRELLIS.2 | "a ribbed clear plastic water bottle" | 📋 | train |
| bottle | square_white | TRELLIS.2 | "a square cross-section white bottle" | 📋 | test |
| bottle | mini_blue | TRELLIS.2 | "a mini blue glass bottle" | 📋 | test |
| bowl | deep_wooden | TRELLIS.2 | "a deep wooden salad bowl" | 📋 | train |
| bowl | shallow_white | TRELLIS.2 | "a shallow white ceramic bowl" | 📋 | train |
| bowl | medium_blue | TRELLIS.2 | "a medium depth blue ceramic bowl" | 📋 | train |
| bowl | flat_ceramic | TRELLIS.2 | "a very flat ceramic plate-like bowl" | 📋 | test |
| bowl | large_metal | TRELLIS.2 | "a large stainless steel mixing bowl" | 📋 | test |

## 实验结果汇总

| Exp | 训练数据 | Train Variants | Test Variants | Train 成功率 | Test 成功率 | 备注 |
|-----|---------|:--------------:|:-------------:|:-----------:|:----------:|------|
| S2-E1 | mug × 3 var × 100ep | mug train (3) | mug test (2) | | | intra-category baseline |
| S2-E2 | mug+bottle × 6 var × 100ep | mug+bottle train (6) | mug+bottle test (4) | | | 2-category |
| S2-E3 | all 3 cat × 9 var × 100ep | all train (9) | all test (6) | | | cross-category |
| S2-Ablation | varying # variants per cat | varying | held-out | | | scaling curve |

---

## 准备工作：物品资产生成

### Step 1: TRELLIS.2 生成 15 个物品

```bash
# 每个变体：T2I prompt → reference image → TRELLIS.2 → GLB → URDF
# 预计 ~5 min/个 × 15 = ~75 min

robotsmith generate "a tall narrow ceramic coffee mug with handle, beige" --quality fast --name mug_tall_ceramic
robotsmith generate "a short wide metallic travel mug with handle, silver" --quality fast --name mug_short_metal
# ... (see full list above)
```

| 指标 | 值 |
|------|-----|
| 总生成数 | /15 |
| 成功 URDF 转换 | /15 |
| 生成总耗时 | |

### Step 2: 物品仿真验证

每个生成的物品需在 Genesis 中验证：
1. 碰撞体合理（不穿透桌面）
2. 质量/摩擦合理（可被 Franka gripper 抓起）
3. 尺寸在 gripper 可抓范围内（~3-8cm 宽度）

```python
# 逐个加载并测试 IK grasp
for asset in asset_library.list("mug", "bottle", "bowl"):
    scene = build_scene(franka, table, asset)
    success = test_ik_grasp(scene, asset)
    print(f"{asset.name}: {'OK' if success else 'FAIL'}")
```

| 物品 | 碰撞体 | 可抓取 | 备注 |
|------|:------:|:------:|------|
| mug_tall_ceramic | | | |
| mug_short_metal | | | |
| ... | | | |

---

## S2-E1：Intra-Category 泛化（Mug Only）

**目的**：验证 gen2sim 同类变体训练是否让 policy 在 unseen 同类物品上泛化。

### 数据采集

```bash
# 3 个 train mug 变体，每个 100 episodes
python scripts/part2/collect_data.py --n-episodes 100 --object mug_tall_ceramic \
    --repo-id local/stage2-mug-train --seed 42

python scripts/part2/collect_data.py --n-episodes 100 --object mug_short_metal \
    --repo-id local/stage2-mug-train --seed 43 --append

python scripts/part2/collect_data.py --n-episodes 100 --object mug_handleless_white \
    --repo-id local/stage2-mug-train --seed 44 --append
```

| 指标 | 值 |
|------|-----|
| 训练 episodes | /300 |
| 采集成功率 | |
| 采集耗时 | |

### 训练

```bash
python scripts/part2/train_smolvla.py \
    --dataset-id local/stage2-mug-train \
    --n-steps 10000 \
    --batch-size 16 \
    --save-dir outputs/stage2/S2-E1
```

| 指标 | 值 |
|------|-----|
| Final loss | |
| 训练耗时 | |
| Epochs | |

### 评估

| 物品 | Split | 成功 / 总数 | 成功率 |
|------|:-----:|:----------:|:-----:|
| mug_tall_ceramic | train | /10 | |
| mug_short_metal | train | /10 | |
| mug_handleless_white | train | /10 | |
| **mug_wide_glass** | **test** | /10 | |
| **mug_tiny_red** | **test** | /10 | |

### 现象


### 分析


### Next Step 判断

- [ ] Test mug 成功率 > 50% → intra-category 泛化有效，进入 S2-E2
- [ ] Test mug 成功率 < 30% → 检查物品资产质量、抓取策略适配

---

## S2-E2：Two-Category 泛化（Mug + Bottle）

**目的**：验证跨两类训练是否提升泛化能力。

### 数据采集

```bash
# 6 个 train 变体 (3 mug + 3 bottle)，每个 100 episodes = 600 ep
# 复用 S2-E1 的 mug 数据 + 新采 bottle 数据
```

### 训练

```bash
python scripts/part2/train_smolvla.py \
    --dataset-id local/stage2-mug-bottle-train \
    --n-steps 15000 \
    --batch-size 16 \
    --save-dir outputs/stage2/S2-E2
```

### 评估

| 物品 | Category | Split | 成功 / 总数 | 成功率 |
|------|----------|:-----:|:----------:|:-----:|
| mug_tall_ceramic | mug | train | /10 | |
| mug_short_metal | mug | train | /10 | |
| mug_handleless_white | mug | train | /10 | |
| **mug_wide_glass** | mug | **test** | /10 | |
| **mug_tiny_red** | mug | **test** | /10 | |
| bottle_slim_green | bottle | train | /10 | |
| bottle_fat_brown | bottle | train | /10 | |
| bottle_ribbed_clear | bottle | train | /10 | |
| **bottle_square_white** | bottle | **test** | /10 | |
| **bottle_mini_blue** | bottle | **test** | /10 | |

### 现象


### 分析


### Next Step 判断

- [ ] Test 成功率 > 50% → 进入 S2-E3
- [ ] Test 成功率提升 vs S2-E1 → 多类别训练有正向收益

---

## S2-E3：Cross-Category 泛化（All 3 Categories）

**目的**：验证 mug + bottle 训练的 policy 在 unseen 类别 (bowl) 上是否有零样本泛化。

### 数据采集

```bash
# 9 个 train 变体 (3 mug + 3 bottle + 3 bowl)，每个 100 episodes = 900 ep
```

### 训练

```bash
python scripts/part2/train_smolvla.py \
    --dataset-id local/stage2-all-train \
    --n-steps 20000 \
    --batch-size 32 \
    --save-dir outputs/stage2/S2-E3
```

### 评估

| 物品 | Category | Split | 成功 / 总数 | 成功率 |
|------|----------|:-----:|:----------:|:-----:|
| (6 test variants from mug + bottle) | mug/bottle | test | | |
| **bowl_flat_ceramic** | bowl | **test** | /10 | |
| **bowl_large_metal** | bowl | **test** | /10 | |
| bowl_deep_wooden | bowl | train | /10 | |
| bowl_shallow_white | bowl | train | /10 | |
| bowl_medium_blue | bowl | train | /10 | |

### 现象


### 分析


### Next Step 判断

- [ ] Bowl test 成功率 > 30% → cross-category 泛化有效
- [ ] Test 成功率 overall > 50% → Stage 2 目标达成

---

## S2-Ablation：Variant Scaling Curve

**目的**：量化 gen2sim 物品变体数量 vs VLA 泛化能力的关系。

### 实验设计

固定 mug 类别，变化训练变体数：

| 条件 | Train Variants | Episodes | Steps | Test |
|------|:--------------:|:--------:|:-----:|------|
| 1-var | mug_tall_ceramic (1) | 100 | 5000 | mug test (2) |
| 2-var | mug_tall + mug_short (2) | 200 | 7500 | mug test (2) |
| 3-var | mug_tall + short + handleless (3) | 300 | 10000 | mug test (2) |

### 结果

| # Train Variants | Test 成功率 | 提升 |
|:----------------:|:----------:|:----:|
| 1 | | — |
| 2 | | |
| 3 | | |

### 分析


---

## 结论

| 实验 | Core Question | 结果 | 结论 |
|------|--------------|------|------|
| S2-E1 | 同类 unseen 变体泛化？ | | |
| S2-E2 | 多类训练提升泛化？ | | |
| S2-E3 | 跨类别零样本泛化？ | | |
| Ablation | 变体数 ↑ → 泛化 ↑？ | | |

**Stage 2 目标是否达成？**

- [ ] Intra-category test > 50%
- [ ] Cross-category test > 30%
- [ ] Scaling 趋势可见

**进入 Stage 3 的条件：**

Stage 2 证明 gen2sim 物品多样性 → VLA 泛化有效，且 SmolVLA 在单步 pick-place 上已达上限 → 升级 StarVLA 做多步 stacking (Stage 3)。
