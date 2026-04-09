# Stage 2：多物体泛化

> Franka 固定臂 pick-place，通过 gen2sim 物品变体实现跨形态/跨类别泛化。
>
> 前置：[Stage 1](../lerobot_from_zero_to_expert/)（单物体 + 位姿泛化）已在 lerobot 项目中验证通过。

---

## 目标

验证核心命题：**RoboSmith gen2sim 生成的物品变体 → post-training 数据 → VLA 在 unseen 物品上泛化**。

成功标准：
- Intra-category（同类 unseen 变体）成功率 > 50%
- Cross-category（整类 unseen）成功率 > 30%
- 变体数量 ↑ → 泛化能力 ↑（scaling 趋势可见）

## 为什么是 Stage 2

| 阶段 | 新增维度 | RoboSmith 角色 |
|:---:|---------|---------------|
| 1 | 单物体位姿泛化 | 不需要 — 用固定 cube 即可 |
| **2** | **多物体形态/外观泛化** | **核心** — gen2sim 生成变体是独特能力 |
| 3 | 多步推理 | 复用 Stage 2 资产 |
| 4 | 长程规划 | 复用 Stage 2 资产 |

Stage 2 是 RoboSmith gen2sim 能力第一次产生实质价值的阶段。

## 物品设计：3 类 × 5 变体 = 15 个

### 类别选择

| 类别 | Affordance 类型 | 抓取难度 | 选择理由 |
|------|----------------|:-------:|---------|
| **mug** | 侧抓 (handle) / 顶抓 | 中 | 形态变化丰富（高矮胖瘦、有无把手） |
| **bottle** | 侧抓 / 顶抓 | 中 | 尺寸变化大（细长 vs 粗短） |
| **bowl** | 顶抓 / 边缘抓 | 低-中 | 与 mug 形态差异大，测试跨类别泛化 |

三类覆盖了 same-affordance（mug ↔ bottle 都可顶抓）和 cross-affordance（bowl 需不同策略）两种泛化。

### 变体设计

每类 5 个变体，3 train + 2 test：

```
mug (杯子)
├── train: tall_ceramic    — 高窄，陶瓷质感，有把手
├── train: short_metal     — 矮胖，金属质感，有把手
├── train: handleless_white — 无把手，白色，直筒
├── test:  wide_glass      — 宽口，玻璃质感，矮         ← intra-category test
└── test:  tiny_red        — 迷你，红色，有把手          ← intra-category test

bottle (瓶子)
├── train: slim_green      — 细长，绿色
├── train: fat_brown       — 粗短，棕色
├── train: ribbed_clear    — 带纹路，透明感
├── test:  square_white    — 方形截面，白色              ← intra-category test
└── test:  mini_blue       — 迷你，蓝色                  ← intra-category test

bowl (碗)
├── train: deep_wooden     — 深碗，木质
├── train: shallow_white   — 浅碗，白色
├── train: medium_blue     — 中等深度，蓝色
├── test:  flat_ceramic    — 极浅/盘状，陶瓷             ← intra-category test
└── test:  large_metal     — 大号，金属质感               ← intra-category test
```

### 变体生成方式

| 来源 | 数量 | 用途 |
|------|:---:|------|
| Objaverse 策划（已有） | 3 | 每类 1 个作为 anchor |
| TRELLIS.2 gen2sim | 12 | 每类 4 个变体（不同 T2I prompt → 不同形态） |
| **总计** | **15** | |

T2I prompt 示例（生成 mug 变体）：

```python
variants = [
    "a tall narrow ceramic coffee mug with handle, beige",
    "a short wide metallic travel mug with handle, silver",
    "a handleless straight white tea cup, minimalist",
    "a wide-mouth short glass mug, transparent blue",
    "a tiny espresso cup with small handle, bright red",
]
```

## 实验设计

### Experiment 1：Intra-Category 泛化（核心）

```
训练: mug_v1, mug_v2, mug_v3 × 100 episodes = 300 ep
测试: mug_v4, mug_v5 (unseen 形态)
任务: pick-place (桌面随机位置 → 固定目标区)
指标: 成功率 (grasp success + place accuracy)
期望: > 50%
```

**证明**：同类别不同形态的训练数据能让 policy 学到 shape-invariant 的抓取策略。

### Experiment 2：Cross-Category 泛化（加分）

```
训练: mug(3 var) + bottle(3 var) × 100 ep = 600 ep
测试: bowl (整类 unseen, 5 variants)
任务: pick-place
指标: 成功率
期望: > 30%
```

**证明**：跨类别训练数据能让 policy 学到通用操作策略。

### Experiment 3：Variant Scaling 消融（故事线）

```
对比:
  - 1 variant/category  × 100 ep → unseen 成功率
  - 3 variants/category × 100 ep → unseen 成功率
  - 5 variants/category × 100 ep → unseen 成功率

期望: 成功率随变体数上升
```

**证明**：gen2sim 物品多样性 → VLA 泛化能力的 scaling 关系，这是 RoboSmith 的核心价值 story。

## 实现步骤

### Step 1：生成物品变体（~75 min GPU）

```bash
# 为每个变体生成 T2I 参考图 → TRELLIS.2 → URDF
robotsmith generate "tall narrow ceramic mug with handle" --quality fast
robotsmith generate "short wide metallic travel mug"      --quality fast
# ... 共 12 个生成任务（3 类 × 4 变体）
```

### Step 2：仿真环境搭建

```python
# Genesis 场景：Franka + 桌面 + 单个目标物品
# 随机化：物品位置/朝向 (position_range 已在 SceneConfig 实现)
# 相机：固定 wrist cam + overhead cam
```

### Step 3：轨迹采集（~2.5 h）

```python
# IK scripted pick-place（复用 lerobot 已有 Franka IK）
# 每个训练变体 100 episodes
# 训练集：9 variants × 100 ep = 900 ep
# 输出：LeRobot v3.0 数据集格式
```

### Step 4：策略训练（~30 min）

```bash
# SmolVLA fine-tune（Stage 1 已验证的 baseline）
python train_smolvla.py --dataset ./stage2_data --n-steps 3000 --batch-size 4
```

### Step 5：评估（~1 h）

```python
# 逐个变体评估，记录 per-variant 成功率
# 重点关注 train vs test split 的成功率差异
# 输出：成功率表 + scaling 曲线
```

## 资源估算

| 环节 | 单项耗时 | 总计 |
|------|---------|------|
| TRELLIS.2 生成 12 变体 | ~5 min/个 | ~60 min |
| URDF 转换 + 入库 | ~1 min/个 | ~12 min |
| IK 轨迹采集 (900 ep) | — | ~2.5 h |
| SmolVLA 训练 | — | ~30 min |
| 评估 (3 experiments) | — | ~1 h |
| **总计** | | **~5 h (单卡 MI308X)** |

## 预期产出

1. **15 个 sim-ready 物品资产**（3 Objaverse + 12 TRELLIS.2 生成）
2. **900 episode LeRobot 数据集**
3. **3 组实验结果**：intra-category / cross-category / scaling
4. **核心结论**：gen2sim 物品变体数量 vs VLA 泛化能力的量化关系

## Affordance 细分（后续可选）

Stage 2 基础版只做 pick-place。如果时间允许，可以进一步区分：

| 子实验 | 含义 | 物品 |
|--------|------|------|
| Same-affordance | mug ↔ bottle（都可顶抓） | mug + bottle |
| Cross-affordance | mug → bowl（不同抓取策略） | mug + bowl |

这可以回答一个更细的问题：policy 泛化的瓶颈是视觉差异还是操作策略差异？

## VLA 模型选型

各阶段对 policy 模型的能力需求不同：

| 阶段 | 任务复杂度 | 关键瓶颈 | 推荐模型 |
|:---:|----------|---------|---------|
| 1 | 单步 pick-place，单物体 | 基础视觉运动映射 | SmolVLA (450M) ✅ 已验证 |
| **2** | **单步 pick-place，多物体** | **视觉编码器跨形态泛化** | **SmolVLA (450M)** — pretrained ViT 够用 |
| 3 | 多步 stacking | 状态跟踪 + 序列决策 | [StarVLA](https://github.com/starVLA/starVLA) (Qwen3-VL 4B) |
| 4 | 长程任务 | 长序列规划 + 条件分支 | StarVLA (Qwen3-VL 4B) |

**升级策略：SmolVLA → StarVLA**

```
Stage 1-2: SmolVLA (450M)          快速迭代 baseline，单卡 6min 训练
               │
               ▼ Stage 3 升级
Stage 3-4: StarVLA (Qwen3-VL 4B)  模块化框架，4 种 action head 可选
               │                    LIBERO avg 96.5%，30K steps 收敛
               │
               ├── StarVLA-OFT    MLP 连续输出，最稳定 (96.6%)
               ├── StarVLA-GR00T  双系统 VLM+FM，最强 (96.5%)
               ├── StarVLA-FAST   离散 token，推理快
               └── StarVLA-PI     Flow-Matching，连续控制
```

> **为什么 Stage 3/4 选 StarVLA？**
> - **LIBERO pipeline 开箱即用**：数据下载→训练→评估全有现成脚本
> - **4 种 action head 可对比**：一套代码内切换 FAST/OFT/PI/GR00T，不需要分别跑 OpenVLA、π0.5
> - **训练效率高**：30K steps (~10 epochs) 收敛，OpenVLA-OFT 需 175K steps (223 epochs)，成本差 ~6 倍
> - **LeRobot v3.0 数据格式**：与 RoboSmith 轨迹采集无缝对接
> - **Qwen3-VL 4B backbone**：比 OpenVLA 7B 更轻量，比 SmolVLA 450M 容量充足

Stage 2 使用 SmolVLA，数据格式为 LeRobot v3.0，后续升级 StarVLA 时数据可直接复用。

## 依赖

| 组件 | 来源 | 状态 |
|------|------|:----:|
| TRELLIS.2 3D 生成 | RoboSmith Part 1 | ✅ |
| AssetLibrary 管理 | RoboSmith Part 1 | ✅ |
| Franka IK 轨迹生成 | lerobot 项目复用 | ✅ |
| Genesis 仿真 | Part 2 新增 | 📋 |
| LeRobot 数据写入 | lerobot 项目复用 | ✅ |
| SmolVLA 训练 | lerobot 项目复用 | ✅ |
