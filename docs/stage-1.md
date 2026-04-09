# Stage 1：单物体 + 位姿泛化（Re-baseline）

> Franka 固定臂 pick cube，位姿随机化，目标 unseen 成功率 **80%+**。
>
> 基于 [lerobot_from_zero_to_expert/10_franka](../../lerobot_from_zero_to_expert/10_franka/) 已有经验，
> 代码迁移至 robotsmith 独立运行，数据管线升级为闭环 DART。

---

## 背景：Stage 1 现状

lerobot 项目中已验证 Stage 1 基础可行性，但成绩不充分：

| 实验 | 数据 | 训练 | Unseen 成功率 | 问题 |
|------|------|------|:---:|------|
| V5 SmolVLA | 100 ep, 开环 IK | 2000 steps, batch 4 | **60%** | 开环数据 + 训练不充分 |
| V6a SmolVLA (vision-only) | 100 ep, 无 goal state | 2000 steps, batch 4 | **40%** | 严重 under-training |

**结论**：60% 不足以支撑 Stage 2 的多物体泛化实验。需要先将单物体 pick 做到 **80%+**，
验证数据管线和训练配方的可靠性后再扩展。

## 核心改进：闭环 DART 数据管线

### 为什么开环完美数据不够

VLA 部署时 policy 预测不可能完美跟踪训练轨迹，会产生 drift。
如果训练数据全是"完美轨迹"，policy 从未见过偏离状态下的 recovery action → **covariate shift**。

在 pick-cube 实验中问题不明显（轨迹短 ~50 steps、Franka 7-DOF 冗余、任务简单），
但要达到 80%+ 并为 Stage 2 打基础，需要更好的数据。

### 闭环 DART 实现

```python
# 从当前状态重新求解，不是回放预规划 waypoint
for each step:
    state = observe()
    target = solve_ik(current_qpos=state, goal_pose=next_waypoint)
    perturbed = target + noise(σ=0.005~0.01)  # 匹配部署时真实 drift
    execute(perturbed)
    record(state, target)  # label 是"从这里应该去哪"
```

关键改进（相比 lerobot P6 实验 σ=0.0005）：
- **扰动幅度要大**：σ=0.005~0.01（匹配部署时 ~0.009 rad/step 的真实误差）
- **Action label 从扰动后状态重新 IK 求解**，不是原始规划 waypoint
- 数据天然包含 recovery supervision

### 数据管线适用边界

| Level | 任务复杂度 | 数据管线 | 适用阶段 |
|:---:|----------|---------|:---:|
| 1 | 单物体/多物体 pick、不同 grasp 姿态 | **闭环 DART** (IK + noise) | Stage 1-2 |
| 2 | pick-and-place、stacking、精确放置 | Task-space planning + 闭环 IK | Stage 3 |
| 3 | insertion、assembly、deformable | **RL Expert → DAgger** 蒸馏 | Stage 4 |

> Stage 1-2 用闭环 DART 足够。Stage 3 开始碰到 IK supervisor 天花板
> （不理解接触力、不知道松手时机），需要 RL expert 替代 IK 作为 supervisor。

## 实验计划

### 改进轴（依优先级）

| # | 改进 | 预期收益 | 成本 |
|---|------|---------|------|
| **A** | 闭环 DART 数据 (σ=0.005~0.01) | 解决 covariate shift | 低 — 已有 `07_closed_loop_ik_data.py` |
| **B** | 增加训练量：20K steps, batch 64 | V6a doc 已指出 2K steps under-training | 低 — 改参数 |
| **C** | 增加数据量：200+ episodes | 更多位姿覆盖 | 中 — ~30min GPU |
| **D** | Vision-only (去掉 goal state) | 对齐 Stage 2 的视觉泛化设定 | 无 — 数据生成时选择 |

### Baseline 实验（复现 V5）

```
B0: 复现 V5 baseline
  数据: 100 ep, 开环 IK (01_franka_pick_data.py)
  训练: SmolVLA, 2000 steps, batch 4
  评估: seed 99, 10 episodes
  期望: ~60% (复现确认管线正确)
```

### 改进实验

```
E1: 闭环 DART 数据
  数据: 100 ep, 闭环 IK + σ=0.005 (07_closed_loop_ik_data.py)
  训练: SmolVLA, 2000 steps, batch 4
  评估: seed 99, 10 episodes
  对比: B0 → E1，衡量 DART 数据的增量收益

E2: 增加训练量
  数据: 同 E1
  训练: SmolVLA, 10000 steps, batch 16
  评估: seed 99, 10 episodes
  对比: E1 → E2，衡量训练量的增量收益

E3: 增加数据量
  数据: 200 ep, 闭环 IK + σ=0.005
  训练: SmolVLA, 10000 steps, batch 16
  评估: seed 99, 10 episodes
  对比: E2 → E3，衡量数据量的增量收益

E4: Vision-only
  数据: 同 E3, 去掉 goal state
  训练: SmolVLA, 10000 steps, batch 16
  评估: seed 99, 10 episodes
  对比: E3 → E4，衡量去掉 goal state 的影响
  目标: 这是 Stage 2 的直接前置条件
```

### 成功标准

| 指标 | B0 (baseline) | 目标 |
|------|:---:|:---:|
| Unseen 成功率 (state+goal) | ~60% | **80%+** |
| Unseen 成功率 (vision-only) | ~40% | **70%+** |
| Training 位置成功率 | ~80% | **90%+** |

达到目标后进入 Stage 2。

## 代码管线

从 `lerobot_from_zero_to_expert/10_franka` 迁移至 robotsmith 独立运行：

```
robotsmith/pipeline/
├── collect_data.py          # ← 10_franka/scripts/01_franka_pick_data.py (开环 IK)
├── collect_data_dart.py     # ← 10_franka/scripts/07_closed_loop_ik_data.py (闭环 DART)
├── train_smolvla.py         # ← 04_post_training/scripts/01_train_smolvla.py
├── eval_policy.py           # ← 10_franka/scripts/02_franka_eval.py
└── plot_loss.py             # ← 04_post_training/scripts/02_plot_loss_curve.py
```

### 一键运行

```bash
# Stage 1 完整管线
python pipeline/collect_data.py --n-episodes 100 --save   # 开环 baseline 数据
python pipeline/train_smolvla.py --dataset-id local/franka-pick-100ep --n-steps 2000
python pipeline/eval_policy.py --policy-type smolvla --checkpoint outputs/smolvla/final

# 闭环 DART 数据
python pipeline/collect_data_dart.py --n-episodes 100 --perturb-sigma 0.005 --save

# 增量训练
python pipeline/train_smolvla.py --dataset-id local/franka-dart-100ep --n-steps 10000 --batch-size 16
```

## 关键参数参考（from lerobot 实验）

### 数据采集

| 参数 | V5 baseline | 推荐 |
|------|:-----------:|:----:|
| Episodes | 100 | 100 (B0) → 200 (E3) |
| Cube X range | 0.4~0.7 | 同 |
| Cube Y range | -0.2~0.2 | 同 |
| FPS | 30 | 同 |
| DART σ | — (开环) | 0.005~0.01 |

### SmolVLA 训练

| 参数 | V5 baseline | 推荐 |
|------|:-----------:|:----:|
| Pretrained | `lerobot/smolvla_base` | 同 |
| Steps | 2000 | 2000 (B0) → 10000 (E2+) |
| Batch size | 4 | 4 (B0) → 16 (E2+) |
| LR | SmolVLAConfig default | 同 |
| chunk_size | 50 | 同 |
| freeze_vision_encoder | True | 同 |
| train_expert_only | True | 同 |

### 评估

| 参数 | 值 |
|------|:---:|
| Seed (unseen) | 99 |
| Seed (training sanity) | 42 |
| Episodes per eval | 10 |
| Max steps | 200 |
| Success: cube lifted | z > 0.05m above table |

## 依赖

| 组件 | 状态 |
|------|:----:|
| Genesis (`genesis-world`) | 需安装 |
| LeRobot (`lerobot`) | 需安装 |
| SmolVLA weights (`lerobot/smolvla_base`) | HuggingFace 自动下载 |
| Franka MJCF (`panda.xml`) | Genesis 内置 |
| torch (CUDA/ROCm) | 需 GPU |
