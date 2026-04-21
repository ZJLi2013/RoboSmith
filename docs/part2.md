# Part 2：Data Engine — done

> TaskSpec + 多任务 IK solver + composable predicates 数采框架，pick / place / stack 三类任务端到端验证通过。

---

## 成果

| 任务 | IK Strategy | 成功率 | 轨迹长度 | 平台 |
|------|-------------|:---:|:---:|:---:|
| `pick_cube` | `pick` | **100%** (20/20) | 135 frames | MI300X |
| `place_cube` | `pick_and_place` | **100%** (20/20) | 225 frames | MI300X |
| `stack_blocks` | `stack` (N=3) | **90%** (18/20) | 675 frames | MI300X |

## 已验证能力

1. **TaskSpec-driven 数采** — `collect_data.py --task <name>` 自动 dispatch 到对应 IK strategy + predicate
2. **Composable predicates** — `object_above`, `object_in_container`, `stacked` 三个谓词覆盖 grasp / place / stack 判定
3. **多任务 IK solver** — `PickStrategy`, `PickAndPlaceStrategy`, `StackStrategy` 三种策略，精度 < 1mm
4. **LeRobot 数据集输出** — 直接兼容 VLA 训练（SmolVLA 已验证）
5. **Hybrid workflow** — RDNA4 data gen + MI300X train/eval 异构方案可行

## 实验记录

详见 [part2-exp.md](part2-exp.md)。

| # | 实验 | 状态 |
|---|------|:---:|
| S1.1 | 开环 IK 100ep | ✅ |
| S1.4 | TaskSpec + pick_cube 20ep | ✅ 100% |
| S1.5 | PickAndPlaceStrategy place_cube 20ep | ✅ 100% |
| S1.6 | StackStrategy stack_blocks 20ep | ✅ 90% |

## 遗留项（不阻塞 Stage 1 收尾）

| 项目 | 说明 | 归属 |
|------|------|------|
| DART 噪声增强 (`--dart-sigma`) | IK solver 内置参数，待集成 | Roadmap Phase 3 |
| 场景模式数采 (`--scene`) | 场景加载 OK，pick Z 偏移需适配 | 工程优化 |

## 代码管线

```
scripts/part2/
├── collect_data.py          # IK 数采 (--task pick_cube/place_cube/stack_blocks)
├── snapshot_scene.py        # 场景布局截图验证
├── train_smolvla.py         # SmolVLA fine-tune (数据质量验证)
└── eval_policy.py           # 闭环策略评估
```

## 关键参数

| 参数 | 值 |
|------|:---:|
| Cube X range | 0.4~0.7 |
| Cube Y range | -0.2~0.2 |
| FPS | 30 |
| Table size | 1.2m x 0.8m |
| Franka base | 桌面上 (z = table_surface) |
