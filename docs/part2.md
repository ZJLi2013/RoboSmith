# Part 2：Data Engine + Eval — done

> TaskSpec + 多任务 IK solver + composable predicates 数采框架，pick / place / stack 三类任务端到端验证通过。
> Eval 通过 vla-eval-harness benchmark plugin 提供，不自建 eval engine。

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

## Action Space 迁移

> 详细分析见 [study.md — §1.5 Action Space 选型](study.md#15-action-space-选型ee-delta-vs-joint-position)。

**默认 action space 切换为 EE delta (7D)**，与主流 VLA（Pi0, StarVLA, OpenVLA, GR00T）对齐。
Joint position (9D) 观测和动作已 **retired** — 不再作为 LeRobot dataset 输出格式。

| | 旧 (retired) | 新 (默认) |
|---|---|---|
| action | 9D joint position `[j1..j7, f1, f2]` | 7D EE delta `[Δx, Δy, Δz, Δrx, Δry, Δrz, grip]` |
| state | 9D joint position | 8D `[eef_pos3, axangle3, gripper2]` |

## 遗留项

| 项目 | 说明 | 归属 |
|------|------|------|
| DART 噪声增强 (`--dart-sigma`) | IK solver 内置参数，待集成 | Roadmap Phase 3 |
| 场景模式数采 (`--scene`) | 场景加载 OK，pick Z 偏移需适配 | 工程优化 |

## Eval: vla-eval Benchmark Plugin

`RoboSmithBenchmark` (`robotsmith/eval/benchmark.py`) 实现 vla-eval `Benchmark` ABC，将 Genesis scene 作为 benchmark plugin：

- Action: 7D EE delta → IK → joint control
- Observation: 8D EE state + overhead + wrist (eye-in-hand) images
- 10+ VLA models 通过 vla-eval 自动可用（Pi0, StarVLA, OpenVLA, GR00T...）
- MI300X smoke test 通过

## 代码管线

```
scripts/part2/
├── collect_data.py          # IK 数采 (--task pick_cube/place_cube/stack_blocks)
├── snapshot_scene.py        # 场景布局截图验证
├── train_smolvla.py         # SmolVLA fine-tune (数据质量验证)
└── test_benchmark.py        # vla-eval benchmark smoke test

robotsmith/eval/
└── benchmark.py             # RoboSmithBenchmark — vla-eval Benchmark ABC
```

## 关键参数

| 参数 | 值 |
|------|:---:|
| Cube X range | 0.4~0.7 |
| Cube Y range | -0.2~0.2 |
| FPS | 30 |
| Table size | 1.2m x 0.8m |
| Franka base | 桌面上 (z = table_surface) |
