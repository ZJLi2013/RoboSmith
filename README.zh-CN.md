# RoboSmith

[English](README.md) | 中文

**在真实场景的仿真孪生里，评测 agentic 机器人与 VLA；运行在 AMD ROCm 上。**

v0.1.0（2026-10-20）是合成数据管线；此后的主线是评测。跑 scenario 仍会写 LeRobot 数据集。

## Goal

标准 benchmark 用的是精选场景。RoboSmith 要回答：agent 或 VLA 进了*你的*场景，还能不能完成任务。

1. **世界：** 一段真实场景的视频变成 Genesis 里可交互的孪生：物体能静止，可移动物体是能抓的资产。
2. **任务：** agent 在这个场景上写任务，由 RoboSmith 自带的基本 agentic 操作证明每道可解。
3. **评测：** 被测方通过一个不绑定任何框架的接口执行。判定读仿真状态，成功判定树对被测方隐藏；每个失败给出从未达成或达成后丢失的里程碑。

结果是孪生里测得的；没做过孪生与真机对照，不代表真机表现。不在范围内：精选 benchmark 套件、策略训练、sim2real 研究。

**MVP：** 一个真实桌面场景、3–5 道任务、一个被测方、一份报告：

```text
RoboSmith eval  scene: <你的桌面>（视频 → 孪生，几何残差中位数 x cm）  evaluatee: <agentic 基线>
task                 certified  success  failures
把杯子放到托盘上      10/10      6/10     never: grasped ×3 · lost: on ×1
把笔放进笔筒          10/10      2/10     never: grasped ×5 · never: inside ×3
```

## 现状

RoboSmith 自带的操作是六动作组合：plan 文件的 `actions` 直接执行，自由段由关节空间 RRT-Connect 规划，每个执行点在物理执行前
过同一道执行门，不可行就带原因零步拒绝（[设计](docs/manipulation.md)）。公开验收场景族上，`main`、playground seed 0–9、单卡 MI300X：

| 场景族 | 约束 | certified |
|---|---|---|
| F0 空桌面 | — | 10/10 |
| F1 贴墙抓取 | 物体一侧有比它高 1 cm 的矮墙 | 10/10 |
| F2 越障搬运 | 物体与放置垫之间有 20 cm 高墙 | 10/10 |

certified = 任务成立，且逐物理步监测的实际执行零违规接触。开发期全程封存的留出族在冻结实现上读了一次：test seed 100–109，
F5 10/10 certified（[验收](docs/acceptance.md)、[结果](docs/manipulation.md#结果)）。[`scenarios/`](scenarios/) 里的其他
scenario 能在这条链路上加载，但在它上面没有读数。

每个场景族在当前链路上录了一个 episode（点图看 mp4）。录制不做逐步接触监测，不是 certified 读数，表里的数字来自 `certify`。
F2 没放录像：这个机位下 20 cm 高墙挡住了 die 和抓取。

| F0 空桌面 · playground seed 0 | F1 贴墙抓取 · playground seed 0 | F5 双矮墙（留出族） · test seed 100 |
|---|---|---|
| [![F0 空桌面，playground seed 0](videos/f0_open_seed0.gif)](videos/f0_open_seed0.mp4) | [![F1 贴墙抓取，playground seed 0](videos/f1_wall_gap_seed0.gif)](videos/f1_wall_gap_seed0.mp4) | [![F5 双矮墙，test seed 100](videos/f5_clutter_seed100.gif)](videos/f5_clutter_seed100.mp4) |

MVP 还要做的，按顺序：

1. **基本 agentic 操作。** agent 写步骤表，现有技能执行，每步判定；计划层失败交回 agent，物理层失败走修复环。不做跨 episode
   记忆（需要时用 RPent 这类外部框架）。pick / place 已改为过执行门的六动作组合，没有碰撞盲段，冻结后的留出读数
   F5 10/10；还差规则物体在杂乱桌面上抓得稳。
2. **原语覆盖**常见室内桌面任务约八成：press、push、rotate、转动关节开合。
3. **真实场景：** 真实视频重建成能静止的孪生，开口、把手由 agent 标注、仿真验证。
4. **评测层：** 任务包、接口、never / lost 归因、报告。

## 现在怎么工作

![RoboSmith 流程](docs/images/pipeline-zh.svg)

- **资产：** 由 agent 生成（默认写 Blender 脚本），LLM 评审在仿真里取证后准入，按哈希发布到 Hugging Face。
- **Scenario：** YAML，写物体、关系、对执行方隐藏的成功判定树，以及 playground / test 两段 seed。
- **操作：** pick / place 是六个 meta action（`sample`、`move_to`、`move_linear`、`gripper`、`attach`、`detach`）的组合，
  由 plan 文件的 `actions` 给出；自由段由关节空间 RRT-Connect 规划，整条轨迹逐点过执行门，过了才执行。pick 用几何候选在仿真里试抓，
  全失败时 agent 修复环介入；抽屉开合沿用拖拽、不过门。
- **验证：** 每一步和整个任务用同一组谓词读仿真状态判定，可离线复判。

主要限制：单台 Franka、一张桌子；手指位置控制（约 0.2 kg 以上会滑脱）；MI300X 上是软件渲染，还不适合评测 VLA；一个进程一个场景。
细节见 [`docs/design.md`](docs/design.md)（含文档地图）。

MVP 之后：房间尺度场景与移动操作、外观保真度与 VLA、外部 agent 框架适配器、有真机时做一次孪生与真机对照、其他具身。

## 上手

[Quick start](docs/quick-start.md) · [设计](docs/design.md) · [调用链](docs/call-chain.md) · [编写 scenario](scenarios/README.md) ·
[技术博客](https://andyluo7.github.io/rocm/amd/mi300x/robotics/embodiedai/sdg/2026/07/01/robosmith-rocm-native-synthetic-data-pipeline/)（写于数据管线时期）

改动经 pull request 合入：[CONTRIBUTING.md](CONTRIBUTING.md)、[AI_POLICY.md](AI_POLICY.md)、[SECURITY.md](SECURITY.md)。License：Apache-2.0。
