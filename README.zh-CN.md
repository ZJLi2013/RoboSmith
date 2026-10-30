# RoboSmith

[English](README.md) | 中文

**在真实场景的仿真孪生里，评测 agentic 机器人与 VLA；运行在 AMD ROCm 上。**

v0.1.0（2026-10-20）是合成数据管线；此后的主线是评测，数据生成原样保留、不再加功能。

## Goal

标准 benchmark 已经很多，场景都是精挑细选的。下一步要回答的是：一个 agent 或 VLA 进了*你的*环境，还能不能完成任务。
RoboSmith 在仿真里回答这个问题：

1. **世界。** 一段真实场景的视频变成 Genesis 里可交互的孪生：物体能静止，可移动物体是独立、能抓的资产，几何与重建结果
   有量化比对。缺的物体由 agent 生成，经 LLM 评审准入。
2. **任务。** agent 在这个场景上写任务，每道都有可解证明：RoboSmith 自带的基本 agentic 操作在孪生里完成过它。
3. **评测。** 被测方（agentic 机器人、VLA 或任何控制器）通过一个不绑定任何框架的接口接入；RoboSmith 自带一个基本的
   agentic 基线。判定读仿真状态，成功判定树对被测方隐藏；每个失败都归到一个里程碑：从未达成，或达成后又丢失。

**说法边界。** 结果是在真实场景的孪生里测得的。还没做过孪生与真机的对照，所以不代表真机上的表现。

**不在范围内。** 精选的 benchmark 套件；策略训练；sim2real 研究。数据生成不再是产品方向：证明可解时录下的 episode
仍以 LeRobot 数据集保留。

### 最小 MVP

一个真实桌面场景、3–5 道任务、一个被测方、一份报告：

```text
RoboSmith eval  scene: <你的桌面>（视频 → 孪生，几何残差中位数 x cm）  evaluatee: <agentic 基线>
task                 certified  success  failures
把杯子放到托盘上      10/10      6/10     never: grasped ×3 · lost: on ×1
把笔放进笔筒          10/10      2/10     never: grasped ×5 · never: inside ×3
说法边界：孪生内结果，未做真机对照
```

MVP 成立的标准：读报告的人能看出被测方在这张桌子上做得怎样、败在哪一步，而且场景与任务是从一段视频出发、由 agent
搭出来的，没有人手调。

## 现状

评测层以下的几层已经存在，在精选场景上能跑。下面两段是 RoboSmith 自带的操作（将来的可解证明执行器），`main` 上 seed 0、
单卡 MI300X：

<table align="center">
  <tr>
    <td width="50%">
      <a href="videos/pick_place_onto_supporter_ep000_up.mp4"><img src="videos/pick_place_onto_supporter_ep000_up.gif" width="100%"></a>
    </td>
    <td width="50%">
      <a href="videos/pick_place_table_to_lower_shelf_ep000_up.mp4"><img src="videos/pick_place_table_to_lower_shelf_ep000_up.gif" width="100%"></a>
    </td>
  </tr>
  <tr>
    <td align="center"><b>双层支架 · 侧插</b><br>从上层取骰子，插入下层</td>
    <td align="center"><b>桌面到下层搁板</b><br>从桌上取骰子，插入下层搁板</td>
  </tr>
</table>

| Scenario（seed 0，当前 `main`） | 步骤 | 结果 |
|---|---|---|
| `pick_place_onto_supporter` | pick → place | ✅ 成功 |
| `pick_place_table_to_lower_shelf` | pick → place | ✅ 成功 |
| `pick_place_into_drawer` | open → pick → place → close | ❌ 规划放置时 rocRobo 服务出错，搬运退回不检查碰撞的直线，物体脱手 |
| `pick_one_of_three` | pick | ❌ 抓取失败（候选试抓时 IK 跳支被拒，或苹果滑脱） |
| `behavior_living_radio`、`office48_desk`、`office48_phone`（导入场景） | pick | ❌ 过不了静止检查：物体一放下就被顶开约 0.1 m 或顶高 6–15 cm；导入当时是能静止的 |

MVP 还缺的：

- **真实视频输入。** 重建链路（视频 → 参考 → agent 建房间 → 快照导入）只在带深度与位姿真值的 BEHAVIOR 仿真视频上跑过；
  换成真实视频，保真度改为与重建结果比对。
- **能静止的导入场景。** 三个导入场景导入时过了 2 s 沉降检查，现在过不了，原因还没查到。
- **导入场景上的任务。** 现有的只是单个 pick；要由 agent 针对场景自己的物体写多步任务，并由证明执行器完成。
- **评测层。** 不含成功判定树的任务包、被测方接口、基本 agentic 基线、never / lost 归因与报告。

## 现在怎么工作

![RoboSmith 流程：agentic 资产与场景喂给 agentic 操作](docs/images/pipeline-zh.svg)

- **资产。** 单个物体由 LLM 写 Blender 脚本生成（默认），也可走图生 3D 后端（TRELLIS.2、Hunyuan3D）再接 `mesh_to_urdf`，
  或把重建房间里的物体按快照导入、各自编成一个资产。资产要在当前构建上有 LLM 评审（自己写仿真探针）的通过记录、许可证
  允许再分发、且不与已准入资产重复，才能准入。构建文件按哈希发布到 Hugging Face 数据集，新 clone 能拉到完全一致的资产库。
- **Scenario。** 一个 YAML 文件：物体及其关系（`on`、`near`、`clear_of`、`facing`）和对执行方隐藏的成功判定树，旁边配一份
  teacher 的步骤计划。agent 按一句话请求和资产目录写出它（[`scenarios/AGENT_PROMPT.md`](scenarios/AGENT_PROMPT.md)），
  求解器按 seed 摆放物体，`split` 字段预留 playground 与 test 两段 seed。
- **操作。** `pick` 在仿真里逐个试几何抓取候选，执行第一个能抬起的；全部失败时，agent 提出修法，复核通过后写回资产。
  `place` 支持放上台面、侧插进搁板与放进抽屉；`open` / `close` 拖动棱柱关节。自由段由 [rocRobo](https://github.com/ZJLi2013/rocRobo)
  做带碰撞检查的规划（JAX，sidecar 容器）。
- **验证。** 放置后先检查物体是否静止；每一步按加载时绑定的谓词判（`grasped`、`on` / `inside`、`joint_opened` /
  `joint_closed`），与任务用的是同一组；失败的 episode 给出第一个失败的步骤和失败的判定树叶子，所有结论都能从记录的状态离线复判。

支持 MI300（gfx942）与 MI350（gfx950）。

## 限制

- 单台 Franka、一张桌子；关节只支持棱柱关节：没有转动关节的开关（门、盖子）、推按，也没有双臂任务。
- 手指是位置控制：约 0.2 kg 以上的物体会滑脱；扁平宽件没有候选族。
- 接触段是不检查碰撞的直线，`plan_motion` 规划不出时自由段也退回这样的直线（日志记为
  `collision-blind straight-line fallback`）；上面两段 demo 里都有这样的段。
- MI300X 上是软件渲染（llvmpipe），离照片级很远；要评测 VLA，先得补外观保真度。
- 一个进程只能建一个场景；新场景形状的第一次 `plan_motion` 要编译约 10 分钟。

按模块的细节见 [`docs/design-cn.md`](docs/design-cn.md#当前限制)。

## MVP 之后

按依赖排：

- 更多场景，再到房间尺度，和移动操作一起做。
- 外观保真度，之后才能评测 VLA。
- 外部 agent 框架（例如 RPent）的适配器，走同一个接口。
- 有真机时做一次小规模孪生与真机对照；之后说法才能超出孪生。
- 其他具身：双臂、人形。

## 上手

- [Quick start](docs/quick-start.md)：从一台全新 GPU 节点跑出第一条 episode。
- [设计](docs/design-cn.md)：数据流、模块、执行、验证、依赖的上游项目（[英文版](docs/design.md)）。
- [编写 scenario](scenarios/README.md)。
- 技术博客：[RoboSmith — 面向具身交互操作的 ROCm-native 合成数据管线](https://andyluo7.github.io/rocm/amd/mi300x/robotics/embodiedai/sdg/2026/07/01/robosmith-rocm-native-synthetic-data-pipeline/)
  （写于 RoboSmith 定位为数据管线时）。

## 参与贡献

所有改动经 pull request 合入，见 [CONTRIBUTING.md](CONTRIBUTING.md)、[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
与 [AI_POLICY.md](AI_POLICY.md)。安全问题按 [SECURITY.md](SECURITY.md) 私下报告。

## License

Apache-2.0，见 [LICENSE](LICENSE)。
