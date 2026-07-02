# RoboSmith

中文 | [English](README-en.md)

**Synthetic Data Generation for Physical AI / Robotics on AMD ROCm.**

RoboSmith 是面向具身**交互操作**的**合成数据基础设施(Data Infra,而非 VLA 训练框架)**:你用几行代码声明一个长程、带关节交互的任务,它负责搭场景、规划抓取、执行避障专家轨迹,并导出可直接训练的 LeRobot 数据集 + 视频。它**站在开源社区的肩膀上**(Genesis 仿真、LeRobot 数据格式、Articraft 关节资产、PyRoki 运动学……),把这些组件在 AMD ROCm 上端到端打通——把单张 MI300 上的一个真实物体,变成长程、可交互的操作数据。

它是 ROCm-native 机器人平台里**已落地的三根支柱之一**(SDG 一环):

| 能力支柱 | 范围 | 引擎 | 状态 |
| --- | --- | --- | --- |
| 运动 / 动力学 | 无碰撞 IK + 避障 trajopt + 段级路由 | **rocRobo** | ✅ landed |
| SDG 合成数据 | 资产 → 场景 → 专家轨迹 → LeRobot 导出 | **RoboSmith** | ✅ landed |
| real2sim | 真实/生成物体 → 可仿真资产 | **rocRecon** | ✅ landed |

<table align="center">
  <tr>
    <td width="50%">
      <video src="https://github.com/ZJLi2013/RoboSmith/raw/main/videos/pick_place_into_drawer_ep000_up.mp4" controls muted loop width="100%"></video>
    </td>
    <td width="50%">
      <video src="https://github.com/ZJLi2013/RoboSmith/raw/main/videos/pick_place_onto_supporter_ep000_up.mp4" controls muted loop width="100%"></video>
    </td>
  </tr>
  <tr>
    <td align="center"><b>抽屉 · 长程关节任务</b><br>open → pick → place → close</td>
    <td align="center"><b>双层支架 · 侧插避障</b><br>从上层取骰子,水平插入下层(垂直下压会被上层挡住)</td>
  </tr>
</table>

## 为什么需要交互数据

Physical AI 的瓶颈不是像素,而是**交互**:接触丰富、多步、带关节的行为数据——恰恰是「视觉质量优先」的合成管线难以覆盖的部分。

- **简单抓取**(抓起一个方块)便宜但信息量低:单步、纯刚体,机械臂几乎不与环境交互。
- **长程交互**(开抽屉 → 放东西 → 关抽屉)才是稀缺、真正有价值的一档:多步、带关节、必须避障。真机采集慢、贵、不安全,也无法从网上爬到「1000 条开抽屉放物体」的演示。

SDG 的价值直接:在仿真里制造演示,物理 / 随机化 / 标签全部可控且免费。RoboSmith 想补的,是 **ROCm 生态内的这块空白**——用开源组件、在 AMD 上把整条链路端到端跑通。

## 能生成什么(asset × action × predicate)

用一段 Python 描述「桌上放什么、做哪几步、怎么算成功」,RoboSmith 就产出一条带视频的连续 episode + LeRobot 数据集 + 成功/失败诊断。作为合成数据生成器,它本质是「给定资产的 affordance,自动展开可生成的 (task, action, predicate)」。当前覆盖两类**可操作**资产:

| 资产类别 | affordance(解锁条件) | 语义动作 → 底层原语 | 成功谓词 |
| --- | --- | --- | --- |
| **rigid 刚体** | 可抓表面(mesh/bbox + upright + metric_scale) | pick / place → learned grasp (GraspGen) | object_in_container / object_above / stacked / objects_aligned |
| **articulated 关节** | task_joints + handles(轻量语义标注) | open / close → drag_handle(棱柱直线,同轴反向) | joint_opened / joint_closed |

- **声明式任务定义(Code-as-Policy)**:用一套小型 API 写任务(成功条件支持 且/或/非 组合),声明「做什么」,引擎负责「怎么做」,不必触碰底层 IK 或轨迹生成。
- **段级运动路由**:自由段(接近/搬运)走 rocRobo 无碰撞规划,接触段(拖把手/下压/侧插)走受控直线,抓/放只动手指——是否避障取决于该段是不是接触段。
- **资产来源可叠加**:内置 **Objaverse**(当前默认)+ **rocRecon** real2sim 重建(文本/图像/真实物体)+ **Articraft** 关节资产生成(图像 + 文本)。

示例场景见 [`scenarios/pick_place_into_drawer.py`](scenarios/pick_place_into_drawer.py),写法见 [`scenarios/README.md`](scenarios/README.md)。

## 快速开始

```bash
pip install -e .

# 多步长程任务(开/关抽屉 e2e,canonical 启动器,自带 rocRobo 避障 sidecar)
bash scripts/setup/run_scenario.sh

# 内层入口:单个 scenario
python scripts/datagen/run_generated_scenario.py \
  --scenario scenarios/pick_place_into_drawer.py --smoke

# 静态布局出图(不跑 rollout)
python scripts/datagen/snapshot_scenario.py \
  --scenario scenarios/pick_place_into_drawer.py
```

### 运行前置

数据生产依赖三样**仓库外**的运行时(各自单独安装/授权,不随本仓库分发):

- **ROCm 运行镜像**:Genesis 物理仿真,按 GPU 架构选镜像,见 [`docker/README.md`](docker/README.md)(MI300/gfx942 用 `docker/Dockerfile.gfx942`)。
- **rocRobo 运动后端**:无碰撞 IK + 避障 trajopt,来自 [rocRobo](https://github.com/ZJLi2013/rocRobo),以 sidecar 形式运行(license-friendly,PyRoki/JAX)。
- **learned grasp 运行时(GraspGen)**:抓取候选模型 + 权重(**NVIDIA research/eval 许可,需自备,不包含在本仓库**)。

> 注:JAX(rocRobo)与 PyTorch(Genesis/GraspGen)在同一 ROCm 进程内不易共存,故运动后端以独立容器 sidecar 形式运行,通过 `docker exec` 跨边界驱动。

## 代码结构

```text
robotsmith/
├── assets/            # Asset / AssetMetadata / AssetLibrary(exact-id),内置资产 + catalog
├── cap/               # 声明式任务定义层(scene/task/success/skill intent)
├── scenes/            # SceneConfig / ObjectPlacement、upright pose、Genesis loader
├── tasks/             # TaskSpec、谓词、任务预设
├── scenario_runtime/  # scenario 加载 / 物化 / 段级闭环 runner
├── grasp/             # grasp planner(learned)+ 可行性 rerank
├── motion/            # 运动规划 + rocRobo 避障后端接入
├── execution/         # executor:把技能展开成关节轨迹
├── skills/            # 技能原语(pick/place/open/close)+ registry
├── sim/               # Genesis SimEnv + Franka + LeRobot recorder
└── rollout/           # rollout 编排、episode 布局、视频导出
```

## 路线图与限制

路线图为**四条纵向能力轴 + 一条横向平台轴**:形态/embodiment(单臂 → 双臂 → 人形)、资产物理(rigid → **articulated(当前)** → deformable → cable)、长程编排、闭环评测(最高优先待建),以及把 sim×motion 桥接与 rollout 收敛成稳定 API。

当前主要限制:

1. 每个 `pick` 依赖 **GraspGen**(NVIDIA research/eval 许可,非商用):构建时从上游拉取、打进本地镜像,不随本仓库分发。抓取模型已隔离在单一 wrapper 后,便于替换为开源/可商用抓取器。
2. **pick 目前对碰撞不敏感**:空桌面稳定,但物体旁若有静态障碍(隔板/柜体),自由段会直穿——让 pick 的自由段也避障是路线图最高优先级。
3. **真实配置下的侧插尚不完全可靠**:孤立场景能成,但真实「上层取物 → 下层放置」时,插入段仍可能规划失败、退化为直线 fallback。

## 参考

- 📝 技术博客:[RoboSmith — 面向具身交互操作的 ROCm-native 合成数据管线](https://andyluo7.github.io/rocm/amd/mi300x/robotics/embodiedai/sdg/2026/07/01/robosmith-rocm-native-synthetic-data-pipeline/)
- [rocRobo](https://github.com/ZJLi2013/rocRobo) — 运动求解(无碰撞 IK + 避障 trajopt)
- [RocRecon](https://github.com/ZJLi2013/RocRecon) — real2sim 资产生成(prompt/图像 → sim-ready URDF)
- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) — GPU 加速物理仿真
- [LeRobot](https://github.com/huggingface/lerobot) — 数据集格式与 VLA 训练基础设施
- [Articraft](https://github.com/mattzh72/articraft) — 图像 + 文本 → 关节资产
- [PyRoki](https://github.com/chungmin99/pyroki) — 运动学(rocRobo 基础)
- [Objaverse](https://objaverse.allenai.org/) — 3D 物体数据集
- [GraspGen](https://github.com/NVlabs/GraspGen) — learned grasping(NVlabs · research/eval 许可)

## License

Apache-2.0,见 [LICENSE](LICENSE)。
