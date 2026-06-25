# AMD Advance AI Day — Interaction-Ready 合成数据 Live Workshop

> **主标题**：Real2Sim for Embodied Data Generation
> **副标题**：*Generating Interaction-Ready Data for Long-Horizon Articulated Manipulation*
>
> 形式：**现场跑的 Jupyter notebook**（不是 slides），沿用上一场已验证的交付方式
> （notebooks.amd.com + ROCm 容器 + 按 GPU 架构分路径）。时长 **≤ 50 min**。
> 硬件：**CDNA3（MI300/MI325）+ RDNA3.5（W7900）**。
> 范围：**只到 interaction-ready 数据生成为止**——本场**不做** SmolVLA 训练 / 闭环评测
> （那条 gen→train→eval 链路上一场已覆盖）。
> 能力事实以 [design.md](design.md) / [grasping.md](grasping.md) /
> [features/drawer_feature.md](features/drawer_feature.md) / [runtime.md](runtime.md) 为准。

---

## 0. 一句话定位

> **上次我们在 AMD 上生成了"能抓东西"的数据；这次生成"能和世界交互"的数据。**

本场只讲一件事：**interaction-ready 数据怎么诞生**——以一个长程铰接体序列
（`open→pick→place→close` 抽屉任务）为载体，覆盖 real2sim 资产 → 声明式任务 →
无碰撞规划 → 段级重锚 / 接触相位 → 带完整交互语义的 LeRobot 数据集，全程现场跑、全程许可干净。

叙事弧：**缺口 → real2sim 资产 → CAP 声明 → rocRobo 引擎 → drawer e2e + 数据 → 改 CAP 重生成**。

---

## 1. 相对上一场的增量（为什么还要这场）

上一场 [Robot_synthetic_data_generation_workshop](https://github.com/PhysicalAI-AIM/Robot_synthetic_data_generation_workshop)
已证明：**SDG → SmolVLA 训练 → 闭环评测整条链路能在 AMD ROCm 上现场跑通**。但那条链路的任务是
**Franka 抓一个红方块**——单步 rigid pick，交互本身是平凡的。

本场不重讲训练/评测，只聚焦一个台阶：**能不能在 AMD 上生成"交互丰富"的数据**。

| 维度 | 上一场（cube pick） | 这一场（interaction-ready data gen） |
|---|---|---|
| 任务 | 单步抓取红方块 | **长程 `open→pick→place→close`** |
| 物体 | 纯刚体 | **铰接体（抽屉，URDF 关节）** |
| 运动 | IK + 直线 | **rocRobo 无碰撞规划 + 段级重锚 + 接触段受控直线** |
| 数据语义 | state/action 9D + 图像 | 同上 **+ 关节开度进 obs、段级重锚、接触相位** |
| 作者侧 | 脚本 | **CAP：~15 行 Python 声明一个交互任务** |

---

## 2. Notebook 大纲（≤50 min，gen-only，cell 级）

| 段 | 内容 | live / 预烤 | 估时 |
|---|---|---|---|
| 0 | 环境 & GPU 探测（沿用上一场）+ 拉 RoboSmith 资产 / 抓取权重（镜像已缓存） | live | 2' |
| 1 | **Hook**：对照上一场 cube pick，抛出 interaction 缺口（放上一场视频引出主题） | 讲 + 视频 | 3' |
| 2 | **Real2Sim 资产入口**：交互任务从哪来——rocRecon 真实物体→可仿真资产（本场用 built-in objaverse 资产兜底），drawer URDF 关节/接触面是交互前提 | 讲（+可选视频） | 4' |
| 3 | **CAP 授权**：打开 `scenarios/pick_place_into_drawer.py`，~15 行读懂分层（scene / task / success / skill intent） | live | 7' |
| 4 | **rocRobo：为什么交互需要无碰撞规划**——从简单 pick 的 IK+直线，到铰接/容器/长程必须避障+接触段受控 | 讲 | 5' |
| 5 | **现场生成交互 episode**（`run_scenario.sh`：rocRobo 避障 + 段级重锚 + 学习抓取候选） | **live（hero）** | 13' |
| 6 | **打开产出**：episode 视频 + LeRobot dataset，逐项指"交互语义"（关节开度进 obs、长程多段、接触相位、按物体实时位姿重锚） | live | 8' |
| 7 | **改 CAP 一行 → 重生成变体**（换目标格 / 换物体），展示"声明式交互数据工厂" | live | 5' |
| 8 | 小结：什么让它 interaction-ready + AMD/ROCm 工程故事（rocRobo 无碰撞规划 + Genesis 物理 on ROCm） | 讲 | 3' |

> 合计 ~50 min。若现场 gen 偏慢，段 7 降级成"放预生成变体视频"。

---

## 3. 各段要点（落地讲法）

### §2 Real2Sim 资产入口（接住主标题）
- 交互数据的前提是**有可交互的 sim 资产**：真实物体/场景 → 网格 / URDF（关节、接触面）→ 进入可声明的 scenario。
- 资产管线走 **[rocRecon](https://github.com/ZJLi2013/RocRecon)**（TRELLIS·Hunyuan3D / gsplat，资产级 real2sim 已开源 release）。
- **当前实际**：本场用 **built-in objaverse 资产**兜底（drawer + die），rocRecon 作为"真实→资产"的上游入口讲清来源即可，**不必现场实跑重建**。明确边界，不夸大。

### §3 CAP（声明式创建交互场景）
- 核心信息：**CAP 把"任务意图"和"执行细节"解耦**——作者只声明 *scene / task / success / skill intent*，底层自动编译成规划 + 接触执行。
- drawer + die 从这一段起作为**贯穿全场的 running example**。
- 卖点：**不同参与者各编辑各的 scenario**，远胜一个写死的 demo——这是"数据工厂"的作者面。

### §4 rocRobo（注意切题，别讲成通用运动规划科普）
- 因果收紧：简单 pick = IK + 直线就够；**一旦有环境交互（铰接、容器、长程），必须无碰撞规划 + 接触段受控**，才需要 rocRobo。
- rocRobo 基于 PyRoki，**license 干净，跑在 ROCm 上**——本场算力头牌。
- 抓取那步：learned 候选按可行性筛选，**叙事中性，不 spotlight GraspGen**（NV license，且非本场增量）。

### §6 打开产出（payoff，这才是 interaction-ready 的证据）
- 把 **episode 视频 + LeRobot dataset** 摊开，逐项指：
  - **关节开度进 obs**（抽屉开合状态可观测）
  - **长程多段**（一个 episode 含 open/pick/place/close 多相位）
  - **接触相位标注**
  - **按物体实时位姿段级重锚**
- 把"交互语义"从文字变成可见数据。

### §7 闭环（首尾呼应 §3 CAP）
- 改 CAP 一行（换目标格 / 换物体）→ 重生成新 episode，证明它是**声明式交互数据工厂**而非一次性 demo。

---

## 4. 落地要点（点到为止）

- **runtime**：e2e 比 cube pick 重得多（学习抓取 + 无碰撞规划 + 多段接触），现场只生成 **1–3 条**。两卡差异：W7900 现场实跑；MI300/325 render 走 CPU（~4× 慢）只跑 1 条示意，其余看预生成（规划仍在 ROCm 上）。
- **镜像**：抓取权重 + rocRobo serve + Genesis/kernel cache + 预生成 episode + objaverse 资产**全预烤进镜像**，gfx942 用旧栈镜像（见 [runtime.md](runtime.md)）。不含 train/eval。
- **AMD 卖点（许可干净）**：rocRecon（资产）/ rocRobo（无碰撞规划）/ Genesis（物理渲染）全跑 ROCm，全程无 NV-licensed 组件；抓取用 learned 候选但**叙事中性、不点名 GraspGen**。每个 demo 挂 AMD 数字（卡型 / ROCm 版本 / per-ep 耗时 / 成功率）。

**已定**：≤50 min；CDNA3 + RDNA3.5；只到数据生成（无 train/eval）；资产入口走 rocRecon（当前 built-in objaverse）。



## notebook 