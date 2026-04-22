# Part 3：Irregular Object Grasping — 异形物体抓取数据生成

> **核心命题**：RoboSmith Part 2 的 IK 数采只能处理规则物品（cube/block），
> 对碗、杯子、瓶子、螺丝刀、玩具等不规则物品缺乏 grasp target 定义。
> 这是 RoboSmith 从 demo 升级到实用工具的关键瓶颈。
>
> **设计原则**：从架构层面解耦 "抓哪里"（Grasp Planning）和 "怎么到达"（Motion Execution），
> 使 data gen pipeline 能自然扩展到任意物体品类。
>
> 相关文档：
> - [design.md — Grasp Affordance Gap](design.md#next-stepgrasp-affordance-gap-与演进路线) — 技术分析
> - [study.md §8.1](study.md#81-内置资产详细规划) — 10 品类抓取挑战分析
> - [part2.md](part2.md) — Part 2 IK 数采现状

---

## 1. 问题诊断

### 1.1 Part 2 的 hardcoded 假设

| 假设 | 硬编码值 | 适用范围 |
|------|---------|---------|
| 垂直顶部抓取 | `grasp_quat=[0,1,0,0]`，只变 Z | cube, block |
| 固定抓取高度 | `grasp_z=0.135` | 高度 ~4cm 的小方块 |
| 固定 finger width | `open=0.04, closed=0.01` | 宽度 < 8cm 的物体 |
| 物体几何不参与规划 | 只用 XY 坐标，不读 mesh/size | 所有物体视为等价 |

### 1.2 架构根因：规划与执行耦合

当前 `ik_strategies.py` **同时承担两个职责**：

```
IKStrategy（当前）= Grasp Planning + Motion Execution
                    "从顶部抓"        "IK 求解 + lerp 插值"
                    ↑ hardcoded        ↑ 通用，应保留
```

`TrajectoryParams` 中 `grasp_quat`, `grasp_z`, `finger_open/closed` 是**抓取决策**（"用什么姿态抓"），
而 `approach_steps`, `_lerp()`, `solve_ik()` 是**运动执行**（"怎么到达那个 pose"）。
两者混在同一个类里，导致每支持一种新物体就要写一个新的 `XxxPickStrategy`。

### 1.3 为什么不规则物品无法处理

| 物品 | 问题 | 根因 |
|------|------|------|
| 碗 (bowl) | 直径 14cm，超 Franka 8cm 最大开口 | 需缩放或边缘抓取 |
| 杯子 (mug) | 需抓 handle，侧面接近 | `grasp_quat` 固定朝下 |
| 瓶子 (bottle) | 需抓瓶身中段，高重心 | `grasp_z` 不随物体变化 |
| 盘子 (plate) | 薄 + 平，需边缘 pinch | 垂直下降无法完成 |
| 玩具 (figurine) | 不规则凸包，无明显抓取面 | 需 grasp candidate sampling |

**一句话**：IK solver 只解决 "几何可达性"（给定 pose 求关节角），但真实抓取是 "语义 + 接触 + 几何" 的组合问题。规划与执行必须分离。

---

## 2. 业界 SDG 方法论

业界已基本不再依赖纯 IK rollout 做 SDG。核心转变：

```
旧范式: 人定义 IK target → 确定性轨迹 → 数据
新范式: 构建 "可产生行为分布的交互系统" → 数据
```

### 2.1 技术路线图谱

| 路线 | 核心思路 | 代表工作 | 与 IK 的关系 |
|------|---------|---------|-------------|
| **Grasp distribution** | 不问 "手到哪"，问 "哪些接触可行" | GPD, DexNet, AnyGrasp | 替代单点 IK target |
| **Policy-in-the-loop** | 让 policy 自己探索，记录成功/失败 | RL/BC policy rollout | IK 只做 seed/fallback |
| **Affordance-driven** | 不标注 pose，标注 affordance | ShapeGen, Where2Act | IK 按 affordance 参数化 |
| **Contact-rich** | 生成接触点/法线/力矩 | 触觉仿真, dexterous manip | 超越 position-level IK |
| **Teleop seed + sim augment** | 少量人操作 + sim 扩展 | MimicGen, RoboTurk | IK 做 augmentation |
| **Task-level SDG** | 多步任务而非单动作 | LIBERO, RoboLab | IK 是 subtask 执行器 |

### 2.2 关键 insight：IK 是执行层，不是规划层

所有路线中，**IK solver 始终存在，但角色是 "最后一公里执行"**：

```
Grasp distribution   → 选出 grasp pose → IK solver 到达
Policy-in-the-loop   → policy 输出 EE delta → IK solver 执行
Affordance-driven    → affordance → grasp params → IK solver 到达
```

> IK solver 不需要被替换，需要被正确定位：它是运动执行器，不是抓取规划器。

### 2.3 行业共识

> SDG ≠ 用 simulator 批量生成 "正确轨迹"
> SDG = 构建一个可以产生 "行为分布" 的交互系统

RoboSmith 当前处于最左端（单点 IK = 规划 + 执行），需要分离并向右演进。

---

## 3. 目标架构

### 3.1 两层分离：Grasp Planner + Motion Executor

```
                    ┌─────────────────────────────────────────────┐
                    │              Data Gen Pipeline               │
                    │                                              │
                    │  ┌──────────────────────────────────────┐   │
                    │  │  Grasp Planning Layer（新增）         │   │
                    │  │  "抓哪里？用什么姿态？"                │   │
                    │  │                                      │   │
                    │  │  GraspPlanner (ABC)                  │   │
                    │  │    ├─ TemplateGraspPlanner            │   │
                    │  │    │   (per-category 人工 template)   │   │
                    │  │    ├─ SamplerGraspPlanner             │   │
                    │  │    │   (mesh-based antipodal)         │   │
                    │  │    └─ LearnedGraspPlanner             │   │
                    │  │        (AnyGrasp / model)             │   │
                    │  │                                      │   │
                    │  │  输出: GraspPlan                      │   │
                    │  │    (grasp_pose, pre_grasp, retreat,   │   │
                    │  │     finger_width, quality)            │   │
                    │  └───────────────┬──────────────────────┘   │
                    │                  │                           │
                    │                  ▼                           │
                    │  ┌──────────────────────────────────────┐   │
                    │  │  Motion Execution Layer（重构自 Part 2）│   │
                    │  │  "怎么到达那个 pose？"                 │   │
                    │  │                                      │   │
                    │  │  MotionExecutor                      │   │
                    │  │    solve_ik(pose) → joint angles     │   │
                    │  │    interpolate(A→B, n_steps) → traj  │   │
                    │  │    compose_trajectory(GraspPlan)     │   │
                    │  │      → full joint-space trajectory   │   │
                    │  └───────────────┬──────────────────────┘   │
                    │                  │                           │
                    │                  ▼                           │
                    │           joint trajectory                   │
                    │                  │                           │
                    │     DataBackend.collect() 执行 + 录制         │
                    │       ├─ IK Scripted: 直接执行               │
                    │       ├─ DART: + noise + re-solve            │
                    │       └─ DAgger: policy rollout + relabel    │
                    │                  │                           │
                    │                  ▼                           │
                    │         LeRobot Dataset                      │
                    └─────────────────────────────────────────────┘
```

### 3.2 核心抽象

> `GraspPlan`, `GraspPlanner`, `MotionExecutor`, `MotionParams` 的完整定义和旧→新映射
> 已移入 [design.md — 核心抽象](design.md#grasp-planning-layer)。
>
> 要点：
> - `GraspPlan` 是两层之间的唯一契约 — planner 不关心 IK，executor 不关心物体语义
> - `MotionExecutor.pick()` 对所有物体通用，差异完全在 `GraspPlanner` 输出的 `GraspPlan` 中
> - 旧 `IKStrategy` 一对一映射：`PickStrategy` → `executor.pick()`，`PickAndPlaceStrategy` → `executor.pick_and_place()`
> - 旧 `StackStrategy` 不再是独立 class — 调用方循环 N 次 `pick_and_place()`

---

## 4. 代码重构计划

### 4.1 当前文件 → 重构后文件

```
robotsmith/tasks/                        # 当前
├── ik_strategies.py                     # TrajectoryParams + IKStrategy + Pick/PnP/Stack
├── task_spec.py                         # TaskSpec (含 ik_strategy 字段)
├── predicates.py                        # PREDICATE_REGISTRY (不变)
├── presets.py                           # TASK_PRESETS (不变)
└── __init__.py

                    ↓ refactor ↓

robotsmith/
├── grasp/                               # 新模块：Grasp Planning Layer
│   ├── __init__.py                      # GraspPlan, GraspPlanner
│   ├── plan.py                          # GraspPlan dataclass
│   ├── planner.py                       # GraspPlanner ABC
│   ├── template_planner.py              # TemplateGraspPlanner + GRASP_TEMPLATES registry
│   ├── sampler_planner.py               # SamplerGraspPlanner (Step 2)
│   └── templates/                       # per-category template 定义
│       └── registry.py                  # GRASP_TEMPLATES: dict[str, GraspTemplate]
│
├── motion/                              # 新模块：Motion Execution Layer (从 ik_strategies 提取)
│   ├── __init__.py                      # MotionExecutor, MotionParams
│   ├── executor.py                      # MotionExecutor (pick, pick_and_place)
│   └── params.py                        # MotionParams (纯运动时间参数)
│
├── tasks/                               # 保留，但简化
│   ├── task_spec.py                     # TaskSpec (删除 ik_strategy，新增 motion_type)
│   ├── predicates.py                    # 不变
│   ├── presets.py                       # 更新 presets，去掉 ik_strategy
│   └── __init__.py
│
├── scenes/                              # 不变
└── assets/                              # 不变
```

### 4.2 重构步骤（按依赖顺序）

**Step 0：向后兼容层**

在重构前，先让旧代码可以通过新接口调用。保留 `ik_strategies.py` 但标记 deprecated，
新代码优先使用 `grasp/` + `motion/` 模块。`collect_data.py` 保持可运行。

**Step 1：提取 GraspPlan + MotionParams**

从 `TrajectoryParams` 中分离：
- 抓取决策字段 → `GraspPlan`（`grasp_quat`, `grasp_z` 等）
- 运动时间字段 → `MotionParams`（`*_steps`）

```python
# 旧 TrajectoryParams 拆分为：
GraspPlan(grasp_pos, grasp_quat, pre_grasp_pos, retreat_pos, finger_open, finger_closed)
MotionParams(approach_steps=40, descend_steps=30, grasp_hold_steps=20, ...)
```

**Step 2：提取 MotionExecutor**

从 `PickStrategy`, `PickAndPlaceStrategy` 中提取通用运动逻辑：
- `_lerp()` → `MotionExecutor._interpolate()`
- `PickStrategy.plan()` → `MotionExecutor.pick(grasp_plan, ...)`
- `PickAndPlaceStrategy.plan()` → `MotionExecutor.pick_and_place(pick_plan, place_plan, ...)`
- `StackStrategy.plan()` → 调用方循环 `pick_and_place()`

**Step 3：引入 GraspPlanner ABC + TemplateGraspPlanner**

- `GraspPlanner.plan(asset, object_pos, object_quat)` → `list[GraspPlan]`
- `TemplateGraspPlanner`：从 `GRASP_TEMPLATES[asset.category]` 查找模板，生成 `GraspPlan`
- 为 block/cube 写第一个 template（等价于当前 hardcoded 行为），验证重构不破坏现有功能

**Step 4：更新 collect_data.py**

```python
# 旧
params = TrajectoryParams(grasp_z=args.grasp_z, ...)
strategy = IK_STRATEGIES[task_spec.ik_strategy]
traj = strategy.plan(target_pos, solve_ik, home, params, z_offset)

# 新
planner = TemplateGraspPlanner()
grasp_plan = planner.plan(asset, object_pos, object_quat, rng)[0]
executor = MotionExecutor()
traj = executor.pick(grasp_plan, solve_ik, home, motion_params)
```

**Step 5：更新 TaskSpec**

- `ik_strategy: str` → `motion_type: str`（`"pick"` / `"pick_and_place"`）
- 新增可选 `grasp_planner: str = "template"`（允许 TaskSpec 指定 planner 类型）
- 更新 `TASK_PRESETS`

**Step 6：清理**

- 删除 `ik_strategies.py`（或保留为 thin wrapper 调用 `motion/executor.py`）
- 删除 `IK_STRATEGIES` registry
- 更新 `robotsmith/tasks/__init__.py` 的导出

### 4.3 向后兼容保证

每个 step 完成后，**`collect_data.py --task pick_cube` 必须仍能运行且结果一致**。
测试方法：refactor 前后用同一 seed 运行 5 episodes，对比轨迹。

---

## 5. GraspTemplate 定义（Step 3 详细）

### 5.1 数据结构

```python
@dataclass
class GraspTemplate:
    """Per-category grasp template — 定义一次，同类资产复用。"""
    category: str
    grasp_type: str                    # "top_down", "side", "rim"
    approach_axis: np.ndarray          # unit vector: 接近方向 (object frame)
    grasp_offset: np.ndarray           # 抓取点相对物体中心偏移 (object frame)
    ee_quat: np.ndarray                # EE 姿态 (wxyz, world frame assuming upright)
    finger_open: float                 # 接近时 finger width (m)
    finger_closed: float               # 夹紧时 finger width (m)
    pre_grasp_distance: float = 0.10   # 沿 approach_axis 的 pre-grasp 偏移
    retreat_height: float = 0.15       # lift 后高度 (相对桌面)
    requires_scale: bool = False
    scale_range: tuple[float,float] = (1.0, 1.0)
```

### 5.2 10 品类模板

| # | Category | grasp_type | approach_axis | ee_quat | finger_open | 说明 |
|---|----------|-----------|:---:|:---:|:---:|------|
| 1 | block | top_down | `[0,0,-1]` ↓ | `[0,1,0,0]` | 0.04 | 当前行为，基线 |
| 2 | box | top_down | `[0,0,-1]` ↓ | `[0,1,0,0]` | 按 `size_cm` 调 | 同 block |
| 3 | fruit | top_down | `[0,0,-1]` ↓ | `[0,1,0,0]` | 0.03~0.05 | 小球体 |
| 4 | bowl | top_down | `[0,0,-1]` ↓ | `[0,1,0,0]` | 0.04 | `requires_scale=True, scale_range=(0.4,0.6)` |
| 5 | can | side | `[0,-1,0]` → | `[0.5,0.5,-0.5,0.5]` | 0.04 | EE 旋转 90°，抓中段 |
| 6 | bottle | side | `[0,-1,0]` → | `[0.5,0.5,-0.5,0.5]` | 0.04 | 抓瓶身中段 |
| 7 | mug | side | `[0,-1,0]` → | `[0.5,0.5,-0.5,0.5]` | 0.02 | `grasp_offset` 偏向 handle |
| 8 | plate | top_down | `[0,0,-1]` ↓ | `[0,1,0,0]` | 0.01 | `grasp_offset` 偏到 rim |
| 9 | lblock | top_down | `[0,0,-1]` ↓ | `[0,1,0,0]` | 0.04 | `grasp_offset` 偏向质心 |
| 10 | figurine | top_down | `[0,0,-1]` ↓ | `[0,1,0,0]` | 0.04 | fallback; 后续升级到 sampler |

### 5.3 TemplateGraspPlanner 逻辑

```python
class TemplateGraspPlanner(GraspPlanner):
    def plan(self, asset, object_pos, object_quat, rng):
        template = GRASP_TEMPLATES.get(self._match_category(asset))
        if template is None:
            template = GRASP_TEMPLATES["block"]  # fallback

        grasp_pos = object_pos + rotate(object_quat, template.grasp_offset)
        grasp_quat = template.ee_quat  # TODO: 后续根据 object_quat 旋转
        pre_grasp_pos = grasp_pos + template.approach_axis * template.pre_grasp_distance
        retreat_pos = np.array([grasp_pos[0], grasp_pos[1],
                                object_pos[2] + template.retreat_height])

        return [GraspPlan(
            grasp_pos=grasp_pos,
            grasp_quat=grasp_quat,
            pre_grasp_pos=pre_grasp_pos,
            pre_grasp_quat=grasp_quat,
            retreat_pos=retreat_pos,
            retreat_quat=grasp_quat,
            finger_open=template.finger_open,
            finger_closed=template.finger_closed,
            quality=1.0,
            metadata={"source": "template", "category": template.category},
        )]

    def _match_category(self, asset):
        """从 asset tags 匹配品类。"""
        for tag in asset.metadata.tags:
            if tag in GRASP_TEMPLATES:
                return tag
        return "block"
```

---

## 6. MotionExecutor 详细设计（Step 2 详细）

### 6.1 从 IKStrategy 提取什么

| 现有 IKStrategy 代码 | 提取到 | 说明 |
|---------------------|--------|------|
| `_lerp(a, b, n)` | `MotionExecutor._interpolate()` | 不变 |
| `PickStrategy.plan()` 中的 waypoint 序列 | `MotionExecutor.pick()` | 从 `GraspPlan` 读 pose，不再 hardcode |
| `PickAndPlaceStrategy.plan()` | `MotionExecutor.pick_and_place()` | 接收 `pick_plan` + `place_plan` |
| `StackStrategy.plan()` | 调用方循环 | 不再是独立 class |
| `solve_ik()` 调用 | 保留为 `MotionExecutor` 的依赖注入 | `solve_ik` 仍由 `collect_data.py` 创建并传入 |

### 6.2 MotionExecutor.pick() 伪代码

```python
def pick(self, plan: GraspPlan, solve_ik, home_qpos, params: MotionParams):
    q_home = home_qpos.copy()
    q_pre   = solve_ik(plan.pre_grasp_pos, plan.pre_grasp_quat, plan.finger_open)
    q_grasp = solve_ik(plan.grasp_pos,     plan.grasp_quat,     plan.finger_open)
    q_close = solve_ik(plan.grasp_pos,     plan.grasp_quat,     plan.finger_closed)
    q_lift  = solve_ik(plan.retreat_pos,   plan.retreat_quat,   plan.finger_closed)

    traj = []
    traj += self._interpolate(q_home,  q_pre,   params.approach_steps)
    traj += self._interpolate(q_pre,   q_grasp, params.descend_steps)
    traj += self._interpolate(q_grasp, q_close, params.grasp_hold_steps)
    traj += self._interpolate(q_close, q_lift,  params.lift_steps)
    traj += [q_lift.copy()] * params.lift_hold_steps
    return traj
```

注意：**这和当前 `PickStrategy.plan()` 结构完全一致**，唯一区别是 pose 来自 `GraspPlan` 而非 hardcoded `TrajectoryParams`。重构不改变运动逻辑，只改变参数来源。

---

## 7. collect_data.py 重构

### 7.1 当前流程

```
main():
  task_spec = TASK_PRESETS[args.task]
  params = TrajectoryParams(grasp_z=args.grasp_z, ...)      # 全局 hardcoded
  strategy = IK_STRATEGIES[task_spec.ik_strategy]            # "pick" → PickStrategy

  for episode:
    sample target_pos (random XY)
    traj = strategy.plan(target_pos, solve_ik, home, params)  # 规划+执行混合
    for step in traj:
      execute + record
```

### 7.2 重构后流程

```
main():
  task_spec = TASK_PRESETS[args.task]
  planner = get_grasp_planner(args.grasp_planner)            # "template" / "sampler"
  executor = MotionExecutor()
  motion_params = MotionParams(approach_steps=args.approach_steps, ...)

  for episode:
    sample target_pos
    asset = resolve_target_asset(scene, task_spec)            # 新：从 scene 获取目标物体

    grasp_plans = planner.plan(asset, target_pos, object_quat, rng)  # 规划：抓哪里
    grasp_plan = grasp_plans[0]                                       # 取最优

    if task_spec.motion_type == "pick":
      traj = executor.pick(grasp_plan, solve_ik, home, motion_params)
    elif task_spec.motion_type == "pick_and_place":
      place_plan = planner.plan_place(asset, place_pos, ...)
      traj = executor.pick_and_place(grasp_plan, place_plan, ...)

    for step in traj:
      execute + record
```

### 7.3 CLI 变更

| 旧 CLI | 新 CLI | 说明 |
|--------|--------|------|
| `--grasp-z 0.135` | 删除 | 由 GraspPlanner 计算 |
| `--hover-z 0.25` | 删除 | `pre_grasp_distance` 来自 template |
| `--lift-z 0.30` | 删除 | `retreat_height` 来自 template |
| — | `--grasp-planner template` | 新增：选择 planner 类型 |
| `--approach-steps 40` | 保留 | 纯运动参数 |
| `--task pick_cube` | 保留 | TaskSpec dispatch |

---

## 8. 实施路线（修订）

### Phase 3.0：架构重构（先做，不加新功能）

| 步骤 | 任务 | 验证标准 |
|:---:|------|---------|
| 0 | 创建 `robotsmith/grasp/` 和 `robotsmith/motion/` 模块 | import 无报错 |
| 1 | 定义 `GraspPlan` + `MotionParams` dataclass | 单元测试 |
| 2 | 提取 `MotionExecutor`（从 `PickStrategy` / `PickAndPlaceStrategy`） | 单元测试：给定相同输入，输出与旧代码一致 |
| 3 | 实现 `TemplateGraspPlanner` + block/cube template | `pick_cube` 重构后结果不变 |
| 4 | 更新 `collect_data.py` 使用新接口 | `--task pick_cube` 20ep 100%（与 Part 2 一致） |
| 5 | 标记 `ik_strategies.py` deprecated | 旧 import path 仍可用但 warn |

**关键原则**：Phase 3.0 完成后，**行为完全不变，只是代码结构变了**。用 `pick_cube` 同 seed 对比验证。

### Phase 3.1：新品类 template（在重构基础上扩展）

| 步骤 | 任务 | 验证标准 |
|:---:|------|---------|
| 1 | bowl template + `pick_bowl` preset | 20ep 成功率 > 80% |
| 2 | can/bottle template | `pick_can` / `pick_bottle` 20ep > 80% |
| 3 | mug template (handle grasp) | `pick_mug` 20ep > 60% |
| 4 | plate template (rim pinch) | `pick_plate` 20ep > 60% |
| 5 | 多品类 pick_and_place presets | `mug_in_bowl` 等端到端 |

### Phase 3.2：Mesh-Based Grasp Sampling

| 步骤 | 任务 | 验证标准 |
|:---:|------|---------|
| 1 | `SamplerGraspPlanner` + antipodal sampling | 每物体 > 10 个有效 candidate |
| 2 | Quality scoring (力闭合 + IK 可达性) | top-5 成功率 > 70% |
| 3 | `--grasp-planner sampler` CLI | 同品类 template vs sampler 对比 |

### Phase 3.3：Learned Affordance（远期）

| 步骤 | 依赖 |
|:---:|------|
| AnyGrasp 集成 | point cloud rendering pipeline |
| Affordance heatmap → grasp selection | 预训练模型 |

---

## 9. 成功标准

| 指标 | Part 2 现状 | Phase 3.0 | Phase 3.1 | Phase 3.2 |
|------|:-----------:|:---------:|:---------:|:---------:|
| 可操作品类 | 1 (cube) | 1（不变，验证重构正确性） | ≥ 6 | ≥ 6 |
| 代码可扩展性 | 每品类需新 IKStrategy | 每品类只需 1 个 GraspTemplate | 零人工（sampler） | — |
| IK 数采成功率 | 100% (cube) | 100% (cube, 验证不退化) | ≥ 80% (per-category 平均) | ≥ 70% |
| 单品类 grasp 多样性 | 1 pose | 1 pose | 1~3 (multi-template) | ≥ 10 candidates |

**核心验证**：Phase 3.0 → `pick_cube` 行为不变。Phase 3.1 → 用 Part 1 非 cube 资产跑完整数采 pipeline，产出 LeRobot 数据集。
