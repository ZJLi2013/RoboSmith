# RoboSmith — 技术调研与前沿分析

> 从 design.md 分离的调研性内容。包含合成数据生态对比、场景多样性技术方案、RoboLab 启发分析等。
> 为 RoboSmith 的架构决策提供依据，但不是实现文档。
>
> 相关文档：
> - [design.md](design.md) — 架构设计与实现
> - [synthetic_data_strategy.md](../../robot_notes/blogs/data_capture_strategy/synthetic_data_strategy.md) — 数据采集策略详细调研

---

## 1. 合成数据生态与 RoboSmith 定位

> 2025-2026 年多个项目已验证"纯合成数据 → VLA 训练"路线可行。
> 但业界同时形成另一个共识：**数据采集策略比数据规模更重要**。
> RoboSmith Part 2 定位在数采策略实验，而非单纯的数据规模扩张。

### 1.1 数采策略生态（远期参考）

> 详细调研见 [synthetic_data_strategy.md](../../robot_notes/blogs/data_capture_strategy/synthetic_data_strategy.md)。

业界对"数据怎么采"形成了清晰的技术路径分类：

| 路径 | 本质 | 代表工作 | RoboSmith 状态 |
|------|------|---------|:---:|
| **IK solver** | 确定性 expert 轨迹 | MimicGen (NVlabs) | **当前核心** ✅ |
| **加噪声** | 扩展 trajectory tube | [DART](https://bair.berkeley.edu/blog/2017/10/26/dart/) (Berkeley, CoRL'17) | **IK 内置参数** ✅ |
| **改采样** | 匹配 on-policy 分布 | [GSWorld DAgger](https://arxiv.org/html/2510.20813v1) (2025) | 扩展 A 📋 |
| **RL 后训练** | advantage conditioning | [π\*0.6 RECAP](https://www.pi.website/blog/pistar06) (闭源) | 扩展 B 📋 |
| **改模型** | 架构缓解 compounding error | Diffusion Policy, ACT, Flow Matching (π0) | 扩展 C |
| **Self-Play** | 自我进化数据飞轮 | [DexFlyWheel](https://arxiv.org/abs/2509.23829) | 扩展 D |

关键 insight：**IK solver 是一切高级策略的前提** — DAgger 需要 IK 作为 expert oracle，
RECAP 需要 IK 生成 seed data。先把多任务 IK 基础设施做扎实。

### 1.2 合成数据 → VLA 训练

| 项目 | 数据规模 | 训练方式 | 核心发现 |
|------|---------|---------|---------|
| [GraspVLA](https://github.com/PKU-EPIC/GraspVLA) CoRL'25 | 10亿帧，240类 | pre-train | 纯合成零样本迁移真机抓取 |
| [InternVLA-A1](https://github.com/InternRobotics/InternVLA-A1) | 63万轨迹，70任务 | pre-train | 合成数据 alone 匹配 π₀ 级别 |
| [EmbodiChain](https://github.com/DexForce/EmbodiChain) | 平台级 | 全流程 | 资产→仿真→数据→VLA 端到端 |
| [ET-VLA](https://arxiv.org/abs/2511.01224) | 少量合成 | SCP warm-up | 合成 warm-up + 真机 fine-tune +53% |

### 1.3 端到端仿真平台：Genie Sim 3.0 (AgiBot)

[Genie Sim 3.0](https://github.com/AgibotTech/genie_sim) (智元机器人, [arXiv 2601.02078](https://arxiv.org/abs/2601.02078))
是当前开源最完整的 real2sim → SDG → VLA 平台，覆盖从真实场景重建到合成数据生成的全链路。

| 模块 | 内容 | 规模 |
|------|------|------|
| **Sim-Ready Assets** | 5,140 个验证 3D 资产（零售/工业/餐饮/家居/办公） | 5 大领域 |
| **3DGS 重建管线** | 真实场景扫描 → 3D Gaussian Splatting → 高精度 mesh → USD | real2sim 核心 |
| **Genie Sim World** | 多模态空间世界模型，多种输入 → 逼真 3D 世界 | 分钟级生成 |
| **LLM 驱动场景生成** | 自然语言 → 仿真场景 + 任务指令 + 评估配置 | 多维泛化 |
| **合成数据集** | 10,000+ 小时，200+ 任务，多传感器流 | 开源 |
| **Benchmark** | 100,000+ 场景，VLM 自动评估 | π0.5 / GR00T-N1.6 / π0 |
| **Sim2Real** | 合成数据 zero-shot 迁移 ≥ 真实数据效果 | 8 任务验证 |

**与 RoboSmith 的关键区别**：

| 维度 | Genie Sim 3.0 | RoboSmith |
|------|--------------|-----------|
| **资产来源** | **real2sim** — 3DGS 扫描真实物品/场景重建 | **gen2sim** — T2I + 3D 生成模型（TRELLIS.2） |
| 资产规模 | 5,140 个（人工策划 + 扫描） | 10 Objaverse 策划 + 按需生成 |
| 仿真引擎 | Isaac Sim (NVIDIA 绑定) | MuJoCo / Genesis (AMD 兼容) |
| 机器人 | Genie G2 人形机器人 | Franka 固定臂 |
| **泛化轴** | **场景泛化** — 人形需跨房间/环境导航 | **物品泛化** — 固定臂只关心桌面物品多样性 |
| 数据规模 | 10,000+ 小时 (pre-train 级) | 5k-10k 轨迹 (post-train 级) |
| GPU 生态 | NVIDIA (Isaac Sim) | **AMD ROCm** |

**为什么泛化轴不同？**

人形机器人（Genie G2）在不同房间、厨房、仓库间移动，场景理解（"我在哪、周围有什么"）
是任务的一部分 → 场景泛化是刚需，LLM 驱动多场景生成有明确价值。

固定臂（Franka）的工作空间是半径 ~85cm 的半球，3~8 step 的桌面操作任务中，
"场景"始终是一张桌子 → 场景泛化 ROI 低，真正影响 policy 泛化的是**物品形态/外观/尺寸的多样性**。

> **启示**：Genie Sim 证明 real2sim（扫描重建）是当前 sim-ready 资产质量最高的路线，
> 但依赖 NVIDIA 生态 (Isaac Sim) 且扫描成本高。
> RoboSmith 的 gen2sim 路线（文本 → 生成 3D → URDF）在定制性和 AMD 兼容性上互补。
> 两条路线未来可能融合：扫描获取高精度几何 + 生成模型补全纹理/变体。

**RoboSmith 定位**：

```
                        数据维度                          核心关注
                        ────────                          ────────
Genie Sim 3.0         5140资产 / 10000h+   pre-train      资产规模 + 场景泛化 (NVIDIA)
GraspVLA              240类 / 10亿帧       pre-train      物品泛化 (Objaverse)
RoboLab               120 tasks            评估 benchmark  composable predicates (NVIDIA, Isaac Lab)
π*0.6 RECAP           —                    RL post-train   高级数采策略 (闭源)
RoboSmith (当前)      10-30类 / 多任务     数采 + 评估     Task 定义 + IK solver + DART + Eval Engine (AMD ROCm) ←
lerobot (已验证)       1类 / 100轨迹       fine-tune       基础验证
```

### 1.4 RoboSmith 的差异化价值与继续方向

在 Genie Sim 3.0 等重量级平台已存在的前提下，RoboSmith 不应试图复制其全栈路线，
而应聚焦自身独特优势。以下是值得继续推进的方向：

**Insight 1：Gen2Sim 的长尾物品覆盖能力是 Real2Sim 的天然补充**

Genie Sim 的 5,140 资产虽然覆盖 5 大领域，但仍是有限集合 — 遇到不在库里的物品
（如特定形状的工具、定制容器）就无法处理。RoboSmith 的"文本描述 → 3D 生成"能力
天然覆盖长尾：任何可以用语言描述的物品，都可以即时生成 sim-ready 资产。

→ **方向：做好 per-category object variation pool**（同一类物品的形态/外观/尺寸变体），
  这是 pre-train 数据集无法覆盖的 post-training 差异化数据。

**Insight 2：AMD ROCm 全栈是唯一性卖点**

目前所有主流仿真平台（Isaac Sim、Genie Sim、OmniGibson）都绑定 NVIDIA。
RoboSmith 是极少数在 AMD GPU 上验证 3D 生成 + 仿真全链路的项目：
TRELLIS.2 (MI300X) → URDF → MuJoCo/Genesis → LeRobot 训练。
这对 AMD 数据中心用户（高校、企业自建集群）有不可替代的价值。

→ **方向：保持 AMD ROCm 全栈可用性作为核心差异化**，
  Part 2 选型（MuJoCo + Genesis）天然 AMD 兼容，不要引入 NVIDIA 绑定依赖。

**Insight 3：多任务 IK 数采基础设施是一切高级策略的前提**

π\*0.6 RECAP 和 GigaBrain RAMP 等高级数采策略虽然效果好，但都没有开源复现，
且依赖成熟的 IK 基础设施作为 expert oracle。
在连多任务 IK 数采都没跑通的情况下，谈 DAgger / RECAP 为时过早。

RoboSmith 当前的核心价值是：

```
1. Task 定义系统 — composable predicates 成功判定
   → 支撑多任务 (pick / pick-and-place / stacking)
   → 不再 hardcode 判定逻辑

2. 多任务 IK solver — per-task waypoint 生成
   → DART 噪声增强作为内置参数
   → 为未来 DAgger (IK 作为 expert oracle) 打基础

3. gen2sim 资产 — 物品变体多样性
   → 覆盖同类别内不同形态/外观/尺寸
```

→ **方向：先搭好多任务 IK 基础设施，高级策略作为远期扩展。**

**Insight 4：Genie Sim Sim2Real 数据验证了合成数据 ≥ 真实数据**

Genie Sim Benchmark 的 Sim2Real 实验（8 任务）显示：
纯合成数据训练的 policy 在真机上的成功率（82.8%）**超过**真实数据训练的 policy（75.2%）。
这从工业级实验彻底验证了合成数据路线的可行性，RoboSmith 的方向是正确的。

→ **方向：不需要怀疑合成数据路线本身**，关键是数据多样性和质量，而非数量。

**Insight 5：Task Coverage > Object Variation > Data Volume**

综合业界经验，对固定臂桌面操作的优先级排序：

```
任务类型广度 (pick / place / stack)     >>> 当前最缺，Part 2 核心建设
  │
  ├── Task 定义系统                     >>> composable predicates
  ├── 多任务 IK solver                  >>> per-task waypoint 逻辑
  └── DART 噪声增强                     >>> IK 的内置增强参数
      │
物品变体多样性 (形态/外观/尺寸)         >> 决定跨物体泛化（Part 1 gen2sim 支撑）
  │
位置/朝向随机化                         >> 已在 ProgrammaticSceneBackend 实现
      │
光照/纹理/相机视角                      > P1，渲染层面 domain randomization
高级数采策略 (DAgger / RECAP)           > 远期，需 IK 基础设施成熟后
```

→ **方向：先扩展任务类型广度（Task 定义 + IK solver），再提升数采策略层级。**

**任务广度 > 物品多样性 > 数据数量**（综合业界经验）：

| 改进维度 | 优先级 | 说明 |
|-----------|:---:|------|
| **任务类型广度** (pick / place / stack) | **P0** | Task 定义 + 多任务 IK solver |
| **DART 噪声增强** | **P0** | IK 的内置参数，低成本提升鲁棒性 |
| 物体位置/朝向随机化 | P0 | 每次采集不同摆放（已实现） |
| 物体类别组合 | P0 | 同场景不同物体搭配 |
| 光照/桌面纹理 | P1 | 减少视觉 overfitting |
| 相机视角 | P1 | 至少 2-3 个视角变化 |
| 高级数采策略 (DAgger / RECAP) | P2 | IK 基础设施成熟后 |

---

## 1.5 Action Space 选型：EE Delta vs Joint Position

> 2026 年社区共识：**EE delta (Cartesian) 是 VLA 标准 action space**。
> RoboSmith 早期选择 joint position（实现简单），需对齐。

### 社区数据

13,000+ 真机 rollouts + 500+ 模型训练实验结论（[arXiv 2602.23408](https://arxiv.org/pdf/2602.23408), 2026）：
**delta actions consistently outperform absolute representations**。

所有主流 VLA 都使用 EE delta：

| VLA | Action Space | 维度 | 训练数据 |
|-----|-------------|:---:|---------|
| Pi0 / Pi0.5 | EE delta + gripper | 7D | OXE / Bridge / LIBERO |
| StarVLA | EE delta + gripper | 7D | LIBERO |
| OpenVLA | EE delta + gripper | 7D | OXE |
| GR00T N1.6 | EE delta + gripper | 7D | LIBERO |
| Diffusion Policy | EE delta + gripper | 7D | 各种 |

### EE Delta vs Joint Position

| | EE Delta (推荐) | Joint Position (已弃用) |
|---|---|---|
| **可学习性** | 小值域 `[-1,1]`，网络易学 | 大值域（rad），难学 |
| **跨 embodiment 泛化** | 与机器人 DOF 无关 | 绑定具体机器人关节 |
| **隐式安全** | clipping = 限速 | 无隐式约束 |
| **VLA 兼容** | 所有主流 VLA | 无主流 VLA 使用 |
| **实现复杂度** | 需 IK/OSC 控制器 | 直接控制 |

### Genesis 支持

Genesis 原生支持 EE delta → joint position 转换：

```python
# 当前 EE pose
ee_pos = franka.get_link('hand').get_pos()
ee_quat = franka.get_link('hand').get_quat()

# EE delta → target EE pose
target_pos = ee_pos + delta_pos
target_quat = apply_delta_rotation(ee_quat, delta_rot)

# IK 求解 → joint position → control
qpos = franka.inverse_kinematics(link=end_effector, pos=target_pos, quat=target_quat)
franka.control_dofs_position(qpos, motors_dof)
```

### 对 RoboSmith 的影响

| 组件 | 当前 (joint position) | 目标 (EE delta) |
|------|----------------------|-----------------|
| `collect_data.py` action | 9D `[j1..j7, f1, f2]` | 7D `[Δx, Δy, Δz, Δrx, Δry, Δrz, grip]` |
| `collect_data.py` state | 9D joint positions | 8D `[eef_pos3, euler/axangle3, gripper2]` |
| `benchmark.py apply_action()` | 直接 `control_dofs_position` | EE delta → IK → `control_dofs_position` |
| VLA 兼容 | 仅自训练模型 | Pi0, StarVLA, OpenVLA, GR00T... |

### 结论

**切换到 EE delta，retire joint position action space。**
IK solver 在底层保留（waypoint 计算仍用 IK），但 LeRobot dataset 中存的 action 改为 EE delta。

---

## 2. 场景多样性 — 借鉴 GraspVLA-playground

> 参考 [GraspVLA-playground](https://github.com/MiYanDoris/GraspVLA-playground)（MuJoCo/Robosuite）。
> 其场景多样性方案中有三个维度的技术可直接迁移到 Genesis + `ProgrammaticSceneBackend`。

### 2.1 三维多样性拆解

| 维度 | GraspVLA 做法 | 核心依赖 | RoboSmith 迁移可行性 |
|------|-------------|---------|-------------------|
| **物体多样性** | Objaverse 预处理子集，`random.sample(candidates, N)` | 离线 mesh 简化 + stable pose 预计算 | ✅ 直接复用 AssetLibrary |
| **Layout 多样性** | trimesh `CollisionManager` 碰撞感知随机摆放 | 纯几何，与仿真器无关 | ✅ 可直接集成到 `ProgrammaticSceneBackend.resolve()` |
| **环境外观多样性** | 38 种地板 + 43 种墙壁纹理随机替换 | Robosuite MuJoCo texture | ⚠️ 需适配 Genesis `gs.surfaces` / texture API |

### 2.2 关键技术：碰撞感知的多物体摆放

GraspVLA 的 layout 随机化**不依赖 BDDL 或 Robosuite**，核心是 `misc/sampling.py` 中的纯几何方案：

```python
# GraspVLA 核心逻辑 (misc/sampling.py)
collision_manager = trimesh.collision.CollisionManager()
for obj in objects:
    mesh = trimesh.load_mesh("simplified.obj")
    stable_pose = random.choice(precomputed_stable_poses)   # 预计算稳定姿态
    rand_pos = [uniform(0.35, 0.7), uniform(-0.2, 0.2), stable_pose.z]
    rand_yaw = uniform(0, 2π)
    transform = compose(rand_pos, rand_yaw @ stable_pose.orientation)
    if collision_manager.min_distance_single(mesh, transform) > 0.02:  # 2cm 间距
        collision_manager.add_object(mesh, transform)   # 接受，加入碰撞管理
```

两个关键点：

1. **Stable pose 预计算** — 每个 Objaverse 物体离线生成若干稳定放置姿态（`table_pose.json`：z 高度 + quaternion），运行时随机选一个。保证物体不会以不稳定姿态出现在桌面上。
2. **逐物体碰撞检测** — 依次放置，每放一个检查与已有物体的最小距离 > 阈值（2cm），最多重试 100 次。纯 trimesh 计算，不需要仿真器参与。

### 2.3 BDDL 的角色澄清

BDDL（Behavior Domain Definition Language）在 GraspVLA 中**仅用于向 Robosuite 声明"场景里有哪些物体、目标是什么"**，不负责 layout 计算。BDDL 本身不绑定 NVIDIA Omniverse（BEHAVIOR benchmark 绑 Omniverse，但 LIBERO benchmark 绑 MuJoCo）。

RoboSmith 使用 Genesis + URDF，**不需要引入 BDDL**。`SceneConfig` + `ProgrammaticSceneBackend` 已覆盖同等功能。

### 2.4 RoboSmith 迁移方案

**Step 1 ✅：碰撞感知 `ProgrammaticSceneBackend.resolve()`**

已实现于 `robotsmith/scenes/backend.py`。核心流程：

```
对每个 ObjectPlacement:
  1. library.search(query) → 匹配资产 (支持 Objaverse + primitive)
  2. _pick_stable_pose() → 按概率采样 (z_offset, quaternion)
  3. 采样 (x, y) ∈ workspace_xy (或 per-object position_range)
  4. z = table_height + table_thickness/2 + stable_pose.z
  5. _CollisionChecker.min_distance_single() → ≥ margin 则接受
  6. 最多 retry 100 次，失败则 skip + warning
```

碰撞检测后端：优先 FCL (`python-fcl`, Linux)，fallback 到 AABB 距离检测 (Windows)。

**Step 2 ✅：Workspace + Stable pose — 两层采样**

```
Workspace XY (机器人属性, 场景级)      Stable pose (物体属性, 资产级)
   "在哪采样位置"                        "以什么姿态放"
        │                                      │
        ▼                                      ▼
   采样 (x, y) ∈ workspace_xy            按概率选稳定朝向 → z + quat
        │                                      │
        └────────────── 组合 ──────────────────┘
                         │
                         ▼
                (x, y, z, qx, qy, qz, qw)
                         │
                    碰撞检测 → 通过则接受
```

`SceneConfig` 已实现 `workspace_xy` 字段（默认 Franka 矩形近似 `[[0.35, -0.25], [0.70, 0.25]]`），
以及 `collision_margin` (2cm) 和 `max_placement_retries` (100)。

Stable poses 已对全部 26 资产预计算（`trimesh.compute_stable_poses(n_samples=500)`，Linux MI300X），
结果存入各 `metadata.json`。典型结果：

| 资产类型 | stable poses 数 | 说明 |
|---------|:-:|------|
| block/box (primitive) | 1 | 方块只有平放 |
| mug (Objaverse) | 4-5 | 正立、侧倒、倒扣 |
| fruit — apple | 2 | 近球形，正/倒 p≈0.5 |
| fruit — banana | 21 | 多角度 |
| plate | 2 | 正面/反面 |
| bottle | 5-66 | 圆柱可多角度静止 |

**Step 3 ✅：Genesis scene loader**

已实现于 `robotsmith/scenes/genesis_loader.py`：

```python
handle = load_resolved_scene(resolved, gs_module=gs, fps=30)
# handle.scene   → gs.Scene (已含 Plane + Table + Objects + Franka)
# handle.franka  → Franka entity
# handle.objects → [obj_entity, ...]  按 placed_objects 顺序
# handle.cameras → {"default": cam}
```

`collect_data.py` 已集成：`--scene tabletop_simple` 使用 ResolvedScene 模式，不带 `--scene` 保持原始 hardcoded cube 模式。

**Step 4：环境外观随机化（P1, 待实现）**

| 方法 | Genesis API | 说明 |
|------|-----------|------|
| 颜色随机化 | `gs.surfaces.Default(color=random_rgb)` | 最简单，已可用 |
| 纹理贴图 | `gs.surfaces.Default(texture=path)` | 取决于 Genesis 渲染后端支持 |
| 光照随机化 | `scene.add_light(...)` 参数随机 | 减少视觉 overfitting |

### 2.5 实施优先级

| 优先级 | 改动 | 状态 | 收益 |
|:---:|------|:---:|------|
| P0 | Step 1 碰撞感知摆放 | ✅ | 多物体场景不穿透 |
| P0 | Step 2 stable pose + workspace | ✅ | 物体物理合理摆放 |
| P0 | Step 3 genesis_loader bridge | ✅ | `collect_data.py --scene` 多物体场景 |
| P1 | Step 4 纹理/颜色随机化 | 📋 | 视觉 domain randomization |

### 2.6 实现笔记 (2026-04-14)

**已完成文件变更：**

| 文件 | 改动要点 |
|------|---------|
| `robotsmith/scenes/config.py` | +`workspace_xy`, `collision_margin`, `max_placement_retries` |
| `robotsmith/scenes/backend.py` | `_CollisionChecker` (FCL/AABB), `_pick_stable_pose`, `_load_collision_mesh`; `resolve()` 碰撞感知 |
| `robotsmith/scenes/genesis_loader.py` | 新文件：`load_resolved_scene()` → `GenesisSceneHandle` |
| `robotsmith/scenes/presets/tabletop_simple.py` | 简化：不再硬编码 position_range，依赖 workspace_xy |
| `robotsmith/assets/schema.py` | +`stable_poses` field |
| `robotsmith/assets/builtin.py` | 7 primitives + table + plane (清理旧 mug/bowl) |
| `scripts/part1/compute_stable_poses.py` | 新脚本：convex_hull + `trimesh.compute_stable_poses(n_samples=500)` |
| `scripts/part1/import_objaverse.py` | 按 10 品类 / 17 variants 下载 |
| `scripts/part2/collect_data.py` | +`--scene` / `--assets-root` 参数，集成 genesis_loader |
| `tests/test_scenes.py` | 37 tests passing (collision, stable pose, workspace, quat) |
| `tests/test_library.py` | 更新 BUILTIN_COUNT=9, 新品类 assertion |

**关键设计决策：**

1. **碰撞检测跨平台**：`python-fcl` 在 Windows 难装，实现 AABB fallback 保证本地单测通过。远端 Linux 使用 FCL 精确碰撞。
2. **Stable pose 远端计算**：`trimesh.compute_stable_poses()` 需要物理模拟，Windows 极慢。在 MI300X Linux 节点批量计算，`scp` 传回 `metadata.json`。
3. **`collect_data.py` 双模式**：`--scene` 模式用 ResolvedScene + genesis_loader；无参数保持原始 hardcoded cube 兼容，避免破坏已有工作流。
4. **资产体积控制**：Objaverse 资产走按需下载 (`scripts/part1/import_objaverse.py`)，Git 仅 track json/urdf，`.gitignore` 排除 glb/obj/stl/png。

---

## 3. RoboLab 启发分析

> [RoboLab](https://github.com/NVLabs/RoboLab)（NVIDIA, arXiv 2604.09860）是 120 task 的评估 benchmark，
> 与 RoboSmith 互补（RoboLab = 评估端，RoboSmith = 数据端）。以下是对 RoboSmith 有启发的设计点。
>
> 已吸收到 design.md 的部分：composable predicates → §2.3, eval → vla-eval benchmark plugin。

### 3.1 Composable Predicates 任务成功判定（已吸收 → §2.3）

RoboLab 最核心的设计是**声明式任务定义** — 每个 task 是一个 dataclass，
成功/失败通过可组合的谓词函数自动判定：

```python
success = DoneTerm(
    func=object_in_container,
    params={"object": "apple", "container": "bowl", ...},
)
```

内置 ~20 个 conditional 函数：`object_in_container`, `object_on_top`, `object_upright`,
`stacked`, `pick_and_place` 等。

**RoboSmith 状态**：§2.3 的 TaskSpec + composable predicates 设计已吸收此思路。
待实现：`object_above`, `object_in_container`, `stacked` 等具体谓词函数。

### 3.2 Server-Client Policy 架构（已吸收 → vla-eval plugin）

RoboLab 把 policy inference 和 sim 评估完全解耦：
- Policy 作为独立 server 运行（WebSocket / ZMQ）
- RoboLab 通过轻量 client 发 observation、收 action

这让它可以评估任意 policy（Pi0.5, GR00T, PaliGemma...）而不需要改 sim 代码。

**RoboSmith 状态**：Eval 通过 vla-eval-harness 实现，RoboSmith 只提供 `RoboSmithBenchmark` plugin。

```python
class InferenceClient(ABC):
    def infer(self, obs: dict) -> np.ndarray: ...
    def reset(self): ...
```

### 3.3 AI-Enabled Task/Scene 生成（Cursor Skill）

RoboLab 提供 `/robolab-scenegen` 和 `/robolab-taskgen` 两个 Claude Code skills —
用自然语言描述就能生成完整的场景和任务文件。

**RoboSmith 迁移方案**：结合已有 gen2sim 能力，实现"一条命令从文本描述到可训练数据集"：

```
用户: "Franka 从桌上抓红色杯子放到碗里"
  → LLM/Skill 自动生成:
    1. SceneConfig (mug_red + bowl + table, workspace 约束)
    2. TaskSpec (instruction + success predicate: object_in_container)
    3. collect_data.py 的参数
```

### 3.4 多环境并行评估

RoboLab 支持 `--num-envs 12` 并行跑 12 个 episode，且每个 env 独立终止
（vectorized conditionals）。100 tasks × 多轮评估时将评估时间从线性降到近常数。

**RoboSmith 状态**：并行评估由 vla-eval-harness 负责，RoboSmith 不自建。

### 3.5 Subtask Progress Tracking

RoboLab 的 subtask 系统可以追踪任务内部的进度
（比如 pick-and-place 分解为 grabbed → lifted → moved → placed），
即使最终失败也能知道"做到了哪一步"。

**RoboSmith 状态**：milestone predicates 设计已完成，后续扩展。

### 3.6 指令变体（Instruction Variants）

RoboLab 的每个 task 可以有 `default / vague / specific` 三种指令变体：

```python
instruction = {
    "default": "Pick up the banana and place it on the plate",
    "vague": "Put stuff on the plate",
    "specific": "Grasp the yellow banana by its middle section...",
}
```

用于测试 policy 对语义模糊度的鲁棒性。RoboSmith 的 TaskSpec 可以轻松支持。

### 3.7 优先级排序

| 优先级 | 借鉴点 | 对应 RoboSmith 阶段 | 实现成本 | 状态 |
|:---:|------|:---:|:---:|:---:|
| **P0** | Composable predicates 成功判定 | Stage 2 前 | 低 | §2.3 已设计，待实现 |
| **P0** | Server-client policy 架构 | — | 中 | 由 vla-eval-harness 提供 |
| **P1** | 并行评估 (vectorized eval) | — | 中 | 由 vla-eval-harness 提供 |
| **P1** | Subtask progress tracking | 后续扩展 | 低 | milestone predicates 已设计 |
| **P1** | AI-enabled task/scene gen (Cursor skill) | 任何时候 | 低 | 📋 |
| **P2** | 资产工具链增强 (batch physics update) | 持续 | 低 | 📋 |
| **P2** | Instruction variants | Stage 2+ | 低 | 📋 |

### 3.8 两者定位对比

```
RoboLab (纯评估端)                     RoboSmith (Data Infra)
120 tasks × 成功判定                    gen2sim → 数据采集 → LeRobot dataset
Isaac Lab + NVIDIA GPU (only)           Genesis + AMD ROCm
server-client policy 接口               vla-eval benchmark plugin (RoboSmithBenchmark)
"你的 policy 在标准任务上多强？"           "帮你生成数据，eval 走 vla-eval-harness"
无数据采集能力                            Part 2 Data Engine
```

**策略**：RoboSmith 不自建 eval engine，评估通过 vla-eval-harness 完成。
RoboSmith 提供 `RoboSmithBenchmark` 作为 vla-eval benchmark plugin。
RoboLab 的设计精髓（composable predicates, subtask tracking）已吸收到 data engine 的 predicate 体系中。
详见 [design.md — Eval：vla-eval Benchmark Plugin](design.md#evalvla-eval-benchmark-plugin)。

---

## 3.5 Grasp 生成方案全景

> 不规则物体（碗、盘、瓶、玩具等）的 grasp pose 生成是 RoboSmith Part 3 的核心问题。
> 详细技术文章见 [general_object_grasp_solution.md](general_object_grasp_solution.md)。
> 本节聚焦方案对比与 RoboSmith 选型结论。

### 3.5.1 方案分类

Grasp 生成方案按方法论分为三类：

```
纯几何（无 ML）         学习型（ML）              端到端 Policy
────────────          ─────────────           ────────────────
Antipodal Sampling     Contact-GraspNet        GraspLDP
Force Closure Metric   GraspGen                AnchorDP3
                       AnyGrasp                Spatial RoboGrasp
                       DexNet / GQ-CNN

输出: grasp pose set   输出: grasp pose set    输出: action trajectory
```

### 3.5.2 主流方案对比

> CUDA → ROCm 技术迁移不是阻碍因素（hipify / PyTorch ROCm 均可解决），因此不作为选型维度。
> 核心筛选维度：**License 可再分发** + **Parallel-jaw gripper 适用** + **技术成熟度**。

| 方案 | 方法 | 输入 | 训练数据 | License | 推理速度 | 精度 | Parallel-jaw |
|------|------|------|---------|---------|:---:|:---:|:---:|
| **Antipodal Sampling** | 表面点配对 + force closure | mesh | 无需 | 自有 | ~1s/物体 | 中 | ✅ |
| **DexNet 2.0** (Berkeley) | GQ-CNN 回归 | depth image | 6.7M grasps | BSD | ~0.3s | 中-高 | ✅ |
| **VGN** (ETH, CoRL'20) | 3D CNN → voxel grid 回归 | TSDF volume | 5M grasps | **BSD-3** | **10ms** | 中-高 (92% 真机) | ✅ |
| **Contact-GraspNet** (NVlabs, ICRA'21) | Per-point 4-DoF 回归 (PointNet++) | 场景点云 | 17M grasps | ❌ NVIDIA proprietary | ~200ms | 高 (>90%) | ✅ |
| **GraspNet-Baseline** (清华, CVPR'20) | 点云 grasp 检测 | 点云+RGB | GraspNet-1B | ⚠️ NOASSERTION | ~50ms | 高 | ✅ |
| **AnyGrasp** (清华, T-RO'23) | 大规模点云检测 | 场景点云 | 大规模 | ❌ 商业 SDK | ~100ms | 高 | ✅ |
| **GraspGen** (NVlabs, ICRA'26) | DiT Diffusion + Discriminator | 物体点云 | 53M grasps | ❌ NVIDIA proprietary | ~50ms (20Hz) | SOTA | ✅ |
| **GraspLDP** (CVPR'26) | Latent Diffusion + Grasp Prior → action chunk | wrist RGB | demos | 未公开 | real-time | 高 | ✅ |
| **GraspLoCoMo** (IROS'18) | Local Contact Moment 匹配 | 点云+法线 | 无需 | **BSD-3** | — | 中 | ✅ |
| **GenDexGrasp** (ICRA'23) | Contact Map CVAE → 多指优化 | 物体点云 | MultiDex | 无 LICENSE | — | — | ❌ **灵巧手专用** |

### 3.5.3 License 约束下的选型

RoboSmith 的硬约束：**可开源再分发**（CUDA→ROCm 技术迁移不是问题，但版权问题是）。

| 方案 | License | 可集成进 RoboSmith | 备注 |
|------|:---:|:---:|------|
| Antipodal Sampling | ✅ 自有代码 | **✅** | 纯几何，无外部依赖 |
| DexNet 2.0 | ✅ BSD | **✅** | TF 老版本，方法论可参考 |
| **VGN** | **✅ BSD-3** | **✅** | **唯一 license 干净 + learned + parallel-jaw 方案** |
| GraspFactory 方法论 | ✅ MIT (Autodesk) | **✅** | 数据集 + sampling 方法论 |
| GraspLoCoMo | ✅ BSD-3 | ⚠️ | C++ 实现，集成成本高，小众 |
| GraspNet-Baseline | ⚠️ NOASSERTION | ⚠️ | license 不清，需联系作者确认 |
| Contact-GraspNet | ❌ NVIDIA proprietary | **❌** | 不可再分发、不可 production |
| GraspGen | ❌ NVIDIA proprietary | **❌** | 同上 + 商用需走 NVIDIA Licensing |
| AnyGrasp | ❌ 商业 SDK | **❌** | 商用需授权 |
| GenDexGrasp | 无 LICENSE | **❌** | 且仅支持灵巧手，不适用 parallel-jaw |

**结论（两条腿走路）**：

- **Phase 3.2(a) — SamplerGraspPlanner**（antipodal sampling + force closure，纯 NumPy/trimesh）
  快速 baseline，无外部依赖，参考 [GraspFactory](https://github.com/AutodeskRoboticsLab/graspfactory) (MIT) 方法论，~3 天
- **Phase 3.2(b) — VGN 集成**（[ethz-asl/vgn](https://github.com/ethz-asl/vgn), BSD-3, PyTorch）
  Learned model，精度天花板更高（92% 真机），10ms inference，无 custom CUDA ops，~3-4 天
- **Phase 3.3**：对比 (a) vs (b)，择优作为默认 planner
- **学术对比**：Contact-GraspNet / GraspGen 可作为 baseline 对比（license 允许内部研究使用），但不打包进 RoboSmith

### 3.5.4 Antipodal Sampling 技术细节

`SamplerGraspPlanner` 的核心算法：

```
输入: mesh (trimesh.Trimesh), gripper_max_opening (0.08m), μ (摩擦系数, 0.5)

1. 表面采样
   points, normals = mesh.sample(N=10000, return_index=True)
   → N 个 (point, face_normal) 对

2. Antipodal 配对
   对每个 p1, 找 p2 满足:
   - ||p1 - p2|| < gripper_max_opening
   - n1 · n2 < -cos(30°)                    ← 法线近似反向
   - 连线 (p2-p1) 在两点的摩擦锥内          ← force closure 条件
   用 KD-Tree 加速邻域搜索

3. Grasp 参数化
   grasp_center = (p1 + p2) / 2
   grasp_axis = normalize(p2 - p1)           ← 两指方向
   approach = 选择与 grasp_axis 正交且朝上的方向
   finger_width = ||p2 - p1||

4. Force Closure 评分
   q = min(
     角度余量: |acos(-n1·n2)| / π,
     摩擦余量: min(friction_cone_margin(p1), friction_cone_margin(p2))
   )

5. 碰撞过滤
   将 gripper mesh 放到候选 pose
   trimesh.collision.CollisionManager 检查 gripper ∩ object
   碰撞的去掉

6. 聚类去重
   对 survivors 在 SE(3) 空间做 agglomerative clustering
   每 cluster 取 quality 最高的

输出: list[GraspPlan], 按 quality 降序
```

**社区参考**：
- [GraspFactory](https://github.com/AutodeskRoboticsLab/graspfactory) (Autodesk, Apache-2.0) — 109M grasps，
  用 antipodal sampling + Isaac Sim 物理验证，Franka Panda 子集 12.2M grasps
- [DexNet](https://github.com/BerkeleyAutomation/dex-net) (Berkeley, BSD) — `PointGraspMetrics3D` 类实现了
  完整的 force closure / wrench space 分析
- GraspGen 的 [GRASP_DATASET_FORMAT.md](https://github.com/NVlabs/GraspGen/blob/main/docs/GRASP_DATASET_FORMAT.md) —
  数据格式参考（grasp pose + success label + gripper config）

### 3.5.5 NVlabs Grasp 系列 License 详解

Contact-GraspNet 和 GraspGen 都来自 NVlabs，license 需要特别说明：

| 项目 | License 文件 | 声明 | 实际含义 |
|------|-------------|------|---------|
| Contact-GraspNet | `License.pdf` (NVIDIA Source Code License) | 仅限非商业研究 | 不可再分发、不可 production |
| GraspGen | `LICENSE` | "Copyright © 2025 NVIDIA. All rights reserved." | 同上 + 商业需走 NVIDIA Research Licensing |
| GraspGen 数据集 | HuggingFace `nvidia/PhysicalAI-Robotics-GraspGen` | **CC-BY 4.0** | **数据可商用**（仅需标注出处） |
| GraspGen checkpoints | HuggingFace `adithyamurali/GraspGenModels` | 同代码 license | 权重不可再分发 |

**技术栈备注**：两者都依赖 `pointnet2_ops`（手写 CUDA kernel），GraspGen 额外依赖 PyG 生态（torch-cluster / torch-scatter）。
CUDA→ROCm 迁移技术上可行（hipify / PyTorch ROCm），但 **license 是硬伤，迁移了也不能分发**。

**PyTorch 社区移植**（Contact-GraspNet）：
- [elchun/contact_graspnet_pytorch](https://github.com/elchun/contact_graspnet_pytorch) (80 stars) — 最活跃
- [sebbyjp/cgn_pytorch](https://github.com/sebbyjp/cgn_pytorch) — pip 可装 (`pip install cgn-pytorch`)
- License 均标记 "Other (NOASSERTION)" — 从 NVlabs 移植，license 继承不清

### 3.5.6 从 0 训练一个 AMD 版本的 GraspGen 模型

> 详细方案（数据集全景、壁垒分析、数据格式统一、MI300X 训练周期估算、实施路线）
> 已迁移至独立项目：[overnight_tasks/graspgen_amd/readme.md](../../../overnight_tasks/graspgen_amd/readme.md)

**核心结论**：
- GraspGen **数据集** CC-BY 4.0 可商用，**代码/权重** NVIDIA 专有不可用 → 需自行实现模型
- 开源可用数据：GraspGen 57M + GraspFactory 109M = **166M grasps**，数据不是壁垒
- MI300X 8 卡纯训练时间 ~3-4 天（Generator + Discriminator）
- 含代码自主实现的完整周期：**~4-5 周**（到 NV 80%+ 精度）
- **捷径**：Phase A+B（PointNet++ 回归），**~1.5-2 周**，精度 70-80%，sim SDG 可能够用

---

## 4. 3D 生成模型全景调研

> 行业正从"生成好看的 3D"过渡到"生成仿真可用的 3D"。
> 按 sim-ready 深度，开源模型可分三层：

### 4.1 三层全景

```
                        视觉保真  物理属性  碰撞精度  关节/铰接  直接入 sim
                        ────────  ────────  ────────  ────────  ─────────
Layer 1: 视觉 SOTA（几何 + 纹理）
  TRELLIS.2              ★★★★★     ❌        ❌        ❌         ✅ ROCm verified
  Hunyuan3D-2.1          ★★★★☆     ❌        ❌        ❌         ❌ 需转换
  TripoSG/SF             ★★★★☆     ❌        ❌        ❌         ❌ 需转换
  Pandora3D              ★★★☆☆     ❌        ❌        ❌         ❌ 需转换
  Rodin Gen-2 (非开源)    ★★★★★     ❌        ❌        ❌         ❌ API only

Layer 2: 真正 sim-ready（物理属性 + 碰撞 + 关节）
  PhysX-Anything (CVPR26)★★★☆☆     ✅ 实测    ✅        ✅ URDF   ✅ MuJoCo/Isaac
  SIMART (SIGGRAPH26)    ★★☆☆☆     ✅        ✅        ✅ URDF   ✅ 分解铰接
  Seed3D (ByteDance)     ★★★☆☆     ✅ 自动    ✅ 水密    ❌         ⚠️ USD/Isaac
  SOPHY                  ★★★☆☆     ✅ 材质    ✅        ❌         ⚠️

Layer 3: 历史里程碑
  shap-e (2023)          ★☆☆☆☆     ❌        ✅ 水密    ❌         ❌ 需转换
  TripoSR (2024)         ★★☆☆☆     ❌        ⚠️        ❌         ❌ 需转换
```

### 4.2 Layer 1 详细对比

| 模型 | 团队 | PBR | VRAM | 速度 | ROCm | 要点 |
|------|------|:---:|------|------|------|------|
| [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) 4B | Microsoft | ✅ | ≥24 GB | ~3s (H100), ~275s (MI300X) | **✅ 已验证** | Mesh 质量天花板，CVPR'25 Spotlight。[ROCm fork](https://github.com/ZJLi2013/TRELLIS.2/tree/rocm) |
| [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | Tencent | ✅ | 10+21 GB | ~60s | **✅ 已验证** | **当前默认后端**。344K verts，AOTriton FA |
| [TripoSG](https://github.com/VAST-AI-Research/TripoSG) 1.5B | VAST-AI | ⚠️ | ≥6 GB | 快 | 待验证 | MoE Transformer，VRAM 低，纹理非 PBR |
| [TripoSF](https://github.com/VAST-AI-Research/TripoSF) | VAST-AI | 配合 SG | ≥12 GB | 中 | 待验证 | 1024³ 超高分辨率 mesh |
| [Pandora3D](https://github.com/Tencent/Tencent-XR-3DGen) | Tencent XR | ✅ | 中 | 中 | 可能 | 多阶段纹理 pipeline |
| [Rodin Gen-2](https://developer.hyper3d.ai/) 10B | Deemos | ✅ 2K | — | ~60s | — | **非开源** API only |

---

## 5. Scene-level 工具调研

**AMD 兼容 / 平台无关：**

| 工具 | 输入 | sim-ready? | 备注 |
|------|------|-----------|------|
| [SceneSmith](https://github.com/nepfaff/scenesmith) ⭐336 | 自然语言 | ✅ | Drake SDF + 碰撞体 |
| [RoboGen](https://github.com/Genesis-Embodied-AI/RoboGen) ICML'24 | LLM | ✅ | Genesis 团队，场景+任务+轨迹 |
| [Holodeck](https://github.com/allenai/Holodeck) CVPR'24 | 自然语言 | ⚠️ | AI2-THOR 绑定 |
| [PhyScene](https://github.com/PhyScene/PhyScene) CVPR'24 | 文本 | ✅ | 面向具身智能 |

**NVIDIA 绑定（思路参考）：**

| 工具 | 特点 |
|------|------|
| [AnyTask](https://anytask.rai-inst.com/) | 最完整的自动任务+数据生成 (Isaac Sim) |
| [GenManip](https://github.com/InternRobotics/GenManip) CVPR'25 | LLM 驱动操作仿真 |

---

## 6. 开源资产来源

| 库 | 数量 | 格式 | sim-ready | 备注 |
|---|---|---|---|---|
| [Objaverse](https://objaverse.allenai.org/) | **百万级** | mesh | ❌ 需转换 | **GraspVLA 等项目的核心资产来源** |
| [Google Scanned Objects](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research) | ~1000 | SDF | ✅ | 高质量 3D 扫描，无生成 artifact |
| [PartNet-Mobility](https://sapien.ucsd.edu/browse) | ~2000 | URDF | ✅ | 铰接物体，ManiSkill 使用 |
| [ManiSkill](https://github.com/haosulab/maniskill) | ~50 | URDF | ✅ | 预打包，可直接用 |
| [PyBullet](https://pybullet.org/) | ~20 | URDF | ✅ | 经典基础资产 |
| [ArtVIP](https://huggingface.co/datasets/X-Humanoid/ArtVIP) ICLR'26 | 206 铰接 | URDF/USD | ⚠️ | 铰接零件级别 |

> **策略建议**：高频类目（杯子、碗、瓶子等）优先从 Objaverse/GSO 策划导入，
> 长尾类目再走 TRELLIS.2 / Hunyuan3D 生成 pipeline。这是 GraspVLA 等大规模项目的验证路线。

---

## 7. 参考链接

**3D 资产生成 & 转换**
- [SceneSmith](https://github.com/nepfaff/scenesmith) — Sim-ready 场景生成
- [mesh-to-sim-asset](https://github.com/nepfaff/mesh-to-sim-asset) — Mesh → SDF 碰撞体
- [ArtVIP](https://huggingface.co/datasets/X-Humanoid/ArtVIP) — 铰接物体资产库 (ICLR'26)
- [URDF-Anything+](https://github.com/URDF-Anything-plus/Code) — 图片 → URDF (ICML'26)

**场景多样性 & 评估**
- [GraspVLA-playground](https://github.com/MiYanDoris/GraspVLA-playground) — GraspVLA 仿真环境（Objaverse 物体 + 碰撞摆放 + 纹理随机）
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) — 终身学习操作 benchmark
- [vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness) — 统一 VLA 评估框架 (Allen AI, 13+ benchmarks, Docker 隔离)

**仿真平台 & 可视化**
- [MuJoCo](https://mujoco.org/) — 精确接触物理引擎
- [mjviser](https://github.com/mujocolab/mjviser) — Web MuJoCo viewer (viser)
- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) — GPU 并行物理仿真
- [PyBullet](https://pybullet.org/) — 轻量级物理仿真

**Task 定义 & 数采参考**
- [RoboLab](https://github.com/NVLabs/RoboLab) — 120 task benchmark, composable predicates (NVIDIA, 2026)
- [MimicGen](https://mimicgen.github.io/) — Scalable synthetic demo generation (NVlabs, CoRL'23)
- [DART](https://bair.berkeley.edu/blog/2017/10/26/dart/) — Noise injection for robust BC (Berkeley, CoRL'17)

**高级数采策略（远期参考）**
- [GSWorld](https://arxiv.org/html/2510.20813v1) — Closed-loop DAgger in sim (2025)
- [π\*0.6 RECAP](https://www.pi.website/blog/pistar06) — RL 后训练 + advantage conditioning (Physical Intelligence, 2025)
- [GigaBrain RAMP](https://gigaai.cc/blog) — 世界模型条件化 VLA (GigaAI, 2026)
- [DexFlyWheel](https://arxiv.org/abs/2509.23829) — Self-improving data flywheel (2025)

**策略训练 & 评估（验证工具）**
- [LeRobot](https://github.com/huggingface/lerobot) — 数据格式 + 训练框架
- [SmolVLA](https://huggingface.co/lerobot/smolvla_base) — 轻量 VLA（默认验证模型）
- [π0.5 / openpi](https://github.com/Physical-Intelligence/openpi) — SOTA VLA
- [OpenVLA](https://github.com/OpenVLA/OpenVLA) — 7B 开源 VLA

**World Model**
- [GigaWorld-0](https://github.com/open-gigaai) — Video + 3D + 物理 WM
- [Kairos 3.0-4B](https://apnews.com/press-release/media-outreach/ace-robotics-open-sources-real-time-generative-world-model-kairos-3-0-4b-3a7b28af3090368478c26f4613504a6d) — 具身 WM
- [DIAMOND](https://github.com/eloialonso/diamond) — Diffusion WM

---

## 8. 实现笔记

> 从 design.md 迁移的详细实现细节、已知问题、prompt 工程等内容。
> 为具体实现提供参考，但不属于架构设计。

### 8.1 内置资产详细规划

#### 10 品类桌面操作物品

> 选品原则：**几何拓扑多样性最大化**（而非按生活用品分类），
> 使每个品类贡献不同的抓取挑战，最大化 VLA policy 泛化收益。

| # | 品类 | 代表几何 | 抓取挑战 | 典型尺寸 | 变体 | 资产来源 |
|---|------|---------|---------|---------|:---:|---------|
| 1 | 马克杯 (mug) | 圆柱 + 把手 | 把手 affordance，朝向影响策略 | 8-10cm | 3 | Objaverse LVIS `mug` |
| 2 | 碗 (bowl) | 凹半球 | 边缘夹取，可堆叠 | 10-14cm | 2 | Objaverse LVIS `bowl` |
| 3 | 积木 (block) | 长方体 | 基线几何，多面可抓 | 3-5cm | 3 | Primitive (颜色变体) |
| 4 | 易拉罐 (can) | 短圆柱 | 光滑曲面，侧面抓 | 6-12cm | 2 | Objaverse LVIS `can` |
| 5 | 瓶子 (bottle) | 高圆柱 + 窄颈 | 高重心，颈/身两种抓法 | 15-22cm | 2 | Objaverse LVIS `bottle` |
| 6 | 水果玩具 (fruit) | 近球/椭球 | 滚动、非对称（香蕉弯曲） | 5-10cm | 3 | Objaverse keyword `toy fruit` |
| 7 | 动物玩具 (figurine) | 不规则凸包 | 复杂几何，无明显抓取面 | 5-10cm | 3 | Objaverse keyword `toy animal` |
| 8 | 盘子 (plate) | 扁圆盘 | 薄 + 平，需边缘 pinch | 12-18cm | 2 | Objaverse LVIS `plate` |
| 9 | L/T 形块 (L-block) | 非凸多面体 | 重心偏移，抓取点选择 | 6-10cm | 2 | Primitive URDF 组合 |
| 10 | 小盒子 (box) | 扁长方体 | 类比纸盒/食品盒，扁平抓 | 8-15cm | 2 | Primitive URDF |

#### 资产尺寸预算

> **目标：全部资产 < 500 MB**（初期控制在 ~100 MB 内）。

| 来源 | 数量 | 单个大小 | 小计 |
|------|:---:|:---:|:---:|
| Objaverse 导入 | 17 | ~3 MB | ~51 MB |
| TRELLIS.2 @ 512 补缺 | ~0-3 | ~2 MB | ~6 MB |
| Primitive URDF | 7 | ~1 KB | ~0 MB |
| **合计** | **24** | | **~60 MB** |

#### 资产来源策略

| 来源 | 适用品类 | 视觉质量 | 单个大小 | 说明 |
|------|---------|:---:|:---:|------|
| **Objaverse 按需导入** ← 主力 | #1-8 (17 个变体) | ★★★★ | 1-5 MB | `objaverse` 包按 UID 下载 |
| **TRELLIS.2 @ 512** ← 补缺 | 搜索未命中时 | ★★★★ | ~2 MB | 512px fast 预设 |
| **Primitive URDF** ← 仅限几何体 | #3 block, #9 L-block, #10 box | ★★ | ~1 KB | 形状本身就是方块 |

### 8.2 TRELLIS.2 vs Hunyuan3D E2E 对比（red ceramic mug, MI300X, 512³）

| 指标 | Hunyuan3D-2.1 PBR | TRELLIS.2-4B |
|------|------|------|
| **总耗时** | 531s | 423s |
| **Shape 顶点** | 492,749 | 5,405,042 → 652,279 (decimated) |
| **GLB 大小** | 1.1 MB | 38.6 MB |
| **纹理** | PBR Paint (153 KB) | O-Voxel PBR (4K, 13.6 MB) |
| **纹理质量** | 外部尚可，内壁破碎 | 4K PBR 完整 |
| **底座 artifact** | 有（需 cleanup） | **无** ✅ |
| **bpy 依赖** | 需 lazy-import patch | **无** |

### 8.3 T2I Prompt 工程

#### 正向 Prompt 模板（约束版）

```python
T2I_PROMPT_TEMPLATE = (
    "a single {obj}, centered, isolated object, "
    "pure white background, no surface, no ground, no table, "
    "front view, slight top-down angle, "
    "clean silhouette, sharp edges, opaque, matte finish, "
    "uniform soft lighting, no shadow, no reflection, "
    "minimalist, object-only, high detail"
)
```

> CLIP 77 token 限制：关键约束（`no surface, no ground, no table`）必须放在前 77 tokens 内。

#### 反向 Prompt

```python
T2I_NEGATIVE_PROMPT = (
    "table, surface, floor, ground, shadow, reflection, mirror, glossy, specular, "
    "studio setup, product display, pedestal, platform, "
    "background texture, gradient background, pattern background, "
    "environment, scene, room, wall, "
    "dramatic lighting, cinematic lighting, "
    "glass, transparent, translucent, "
    "depth of field, blur, bokeh"
)
```

#### 参数约束

| 参数 | SDXL-Turbo | SDXL-Base |
|------|:-:|:-:|
| `guidance_scale` | **0.0** | **≤5.0** |
| `resolution` | 512×512 | **≥768×768** |
| `num_steps` | 4 | 25 |
| `negative_prompt` | 无效 | **必须使用** |

### 8.4 已知问题

#### 底部平面 artifact（Hunyuan3D 备选后端）

> 切换 TRELLIS.2 默认后端后已不再出现。以下保留作为 Hunyuan3D 参考。

- **根因**：Shape 模型训练数据偏差 — 物体都放在平面上
- **缓解**：Mesh trim（移除法向量朝下 + 最底部 5% 高度面）+ bbox fallback 物理估算

#### Sim-ready 成熟度

| 层级 | 要求 | 状态 |
|:---:|------|:---:|
| L0 | 能加载到仿真器 | ✅ |
| L1 | 碰撞生效 | ⚠️ 凸包近似 |
| L2 | 材质精确物理属性 | ❌ |
| L3 | 视觉逼真 (PBR) | ✅ TRELLIS.2 |

改进路线：P0 策划资产扩充 + 材质感知密度摩擦 → P1 URDF 颜色 + 凸分解 → P2 Genesis 材质属性。

#### bpy (Blender Python) 依赖

Hunyuan3D PBR Paint 最终步骤需 `bpy`（OBJ→GLB），但 bpy 无 Python 3.12 wheel。
解决方案：lazy-import patch + `save_glb=False` + trimesh GLB 导出。

### 8.5 mjviser 集成详情

[mjviser](https://github.com/mujocolab/mjviser) 基于 viser 的 web MuJoCo viewer。

```python
import mujoco
from mjviser import Viewer

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
Viewer(model, data).run()  # 浏览器打开 http://localhost:8080
```

三个回调：`step_fn(model, data)`、`render_fn(scene)`、`reset_fn(model, data)`。
集成计划：Step 1 单 URDF 预览 → Step 2 MJCF 多物体 → Step 3 轨迹回放 → Step 4 Policy 在线推理。

---

## 9. Genesis AMD 渲染后端调研

> 2025-04-10 调研。基于 [Genesis AMD issues](https://github.com/Genesis-Embodied-AI/Genesis/issues?q=AMD)、
> PR [#2393](https://github.com/Genesis-Embodied-AI/Genesis/pull/2393)、
> PR [#2680](https://github.com/Genesis-Embodied-AI/Genesis/pull/2680)、
> 以及本地 MI300X 实测。

### 9.1 当前后端确认

在 MI300X Docker 容器内执行 `gs.init(backend=gs.gpu)` 输出：

```
Running on [AMD Instinct MI300X] with backend gs.amdgpu. Device memory: 191.98 GB.
```

**结论**：Genesis 0.4.3 在 MI300X 上自动解析 `gs.gpu` → **`gs.amdgpu`**（非 OpenGL / Vulkan / CPU）。

### 9.2 Genesis 后端架构

Genesis 有 5 种后端（`genesis/constants.py`）：

| 后端 | 编号 | 引擎 | 用途 |
|------|:----:|------|------|
| `cpu` | 0 | Quadrants CPU | 开发调试 |
| `gpu` | 1 | 自动解析 → cuda / amdgpu / metal | 默认推荐 |
| `cuda` | 2 | Quadrants CUDA | NVIDIA GPU 物理计算 |
| `amdgpu` | 3 | Quadrants HIP (ROCm) | AMD GPU 物理计算 |
| `metal` | 4 | Quadrants Metal | macOS |

`gs.gpu` 的解析顺序：CUDA → AMDGPU → Metal → CPU fallback。
MI300X 上 ROCm/HIP 可用 → 解析为 `amdgpu`。

**物理引擎**：通过 [Quadrants](https://github.com/Genesis-Embodied-AI/quadrants)（Taichi fork）的 AMDGPU 后端，使用 HIP kernel 做物理仿真，直接跑在 GPU 上。

**渲染引擎**：Genesis 的相机渲染使用 OpenGL/EGL（pyrender）。MI300X 是 CDNA 架构（纯计算卡，无光栅化硬件），渲染走 Mesa EGL → **llvmpipe（CPU 软件光栅化）**，这是 data gen ~28s/ep 的主要瓶颈。相比之下，NVIDIA 4090 有硬件光栅化 + EGL 加速。

### 9.3 社区 MI300X 性能基准（PR #2680, v01dXYZ, 2025-04-09）

v01dXYZ 在 MI300X 上运行了 Genesis 官方 benchmark，与 RTX Pro 6000 对比：

**物理仿真 FPS（batch_size=30000, 纯物理无渲染）**：

| 场景 | MI300X (amdgpu) | RTX 6000 (cuda) | MI300X / RTX 6000 |
|------|----------------:|-----------------:|:------------------:|
| franka | 3.12M | 5.43M | 57% |
| franka_free | 4.55M | 1.83M | **249%** |
| franka_random (gjk=off) | 2.55M | 6.74M | 38% |
| franka_random (gjk=on) | 1.98M | 6.69M | 30% |
| anymal_random | 1.33M | 6.92M | 19% |
| duck_in_box_easy | 3.04M | 6.95M | 44% |

**分析**：
- 无碰撞检测场景（`franka_free`）MI300X **大幅领先**（2.5×），得益于 HBM 带宽优势
- 有碰撞检测场景 MI300X 约为 RTX 6000 的 **30-57%**，碰撞检测 kernel 尚未针对 CDNA 优化
- 编译时间 MI300X ~50-80s vs RTX 6000 ~27-45s（Quadrants JIT 编译开销更大）

### 9.4 已知 AMD 兼容性问题

| Issue | 状态 | 影响 | 描述 |
|-------|------|------|------|
| [#2570](https://github.com/Genesis-Embodied-AI/Genesis/issues/2570) | **已修复** | **MI300X** | LLVM ISel 失败：convex collision kernel 在 CDNA3 (gfx942) 上无法编译。**我们提交的 issue**。Genesis ≤0.4.3 + Quadrants ≤0.4.4 触发；**Genesis 0.4.5 + Quadrants 0.5.2 已不再复现**（[v01dXYZ 验证](https://github.com/Genesis-Embodied-AI/Genesis/issues/2570#issuecomment-4221044595) + 本地 MI300X 确认）。`--no-bbox-detection` workaround 不再需要。 |
| [#2434](https://github.com/Genesis-Embodied-AI/Genesis/issues/2434) | Open | RDNA3 | hipMemset >1GB 数组触发 `hipErrorInvalidValue`。Workaround: `device_memory_GB=20` |
| [#2669](https://github.com/Genesis-Embodied-AI/Genesis/issues/2669) | Open | gfx1150 | `hipErrorInvalidKernelFile`：ISA 不兼容，需要 `HSA_OVERRIDE_GFX_VERSION` |
| [#2680](https://github.com/Genesis-Embodied-AI/Genesis/pull/2680) | Open PR | MI300X | 修复 `rocm-smi` fallback 和 KFD sysfs 内存报告，已含 benchmark 数据 |
| [#2679](https://github.com/Genesis-Embodied-AI/Genesis/issues/2679) | Open | CI | 提议将 MI300X 加入 GitHub CI runner（Hot Aisle $1.99/hr） |
| [#2393](https://github.com/Genesis-Embodied-AI/Genesis/pull/2393) | Merged | 全平台 | 用 `amdgpu` 后端替换了有 bug 的 `vulkan` 后端 |

### 9.5 对 RoboSmith 的影响与建议

| 环节 | 当前状态 | 瓶颈 | 可能优化 |
|------|---------|------|---------|
| **物理仿真** | `amdgpu` 后端，GPU 加速 | 碰撞检测 kernel 未优化（30-57% of CUDA） | 等待 Quadrants CDNA3 优化；简单场景影响不大 |
| **相机渲染** | Mesa EGL → **llvmpipe (CPU 软件光栅化)** | **主要瓶颈**：CDNA 无光栅化硬件，~28s/ep 中大部分是 CPU 渲染开销 | 1. 减少相机分辨率 2. 减少渲染帧数 3. 未来考虑 Vulkan compute-based renderer |
| **碰撞检测** | `box_box_detection=True` **已可用** | [#2570](https://github.com/Genesis-Embodied-AI/Genesis/issues/2570) 已修复：Genesis 0.4.5 + Quadrants 0.5.2 不再触发 LLVM ISel 崩溃。300 步物理仿真通过验证，cube 位置正确 | `--no-bbox-detection` workaround 可移除，E1+ 实验启用 box_box_detection=True |
| **并行环境** | batch_size=1（当前） | 未利用 GPU 并行 | Data gen 可考虑 batch scene 加速 |

### 9.6 关键结论

1. **物理仿真走 `amdgpu` GPU 后端**，不是 CPU fallback
2. **相机渲染走 CPU（llvmpipe）**——CDNA 是纯计算架构无光栅化硬件，这是 data gen ~28s/ep 的主要瓶颈（NVIDIA 4090 有硬件光栅化 + EGL 加速，快得多）
3. **#2570 box_box_detection 已修复**：Genesis 0.4.5 + Quadrants 0.5.2 不再触发 LLVM ISel 崩溃（[v01dXYZ 确认](https://github.com/Genesis-Embodied-AI/Genesis/issues/2570#issuecomment-4221044595)）。本地 MI300X 验证 `box_box_detection=True` 通过 300 步物理仿真，cube 稳定落在桌面 (z=0.0195)。**`--no-bbox-detection` workaround 可移除**。注：@duburcqa 指出 Quadrants 底层 LLVM bug 未真正修复，只是 Genesis 源码变化使其不再触发
4. **Genesis AMD 生态正在快速演进**：PR #2393（2月）加入 amdgpu 后端，PR #2680（4月）修复兼容性并提供首批 MI300X benchmark
5. **box_box_detection 对实验的影响**：
   - **单物体 pick (Stage 1)**：影响较小，True/False 差异不大
   - **多物体交互 (Stage 2+)**：**必须启用**——box-box 专用碰撞检测对堆叠、推挤等场景的物理稳定性至关重要
   - **数据质量**：True 生成的 demonstration 物理行为更真实，VLA 策略更 robust
   - **E1+ 建议**：统一启用 `box_box_detection=True`

### 9.7 Genesis 渲染技术栈：CDNA3 vs RDNA4

> CDNA3 (MI300X/MI300X) = 纯计算卡，无图形流水线硬件。
> RDNA4 (RX 9070 XT) = 完整图形+计算架构，有光栅化/RT 硬件。

#### 系统级图形驱动支持

| 技术栈 | CDNA3 (gfx942) | RDNA4 (gfx1200) |
|--------|:--------------:|:---------------:|
| **OpenGL (radeonsi)** | **不支持** — 无光栅化硬件 | **支持** — OpenGL 4.6, Mesa radeonsi |
| **Vulkan (RADV)** | **显式拒绝** — Mesa 25.0 起 bail out ([Phoronix](https://www.phoronix.com/news/Mesa-25.0-RADV-No-CDNA)) | **支持** — Vulkan 1.4 + RT, Mesa RADV |
| **EGL headless** | Mesa EGL → **llvmpipe (CPU)** | Mesa EGL → **radeonsi (GPU 硬件加速)** |
| **ROCm compute (HIP)** | **完整支持** — 主要用途 | **支持** — ROCm 6.4.1+ ([参考](https://kaeru.my/notes/amd-radeon-9070-xt-on-ubuntu-linux-25-04-with-rocm-6-4-1-and-mesa-25)) |

#### Genesis 渲染后端适配

| Genesis 后端 | 底层技术 | CDNA3 (MI300X) | RDNA4 (RX 9070 XT) | NVIDIA (4090) |
|-------------|---------|:--------------:|:-------------------:|:-------------:|
| **Rasterizer** (默认) | OpenGL / EGL | llvmpipe (**CPU**, ~5 FPS) | radeonsi (**GPU**, ~100+ FPS) | **GPU**, ~200+ FPS |
| **RayTracer** | LuisaRender + OptiX | **不可用** (需 CUDA + OptiX) | **不可用** (LuisaCompute 无 HIP 后端) | **可用** |
| **BatchRenderer** | gs-madrona (CUDA) | **不可用** (CUDA-only) | **不可用** (CUDA-only) | **可用** |

#### CDNA3 上的渲染优化选项

| 方案 | 预计收益 | 复杂度 | 可行性 |
|------|---------|--------|--------|
| 降低分辨率 (640→320) | ~2-4× | 低 | **推荐**：像素量减 75%，llvmpipe 近线性加速 |
| 跳帧渲染 (每 N 步) | ~2-3× | 中 | 需改 data collection 逻辑 |
| HIP compute rasterizer | ~10-20× (理论) | 极高 | 用 HIP 实现软件光栅化，工程量巨大，非标准方案 |

#### RDNA4 上的渲染加速路径

| 方案 | 预计收益 vs CDNA3 llvmpipe | 复杂度 | 可行性 |
|------|:-------------------------:|--------|--------|
| **EGL + radeonsi** (Genesis Rasterizer 默认路径) | **20-60×** | **零改动** | **直接可用**：Genesis 检测到 RDNA GPU 自动走硬件 OpenGL |
| RADV Vulkan renderer (如有 Genesis 支持) | ~20-60× | 高 | Genesis 暂无 Vulkan 渲染后端 |
| LuisaRender HIP 后端 | ~20-50× | 极高 | LuisaCompute 目前仅 CUDA/DX12/Metal，无 HIP |
| gs-madrona ROCm 移植 | 待定 | 高 | gs-madrona 是 CUDA-only，需社区移植 |

#### RDNA4 作为渲染加速卡的可行性

**混合架构方案**：MI300X (CDNA3) 做物理仿真 + VLA 训练，RDNA4 (RX 9070 XT) 做渲染。

| 考量 | 评估 |
|------|------|
| **硬件可用性** | RX 9070 XT 是消费级卡 (~$550)，PCIe 插槽需要确认服务器兼容性 |
| **Genesis 支持** | Rasterizer 后端走 EGL + radeonsi，**零代码改动**即可 GPU 加速渲染 |
| **ROCm 兼容** | ROCm 6.4.1 已支持 RDNA4 ([参考](https://kaeru.my/notes/amd-radeon-9070-xt-on-ubuntu-linux-25-04-with-rocm-6-4-1-and-mesa-25)) |
| **Headless 渲染** | 需确认 RDNA4 无显示器时 EGL headless 是否正常（RDNA3 W7900 已验证） |
| **多 GPU 路由** | Genesis 物理仿真和渲染可能需要在不同 GPU 上，需验证 `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` 路由 |
| **Data gen 提速** | 渲染从 CPU ~5 FPS → GPU ~100+ FPS，**data gen 每 ep 可能从 28s 降到 3-5s** |

#### 总结

| | CDNA3 (MI300X) | RDNA4 (RX 9070 XT) | NVIDIA (4090) |
|-|:-:|:-:|:-:|
| 物理仿真 | GPU (amdgpu) | GPU (amdgpu) | GPU (cuda) |
| 渲染 | **CPU** (llvmpipe) | **GPU** (radeonsi) | **GPU** (OpenGL/EGL) |
| RayTracer | 不可用 | 不可用 | 可用 (OptiX) |
| BatchRenderer | 不可用 | 不可用 | 可用 (CUDA) |
| VLA 训练 | **最优** (192GB HBM3) | 受限 (16GB GDDR6) | 受限 (24GB GDDR6X) |

**建议**：
- **短期 (Stage 1)**：继续在 MI300X + CPU 渲染，data gen 慢但仅需跑一次，训练迭代瓶颈已解决
- **中期 (Stage 2-3)**：评估混合架构——MI300X 训练 + RDNA4 渲染加速，data gen 可提速 5-10×
- **长期**：关注 Genesis 社区 gs-madrona ROCm 移植 和 LuisaCompute HIP 后端进展

### 9.8 #2570 box_box_detection 修复验证（2025-04-10）

> 背景：[#2570](https://github.com/Genesis-Embodied-AI/Genesis/issues/2570) 是我们提交的 issue。v01dXYZ [回复](https://github.com/Genesis-Embodied-AI/Genesis/issues/2570#issuecomment-4221044595)
> 在 MI300X VF 上测试，Genesis 0.4.4+ (Quadrants 0.4.5+) 无法复现，0.4.3 及以下可复现。

**本地 MI300X 验证**：

| 测试项 | 结果 |
|--------|------|
| Genesis 版本 | 0.4.5 |
| Quadrants 版本 | 0.5.2 |
| `box_box_detection=True` + Franka + Box | **PASS** — scene.build() 成功，无 LLVM Fatal Error |
| 物理仿真 300 步 | **PASS** — cube z=0.0195 (预期 ~0.02)，稳定落在桌面 |
| scene.build() 耗时 | 27.6s（含 JIT 编译） |
| 运行 FPS | ~45-47 FPS |

**box_box_detection 功能说明**：

| | `box_box_detection=True` | `box_box_detection=False` |
|--|--|--|
| 碰撞算法 | 专用 `func_narrow_phase_convex_specializations` kernel | 通用 convex 碰撞回退 |
| Cube-Table 接触 | 精确 box-plane 接触面 | 通用凸包近似 |
| Gripper-Cube 夹持 | 精确 finger-box 侧面接触力 | 近似接触，夹持力可能不精确 |
| 多 Box 堆叠 (Stage 2) | **稳定**——精确 box-box 接触 | 可能穿模或堆叠不稳 |
| Stage 1 影响 | 较小 | 较小 |
| Stage 2+ 影响 | **必须启用** | 碰撞退化 |

**结论**：`--no-bbox-detection` workaround 不再需要。E1+ 实验及后续 Stage 2 均应启用 `box_box_detection=True`。

> ⚠ @duburcqa [指出](https://github.com/Genesis-Embodied-AI/Genesis/issues/2570#issuecomment-4221044595)：
> Quadrants 底层 LLVM ISel bug 并未真正修复，只是 Genesis 源码变化使其不再触发。
> 若未来 Genesis 更新导致该 kernel 路径变化，可能重新触发。建议持续跟踪。
