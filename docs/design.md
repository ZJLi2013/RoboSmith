# RoboSmith — Design Document

> 具身数据基础设施：资产生成 + 数据生产。
> **定位：Data Infra，不是 VLA 训练框架。** 你定义任务，我们生成资产、生产数据。你来训练 VLA。
>
> Part 1: Sim-ready 3D gen (done)
> Part 2: Data Engine — GraspPlanner + MotionExecutor + 可插拔 DataBackend (IK ✅ / DART / DAgger) + vla-eval benchmark plugin
>
> 相关文档：
> - [README.md](../README.md) — 项目概览与当前状态
> - [study.md](study.md) — 技术调研、前沿分析、实现笔记、参考链接
> - [general_object_grasp_solution.md](general_object_grasp_solution.md) — 不规则物体抓取技术方案全景（三层框架 + 横向对比）
> - [background.md](background.md) — 技术背景（水密网格、URDF、凸包近似等）
> - [part3-exp.md](part3-exp.md) — Phase 3.0 回归测试实验记录

---

## 项目总览

```
Part 1 ── 数字资产库框架 ✅
│
│   text query → search → 命中 → URDF
│                    │
│                    └→ 未命中 → T2I (SDXL-Turbo) → TRELLIS.2-4B (默认) → mesh_cleanup → URDF → 入库
│
│   入库 = assets/generated/ + catalog.json（远端 sync 回本地，.gitignore 大文件）
│   内置 26 个策划资产 (17 Objaverse + 7 primitive + 2 scene) + 场景预设
│   所有资产预计算 stable_poses (trimesh)
│   可视化: viser SceneViewer (静态预览)
│
────────────────────────────────────────────────────────────────────
│
│               ┌─────────────────── TaskSpec ───────────────────┐
│               │ (instruction + success predicates + scene)     │
│               │  composable predicates 成功判定                │
│               │  两端共享，一次定义 → 数采 + 评估同时生效         │
│               └──────────┬────────────────────┬───────────────┘
│                          │                    │
│                          ▼                    ▼
│
Part 2 ── Data Engine + Eval ✅
│
│   TaskSpec → SceneConfig → Genesis
│     └─ skills: [Skill("pick", ...), Skill("place", ...)]
│
│   Orchestration (编排: 做什么顺序)
│     run_skills() 按序消费 skill 序列
│       │
│       ├─ GraspPlanner (决策: 抓哪里)
│       │    └─ TemplateGraspPlanner ✅ (per-category template)
│       │        → GraspPlan
│       │
│       └─ MotionExecutor (执行: 怎么到达)
│            .pick() / .place()
│                → joint trajectory
│
│   DataBackend (录制: 如何生产数据)
│     ├─ IK scripted (✅ default)
│     ├─ DART (IK + noise, planned)
│     ├─ DAgger (policy + IK, planned)
│     └─ Online RL (planned)
│   → LeRobot v3.0 数据集
│
│   Eval: vla-eval benchmark plugin (RoboSmithBenchmark)
```

**验证状态**：

| 引擎 | 目标 | 状态 |
|------|------|:---:|
| **Data Engine — IK** | GraspPlanner → MotionExecutor → pick / place / stack → LeRobot 数据集 | ✅ |
| **Grasp Affordance — 架构** | GraspPlanner + MotionExecutor 解耦，TemplateGraspPlanner 实现 | ✅ |
| **Grasp Affordance — 品类扩展** | bowl / mug / bottle 等品类 GraspTemplate | 🔴 **Next** |
| **Data Engine — DART** | IK + noise 注入 + IK 重求解 → recovery data | 📋 |
| **Eval Engine** | Server-client 接口 → 并行评估 → composable predicate → eval report | 📋 |
| **闭环验证** | eval report → 诊断 → 针对性补采 / 切换后端 → 验证提升 | 📋 |

## 当前状态

| 模块 | 状态 | 说明 |
|------|:----:|------|
| 资产库 (`AssetLibrary`) | ✅ | 26 资产 (17 Objaverse + 7 primitive + 2 scene)，`objects/` + `generated/` 双目录扫描 |
| Stable poses | ✅ | 全部 26 资产预计算（trimesh, Linux MI300X），存入 `metadata.json` |
| 资产持久化 & 同步 | ✅ | `catalog.json` 索引，`.gitignore` 排除大文件，只 track json/urdf |
| 碰撞感知摆放 | ✅ | `ProgrammaticSceneBackend` + workspace_xy + stable pose 采样 + AABB/FCL 碰撞检测 |
| Genesis scene loader | ✅ | `genesis_loader.py`: ResolvedScene → Genesis entities |
| 场景预设 | ✅ | `tabletop_simple` (mug + bowl + 3 blocks, collision-aware) |
| 3D 生成后端 | ✅ | **TRELLIS.2-4B (默认, 1K PBR)** + Hunyuan3D-2.1 (备选)，MI300X 验证 |
| mesh → URDF 转换 | ✅ | trimesh 凸包 + 物理属性估算 |
| T2I 桥接 (text→image) | ✅ | SDXL-Turbo 默认，3D 友好 prompt 优化 |
| 静态场景可视化 | ✅ | viser SceneViewer |
| Task 定义系统 | ✅ | `robotsmith/tasks/`: TaskSpec + PREDICATE_REGISTRY (`object_above`, `object_in_container`, `stacked`) |
| IK 数采 (`collect_data.py`) | ✅ | TaskSpec-driven: `--task pick_cube/place_cube` dispatch，支持 pick + pick_and_place + `--scene` |
| Data Backend: DART | 📋 | IK + noise 注入 + IK 重求解，作为 `DataBackend` 可插拔后端 |
| Data Backend: DAgger | 📋 | Policy rollout + IK expert relabel，`DataBackend` 后端 |
| 多任务 IK solver | ✅ | `pick` ✅ (20ep 100%) + `pick_and_place` ✅ (20ep 100%, xy_err 4.7mm) + `stack` ✅ (20ep 90%, 3-block) |
| VLA 验证 (SmolVLA) | ✅ | baseline 已验证（单任务 cube 60%），作为数据质量验证工具 |
| **vla-eval Plugin** | ✅ | `RoboSmithBenchmark` — Genesis scene 作为 vla-eval benchmark plugin，MI300X 验证通过 |

---

# Part 1：数字资产库框架

## 1.1 架构

```
用户请求: "红色水杯" / 图片 / 文本描述
                │
                ▼
    ┌───────────────────┐
    │   资产检索引擎      │  tag 匹配 (MVP) / embedding 语义 (后续)
    └────────┬──────────┘
             │
     ┌───────┴────────┐
     │                │
  命中 ✅           未命中 ❌
     │                │
     ▼                ▼
  返回资产     ┌──────────────┐
  (URDF)      │ T2I 桥接      │  text → 参考图 (SDXL-Turbo)
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │ TRELLIS.2-4B │  image → GLB (PBR)
              └──────┬───────┘
              (备选: Hunyuan3D-2.1)
                     ▼
              ┌──────────────┐
              │ mesh_to_urdf │  GLB + OBJ 凸包碰撞 + 物理属性
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │ 自动入库       │  tag + metadata.json + catalog.json
              └──────────────┘
```

### 核心接口

```python
from robotsmith.assets import AssetLibrary

lib = AssetLibrary("./assets")
cup = lib.search("红色水杯")
if cup is None:
    cup = lib.generate("红色水杯", image_path="mug.png")

cup.urdf_path     # assets/generated/red_ceramic_mug_.../model.urdf
cup.metadata      # {"mass": 0.25, "friction": 0.6, "source": "generated", ...}
```

### 资产存储

```
assets/
├── objects/          # 内置/策划资产（git tracked）
│   ├── mug_red/
│   │   ├── model.urdf
│   │   └── metadata.json
│   └── ...
├── generated/        # 管线生成资产（git ignored）
└── catalog.json      # 轻量索引 (git tracked)
```

## 1.2 内置资产

26 个策划资产：17 Objaverse 导入 + 7 primitive URDF + 2 scene 基础设施。
10 品类桌面操作物品，覆盖 7 种基础几何拓扑（球/椭球、圆柱、长方体、凹形、扁平、不规则、非凸）。

| 来源 | 用途 | 质量 | 底座 artifact |
|------|------|:---:|:---:|
| **Objaverse 策划** | 高频类目（10 种桌面物品） | ★★★★☆ | **无** |
| **TRELLIS.2 生成** ← 默认兜底 | 搜索未命中的长尾类目 | ★★★★☆ | **无** |
| Hunyuan3D 生成 ← 备选 | 备选后端 | ★★★☆☆ | 有（需 trim） |
| 程序化原语 | 仅供 table / ground plane / block | ★☆☆☆☆ | 无 |

> 资产详细品类规划、尺寸预算、来源策略见 [study.md §8.1](study.md#81-内置资产详细规划)。

### 场景预设

| 场景 | 描述 | 包含物品 |
|------|------|---------|
| tabletop_simple | 1 桌 + 3-5 物品随机摆放 | 杯、碗、积木 |

## 1.3 3D 生成管线

### 默认后端：TRELLIS.2-4B

| 项目 | 值 |
|------|-----|
| 模型 | [TRELLIS.2-4B](https://github.com/microsoft/TRELLIS.2) (Microsoft, 4B, CVPR'25) |
| 输入 | 单张参考图片 (image-to-3D) |
| 输出 | GLB (PBR 纹理嵌入)，visual.glb + collision.obj |
| ROCm | **✅ 已验证** — MI300X, ROCm 6.4 |
| 优势 | 无底座 artifact · PBR 纹理 · mesh 质量高 · 无 bpy 依赖 |

纹理分辨率预设：**fast** (512px, ~2MB) / **balanced** (1K, ~8MB, 默认) / **high** (4K, ~38MB)。

### 备选后端：Hunyuan3D-2.1

| 项目 | 值 |
|------|-----|
| 模型 | [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) (Tencent, 3.3B shape + 2B PBR) |
| ROCm | **✅ 已验证** |
| 限制 | 内壁纹理破碎 · 底座 artifact · bpy 依赖 (已绕过) |

> 后端对比详情见 [study.md §8.2](study.md#82-trellis2-vs-hunyuan3d-e2e-对比red-ceramic-mug-mi300x-512³)。

### Mesh → Sim-ready 转换

GLB mesh（任何后端输出）→ sim-ready URDF，由 `mesh_to_urdf` 统一处理：

| 步骤 | 工具 | 说明 |
|------|------|------|
| 尺度标准化 | trimesh | 归一到物理尺寸 (meters) |
| 居中 | trimesh | 质心平移至原点 |
| 碰撞体生成 | trimesh convex_hull | 凸包近似 (MVP)；后续 V-HACD / CoACD |
| 物理属性 | 自动估算 | 体积 → 质量 (800 kg/m³)，惯性张量 |
| URDF 打包 | `mesh_to_urdf` | visual.glb + collision.obj + model.urdf + metadata.json |

### Sim-ready 成熟度

```
层级   要求                          状态
────   ────                          ──────
L0     能加载到仿真器                 ✅
L1     碰撞生效                       ⚠️ 凸包近似（凹面丢失）
L2     物理属性精确                   ❌ 待实现
L3     视觉逼真（PBR）               ✅ TRELLIS.2 PBR (1K 默认, 可选 512/4K)
```

## 1.4 T2I 桥接

所有 SOTA 3D 生成模型均为 image-to-3D，不支持纯文本。
未命中检索时，由 T2I 自动生成参考图片。

| 模型 | ROCm | 许可 | 备注 |
|------|:---:|------|------|
| **SDXL-Turbo** ← 默认 | ✅ | Apache-2.0 | 4步, 0.3s/img, 8GB |
| FLUX.1-dev ← 升级路径 | ⚠️ gated | Non-commercial | 12B, 质量更高 |

> Prompt 工程详情（约束模板、反向 prompt、参数、踩坑修复）见 [study.md §8.3](study.md#83-t2i-prompt-工程)。

## 1.5 资产检索

Tag 精确匹配（已实现）。后续加 CLIP/SentenceTransformer embedding 语义检索。

---

# Part 2：Data Engine — 可插拔数据生产

> 核心：TaskSpec 驱动的声明式数据生产，后端可插拔。
> 数据管线 = **GraspPlanner**（抓哪里）→ **MotionExecutor**（怎么到达）→ **DataBackend**（如何录制）。
>
> 数据生产不只是 IK scripting — 真正的生产痛点是如何为 VLA 生成 **高质量、高多样性、含纠错行为** 的训练数据。
> RoboSmith 将数据生产抽象为 `DataBackend` ABC，不同策略作为可插拔后端：

### 数据生产后端（Data Production Backends）

| 后端 | 方法 | 数据特征 | 适用场景 | 状态 |
|------|------|---------|---------|:---:|
| **IK scripted** | 开环 IK waypoint 回放 | 最优轨迹，无纠错 | 初始数据集、seed data | ✅ Done |
| **DART** | IK + 噪声注入 + IK 重求解 | 含 recovery supervision | 提升 policy 鲁棒性 | 📋 Planned |
| **DAgger** | Policy rollout + IK expert relabel | 针对 policy 薄弱环节 | 缩小 distribution gap | 📋 Planned |
| **Online RL** | Policy exploration + reward signal | 自主探索最优策略 | 精细操作、长程任务 | 📋 Planned |

> **渐进关系**：IK 是所有后端的基础 — DART = IK + noise，DAgger = policy + IK oracle，Online RL 可用 IK 作 warm-start。
> 每个后端产出统一的 LeRobot v3.0 数据集，下游 VLA 训练和评估无需感知数据来源。

## 2.1 仿真平台

| 平台 | 优势 | AMD | 定位 |
|------|------|:---:|------|
| **MuJoCo** | 精确接触、URDF 直接加载、生态成熟 | ✅ | **主力引擎** |
| **Genesis** | GPU 并行、速度快 | ✅ | 大规模采集加速 |


URDF → MuJoCo 原生支持加载（自动转 MJCF），Part 1 资产无需额外格式转换。
仿真可视化：[mjviser](https://github.com/mujocolab/mjviser)（web MuJoCo viewer），详见 [study.md §8.5](study.md#85-mjviser-集成详情)。

## 2.2 Task 定义系统 + IK 数采

### TaskSpec

参考 [RoboLab](https://github.com/NVLabs/RoboLab) 的 composable predicates 设计，
每个任务通过 `TaskSpec` dataclass 声明式定义：

```python
@dataclass
class TaskSpec:
    name: str                          # 任务名
    instruction: str                   # 语言指令
    scene: str                         # SceneConfig preset 名称
    contact_objects: list[str]         # 需要接触检测的物体
    success_fn: str                    # 成功判定谓词名（PREDICATE_REGISTRY key）
    success_params: dict               # 谓词参数（纯 dict，JSON 可序列化）
    motion_type: str = "pick"          # "pick" | "pick_and_place"
    episode_length: int = 200          # 最大 step 数
    dart_sigma: float = 0.0            # DART 噪声 (0 = 纯 IK)
    grasp_planner: str = "template"    # GraspPlanner 后端名
    is_stack: bool = False
    n_stack: int = 3
```

> **设计选择**：`success_fn` 用注册表 key（str）而非 lambda/Callable，借鉴 RoboLab 的
> `TerminationTermCfg(func=named_fn, params=dict)` 模式。
> 天然可序列化，可跨进程传递给 eval Docker 容器，可写入 eval report。
> 完整演进路线见 [核心抽象 → TaskSpec 演进设计](#演进设计渐进式)。

### Composable Predicates

成功/失败判定通过可组合的谓词函数定义，不再 hardcode：

| 谓词 | 含义 | 用于任务 |
|------|------|---------|
| `object_above` | 物体高于参考面 | pick |
| `object_in_container` | 物体在容器内 | pick-and-place |
| `object_on_top` | 物体稳定放置在另一物体上 | stacking |
| `stacked` | 多物体按序堆叠 | stacking |
| `object_upright` | 物体正立 | mug upright |

**组合逻辑**：`all(objects)` / `any(objects)` / `choose(objects, k)`

```python
# pick: 抓起 cube
TaskSpec(
    name="pick_cube",
    instruction="Pick up the cube",
    scene="tabletop_simple",
    contact_objects=["cube", "table"],
    success_fn="object_above",
    success_params={"object": "cube", "reference": "table", "z_margin": 0.05},
    motion_type="pick",
)

# pick-and-place: 把 mug 放进 bowl
TaskSpec(
    name="mug_in_bowl",
    instruction="Place the mug in the bowl",
    scene="tabletop_simple",
    contact_objects=["mug", "bowl", "table"],
    success_fn="object_in_container",
    success_params={"object": "mug", "container": "bowl"},
    motion_type="pick_and_place",
)

# stacking: 3 blocks 堆叠
TaskSpec(
    name="stack_blocks",
    instruction="Stack the red, green, and blue blocks",
    scene="tabletop_simple",
    contact_objects=["block_red", "block_green", "block_blue", "table"],
    success_fn="stacked",
    success_params={"objects": ["block_red", "block_green", "block_blue"]},
    motion_type="pick_and_place",
    is_stack=True,
    n_stack=3,
)
```

### 多任务 Motion Execution

每种任务类型对应一套 waypoint 生成逻辑（由 `MotionExecutor` 执行）：

| 任务类型 | Waypoint 序列 | 状态 |
|---------|-------------|:---:|
| `pick` | reach → pre-grasp → grasp → lift (135 frames) | ✅ 20ep 100% |
| `pick_and_place` | pick + transport → pre-place → place → release → retreat (225 frames) | ✅ 20ep 100% |
| `stack` | N 轮 pick_and_place，每轮 place_z 递增 (N×225 frames) | ✅ 20ep 90% (N=3) |
| `push` | approach → contact → slide → release | 📋 |

### 数据管线架构

```
                    ┌─────────────────────────────────────────────┐
                    │              Data Gen Pipeline               │
                    │                                              │
                    │  ┌──────────────────────────────────────┐   │
                    │  │  Orchestration Layer                   │   │
                    │  │  "做什么顺序？"                        │   │
                    │  │                                      │   │
                    │  │  TaskSpec.skills → run_skills()       │   │
                    │  │    按序消费 [pick, place, ...]         │   │
                    │  └───────────────┬──────────────────────┘   │
                    │        每个 skill │ 调用 ↓                    │
                    │  ┌──────────────────────────────────────┐   │
                    │  │  Grasp Planning Layer                 │   │
                    │  │  "抓哪里？用什么姿态？"                │   │
                    │  │                                      │   │
                    │  │  GraspPlanner (ABC)                  │   │
                    │  │    └─ TemplateGraspPlanner ✅         │   │
                    │  │        (per-category 人工 template)   │   │
                    │  │                                      │   │
                    │  │  输出: GraspPlan                      │   │
                    │  └───────────────┬──────────────────────┘   │
                    │                  │                           │
                    │                  ▼                           │
                    │  ┌──────────────────────────────────────┐   │
                    │  │  Motion Execution Layer ✅             │   │
                    │  │  "怎么到达那个 pose？"                 │   │
                    │  │                                      │   │
                    │  │  MotionExecutor                      │   │
                    │  │    .pick(GraspPlan)                  │   │
                    │  │    .place(GraspPlan)                 │   │
                    │  └───────────────┬──────────────────────┘   │
                    │                  │                           │
                    │                  ▼                           │
                    │           joint trajectory                   │
                    │                  │                           │
                    │     DataBackend.collect() 执行 + 录制         │
                    │       ├─ IK Scripted: 直接执行 ✅             │
                    │       ├─ DART: + noise + re-solve 📋         │
                    │       └─ DAgger: policy rollout + relabel 📋 │
                    │                  │                           │
                    │                  ▼                           │
                    │         LeRobot Dataset                      │
                    └─────────────────────────────────────────────┘
```

三层职责分离：Orchestration 决定 "做什么顺序"，GraspPlanner 决定 "抓哪里"，MotionExecutor 决定 "怎么到达"。
新增 task = 定义 skill 序列，新增品类 = 注册 GraspTemplate，新增执行方式 = 添加 DataBackend。三者正交。

### DataBackend ABC — 可插拔数据生产

所有数据生产后端实现统一接口：

```python
class DataBackend(ABC):
    @abstractmethod
    def collect(
        self, task: TaskSpec, scene: ResolvedScene, n_episodes: int,
        rng: np.random.Generator, **kwargs,
    ) -> LeRobotDataset:
        """生产 n_episodes 条数据，返回统一 LeRobot 数据集。"""

class IKScriptedBackend(DataBackend):     # ✅ Done — 当前 collect_data.py 核心逻辑
class DARTBackend(DataBackend):           # 📋 — IK + noise injection + IK re-solve
class DAggerBackend(DataBackend):         # 📋 — policy rollout + IK expert relabel
class OnlineRLBackend(DataBackend):       # 📋 — policy exploration + reward signal
```

**后端间关系（渐进增强）**：

```
IK scripted (seed data, 最优轨迹，无纠错)
    │
    ├── + noise injection ──→ DART (含 recovery supervision, 开环)
    │
    └── + policy-in-the-loop ──→ DAgger (闭环 relabel)
                                    │
                                    └── + reward shaping ──→ Online RL
```

#### IK Scripted Backend (当前实现)

开环 IK waypoint 回放：`GraspPlanner` 生成 `GraspPlan`，`MotionExecutor` 转为 joint trajectory，直接执行。
适用于初始数据集生产、seed data、策略验证。详见 `collect_data.py`。

#### DART Backend (Planned)

在 IK 执行时注入高斯扰动，然后从扰动后状态重新 IK 求解 action label。
数据天然包含 recovery supervision — policy 不参与，纯数据增强。

可选结构化噪声（`--dart-schedule structured`），按轨迹阶段施加不同强度：

| 阶段 | sigma | 原因 |
|------|:---:|------|
| Approach | 0.01-0.02 | 覆盖 almost-aligned 状态 |
| Pre-grasp | 0.005 | 覆盖 partial miss |
| Grasp-to-lift | 0.002 | 夹取时容错低 |
| Transport | 0.01 | lift 后容错高 |

#### DAgger Backend (Planned)

Policy rollout + IK expert relabel。需要当前 policy checkpoint：

1. 用当前 policy 在 sim 中 rollout
2. 对 policy 轨迹中每个状态，用 IK solver 重新标注 action
3. 合并新数据到训练集，重新训练
4. 迭代直到收敛

#### 概念辨析：IK / DART / DAgger

| 维度 | IK (开环) | DART (开环 + noise) | DAgger (闭环) |
|------|-----------|---------------------|--------------|
| **老师** | IK solver | IK solver | IK solver |
| **数据来源** | 老师最优轨迹 | 老师扰动轨迹 + 重求解 | **学生轨迹** + 老师标注 |
| **学生参与** | ❌ | ❌ | ✅ policy-in-the-loop |
| **纠错行为** | 无 | 隐式（噪声后重求解） | 显式（relabel 学生错误） |
| **适用阶段** | seed data | 增强鲁棒性 | 缩小 distribution gap |

### 数采流程总览

```
TaskSpec (声明式定义)
    │
    ├── scene → SceneConfig → ProgrammaticSceneBackend → ResolvedScene
    ├── grasp_planner → GraspPlanner → GraspPlan
    ├── motion_type → MotionExecutor.pick() / .pick_and_place() → waypoints
    └── success_fn + success_params → PREDICATE_REGISTRY → 自动成功/失败判定
    │
    ▼
DataBackend.collect(task, scene, n_episodes)
    │  ┌─ IKScriptedBackend: 直接执行 IK waypoints
    │  ├─ DARTBackend: IK waypoints + noise + re-solve
    │  └─ DAggerBackend: policy rollout + IK relabel
    ▼
LeRobot v3.0 数据集 (统一输出，下游无需感知生产方式)
```

## 2.3 任务规划

### 交互类型分类（社区 benchmark 对齐）

> 参考 LIBERO (130 tasks)、RoboEval (28 tasks)、Colosseum (20 tasks)、ManipArena (20 tasks) 的任务谱系。

| 交互类型 | 代表任务 | IK 建模方式 | 资产需求 | RoboSmith 状态 |
|---------|---------|-----------|---------|:---:|
| **抓取 (grasp)** | pick, lift | 位置 IK waypoint | primitive / Gen2Sim | ✅ `pick_cube` |
| **抓放 (pick-and-place)** | place, put-in-container | 位置 IK + transport | primitive / Gen2Sim | ✅ `place_cube` |
| **多步抓放 (sequential)** | stack, sort-by-color | 多轮 pick_and_place | 积木、多色物体 | ✅ `stack_blocks` |
| **铰接操作 (articulated)** | open/close drawer, door | 弧形轨迹 IK | **铰接 URDF** (新资产类型) | 📋 |
| **推/滑 (push/slide)** | push-to-target, sweep | 力控 / impedance control | 瓶、罐 | 📋 需力控支持 |
| **工具使用 (tool-use)** | press button, turn knob | 精细位置 IK | 按钮、旋钮 URDF | 📋 |
| **双臂协作 (bimanual)** | handover, bimanual-lift | 双臂 IK 协调 | 双机器人 | 📋 长期 |

### RoboSmith 任务路线

**Phase 1（已完成）**：抓握三件套 — 验证 TaskSpec + 数采框架

| 任务 | motion_type | 轨迹 | 状态 |
|------|-----------|:---:|:---:|
| `pick_cube` | `pick` | 135 frames | ✅ 20ep 100% |
| `place_cube` | `pick_and_place` | 225 frames | ✅ 20ep 100% |
| `stack_blocks` | `pick_and_place` (is_stack) | 675 frames (3×225) | ✅ 20ep 90% |

**Phase 2**：铰接 + 推滑 — 扩展交互类型覆盖

| 任务 | 新增需求 | 优先级 |
|------|---------|:---:|
| `open_drawer` | 铰接 URDF + 弧形 IK | 高 |
| `push_to_target` | Genesis 力控 API | 中 |
| `sort_by_color` | 条件分支逻辑 + 多色物体 | 中 |

**Phase 3**：精细操作 + 双臂 — 长期扩展

| 任务 | 新增需求 |
|------|---------|
| `press_button` | 精细位置 IK + 小型交互件 URDF |
| `turn_knob` | 旋转轨迹 IK + 旋钮 URDF |
| `bimanual_handover` | 双臂配置 + 协调 IK |

## 2.4 调研结论

> 详细调研已移至 [study.md](study.md)，此处仅保留关键结论。

1. **IK solver 是一切高级策略的前提** — DAgger 需要 IK 作为 expert oracle，RECAP 需要 IK 生成 seed data
2. **合成数据 ≥ 真实数据**已被工业级实验验证（Genie Sim Sim2Real: 82.8% > 75.2%）
3. **Task Coverage > Object Variation > Data Volume** — 先扩展任务类型广度，再提升数采策略层级
4. **AMD ROCm 全栈**是 RoboSmith 的唯一性差异化
5. **Gen2Sim 覆盖长尾物品**是 Real2Sim 的天然补充

> 场景多样性方案（碰撞感知摆放 ✅ + stable pose ✅ + Genesis loader ✅）详见 [study.md §2](study.md#2-场景多样性--借鉴-graspvla-playground)。
> RoboLab 启发分析详见 [study.md §3](study.md#3-robolab-启发分析)。

## 2.5 Grasp Planning 设计原理

### 为什么分离 Grasp Planning 和 Motion Execution

IK solver 解决 "几何可达性"（给定 pose 求关节角），但真实抓取是 "语义 + 接触 + 几何" 的组合问题。
业界 SDG 方法论（grasp distribution, affordance-driven, policy-in-the-loop 等）中，IK solver 始终存在，但角色是 "最后一公里执行"，不是规划层。

RoboSmith 因此将管线分为 `GraspPlanner`（决策: 抓哪里）→ `GraspPlan`（契约）→ `MotionExecutor`（执行: 怎么到达），
新增物体品类只需提供 `GraspTemplate`，无需改动执行逻辑。

> 不规则物体抓取技术方案全景见 [general_object_grasp_solution.md](general_object_grasp_solution.md)。

### 品类覆盖与限制

当前 `TemplateGraspPlanner` 已覆盖 block/cube（top-down 抓取），但碗、杯子、瓶子等品类需要不同的抓取姿态、接近方向和 finger width，须逐品类添加 `GraspTemplate`。

| 品类 | 抓取方式 | 与 block 的差异 |
|------|---------|---------------|
| bowl | top-down, scaled finger | 直径超 Franka 最大开口，需缩放或边缘抓取 |
| mug | side, handle | 侧面接近，`ee_quat` 旋转 90° |
| bottle | side, mid-body | `grasp_z` 随物体高度变化 |
| plate | top-down, rim pinch | 薄 + 平，`finger_closed` 极小 |
| figurine | top-down (fallback) | 不规则凸包，后续升级到 `SamplerGraspPlanner` |

### pick_cube 全链路示例

以最简 task 展示 `TemplateGraspPlanner` → `GraspPlan` → `MotionExecutor` 数据流：

**1. Planner 查表** — `planner.plan(object_pos=[0.55, 0.0, 0.02], category="block")` 匹配 block template：

```
GraspTemplate(block):
  ee_quat       = [0, 1, 0, 0]   ← EE 朝下
  finger_open   = 0.04m
  finger_closed = 0.01m
  grasp_z       = 0.135m          ← 绝对 Z
  hover_z       = 0.25m
  retreat_z     = 0.30m
```

**2. Planner 输出 GraspPlan** — template 绝对 Z + 物体 XY → 世界坐标 pose：

```
GraspPlan:
  pre_grasp_pos = [0.55, 0.0, 0.25]   ← hover
  grasp_pos     = [0.55, 0.0, 0.135]  ← grasp
  retreat_pos   = [0.55, 0.0, 0.30]   ← lift
  grasp_quat    = [0, 1, 0, 0]
```

**3. Executor 消费 GraspPlan** — IK 求解 + 线性插值：

```
home ──40steps──→ pre_grasp (open)
     ──30steps──→ grasp (open)       ← descend
     ──20steps──→ grasp (closed)     ← close fingers
     ──30steps──→ retreat (closed)   ← lift
     ──15steps──→ hold               Total = 135 frames
```

`pick_and_place` 追加 transport → place → release（共 225 frames）。
`stack_blocks` 循环 N 次 pick_and_place，每轮 `place_z += 0.04m`（共 N×225 frames）。
三个 task 共享同一个 block template，差异全在编排层。

## 2.6 Task Orchestration

Task = 原子 Skill 序列，通用 runner 按序执行。

### Skill 定义

```python
@dataclass
class Skill:
    name: str               # "pick" | "place"
    target: str             # object name
    category: str           # GraspTemplate category
    params: dict = field(default_factory=dict)
```

### Task = Skill 序列

`TaskSpec.skills` 替代 `motion_type` + `is_stack` + `n_stack`：

```python
# pick_cube
skills = [Skill("pick", "cube", "block")]

# pick_and_place
skills = [Skill("pick", "cube", "block"), Skill("place", "target", "block")]

# stack_blocks (3 层)
skills = [
    Skill("pick", "block_red",   "block"),
    Skill("place", "stack_center", "block", {"place_z": 0.15}),
    Skill("pick", "block_green", "block"),
    Skill("place", "stack_center", "block", {"place_z": 0.19}),
    Skill("pick", "block_blue",  "block"),
    Skill("place", "stack_center", "block", {"place_z": 0.23}),
]

# 新品类 — 只改 category:
skills = [Skill("pick", "bowl", "bowl")]
skills = [Skill("pick", "mug",  "mug")]
```

### 通用 Runner

```python
def run_skills(skills, planner, executor, solve_ik, scene_state, params):
    traj, qpos = [], scene_state["home_qpos"]
    for skill in skills:
        obj_pos = scene_state["positions"][skill.target]
        if skill.name == "pick":
            plan = planner.plan(obj_pos, category=skill.category)[0]
            traj += executor.pick(plan, solve_ik, qpos, params)
        elif skill.name == "place":
            plan = planner.plan_place(obj_pos, category=skill.category,
                                      place_z_override=skill.params.get("place_z"))
            traj += executor.place(plan, solve_ik, qpos, params)
        qpos = traj[-1]
    return traj
```

新增 task = 定义 skill 列表。新增品类 = 注册 `GraspTemplate`。两者正交。

### MotionExecutor 拆分

当前 `.pick_and_place()` 需拆为 `.pick()` + `.place()`（place = transport → descend → release → retreat），
使每个 Skill 对应一个 executor 方法。纯拆分重构，轨迹数学不变。

### 模块结构

```
robotsmith/orchestration/
├── __init__.py             # run_skills()
└── skills.py               # Skill dataclass
```

---

# 路线图与验证状态

## GraspPlanner 品类扩展

> 架构重构已完成并验证（[part3-exp.md](part3-exp.md)），当前瓶颈为品类 template 覆盖。

| 阶段 | 内容 | 状态 |
|------|------|:---:|
| block / cube template | top-down 抓取，验证架构 | ✅ |
| bowl / can / bottle / mug / plate / fruit | per-category `GraspTemplate` | 🔴 **Next** |
| `SamplerGraspPlanner` | mesh antipodal sampling，零人工 | 📋 |
| `LearnedGraspPlanner` | AnyGrasp / model prediction | 📋 远期 |

> 社区参考：[ShapeGen](https://wangyr22.github.io/ShapeGen/)（清华, 2026）— per-category template shape + spatial warping 对齐 RoboSmith 的 per-category `GraspTemplate` 方案。

## Phase 3.0 回归测试

> 详见 [part3-exp.md](part3-exp.md)。

| 任务 | Phase 3.0 (10ep) | Part 2 Baseline (20ep) | 判定 |
|------|:---:|:---:|:---:|
| `pick_cube` | **100%** (10/10) | 100% (20/20) | ✅ 无退化 |
| `place_cube` | **100%** (10/10) | 100% (20/20) | ✅ 无退化 |
| `stack_blocks` | **100%** (10/10) | 90% (18/20) | ✅ 无退化 |

## 成功标准

| 指标 | Part 2 | Phase 3.0 ✅ | Phase 3.1 | Phase 3.2 |
|------|:---:|:---:|:---:|:---:|
| 可操作品类 | 1 (cube) | 1（验证正确性） | ≥ 6 | ≥ 6 |
| 代码可扩展性 | 每品类需新 strategy | 每品类只需 1 个 GraspTemplate | 零人工（sampler） | — |
| IK 数采成功率 | 100% (cube) | 100% (cube) ✅ | ≥ 80% (per-category 平均) | ≥ 70% |
| 单品类 grasp 多样性 | 1 pose | 1 pose | 1~3 (multi-template) | ≥ 10 candidates |

---

# 扩展方向

## 扩展 A：RL Post-Training（RECAP）

对标 [π\*0.6 RECAP](https://www.pi.website/blog/pistar06)。
与 DAgger 的核心区别：DAgger 用 expert label 替换 policy action（"你应该做什么"），
RECAP 用 advantage signal 标注 policy action（"你做的好不好"），让 policy 自己学。
可作为未来 `OnlineRLBackend` 的实现参考。

## 扩展 B：更多 VLA 验证目标

所有新 VLA models 通过 vla-eval-harness 引入，RoboSmith 不增加 model server：

| 模型 | 参数量 | 引入方式 |
|------|--------|---------|
| **OpenVLA / StarVLA / π0 / GR00T** | 各异 | vla-eval 内置 model server |
| 任何新 VLA | — | 通过 vla-eval 社区贡献 |

## 扩展 C：World Model / Self-Play

| 方向 | 代表工作 |
|------|---------|
| World Model 条件化 | [GigaBrain RAMP](https://gigaai.cc/blog) |
| Self-Play 数据飞轮 | [DexFlyWheel](https://arxiv.org/abs/2509.23829) |

---

# 核心抽象

> 本章梳理 RoboSmith 全项目的核心类和接口。
> 目的：（1）从设计角度理解代码；（2）明确 TaskSpec 等关键抽象的演进路线。

## 概览

```
                            ┌───────────────────────┐
                            │       TaskSpec         │  ← 全局枢纽
                            │  (声明式任务定义)       │
                            └──────┬────────────────┘
                                   │
     ┌─────────────────────────────┼─────────────────────────┐
     │                             │                         │
┌────▼─────┐   ┌─────────────────▼──────────────────┐  ┌───▼─────────────┐
│ Part 1   │   │         Data Gen Pipeline           │  │ Eval            │
│ Assets   │   │                                     │  │ (vla-eval)      │
└──────────┘   │  GraspPlanner → GraspPlan           │  └─────────────────┘
Asset          │       │                             │  RoboSmithBenchmark
AssetMetadata  │       ▼                             │
AssetLibrary   │  MotionExecutor → joint trajectory  │
GenBackend     │       │                             │
               │  DataBackend.collect() 执行 + 录制   │
               │    ├ IKScripted ✅                   │
               │    ├ DART 📋                        │
               │    ├ DAgger 📋                      │
               │    └ OnlineRL 📋                    │
               │       │                             │
               │       ▼                             │
               │  LeRobot Dataset                    │
               └─────────────────────────────────────┘
```

## Part 1：Asset Layer

### `Asset` / `AssetMetadata`

```
robotsmith/assets/schema.py
```

| 类 | 说明 |
|---|---|
| `AssetMetadata` (dataclass) | 物理属性 + catalog 元数据（mass, friction, restitution, size_cm, tags, stable_poses） |
| `Asset` (dataclass) | Sim-ready 资产：URDF 路径、mesh 路径、AssetMetadata。所有下游模块引用资产的唯一句柄 |

### `AssetLibrary`

```
robotsmith/assets/library.py
```

目录驱动的资产库。管理 `objects/`（内置）和 `generated/`（生成）两套资产源，
提供 tag 检索（`search()`）、入库（`add()`）、catalog.json 持久化。

### `GenBackend` (ABC)

```
robotsmith/gen/backend.py
```

3D 生成后端抽象。`text/image → trimesh.Trimesh`。

| 实现 | 文件 | 状态 |
|------|------|:---:|
| `Trellis2Backend` | `trellis2_backend.py` | ✅ |
| `Hunyuan3DBackend` | `hunyuan3d_backend.py` | ✅ 默认 |
| `TripoSGBackend` | `triposg_backend.py` | 📋 stub |

注册表模式：`register_backend` / `get_backend` / `list_backends`。

## Grasp Planning Layer

> Part 3 新增。将 "抓哪里" 的决策从 IKStrategy 中分离为独立模块。

### `GraspPlan`

```
robotsmith/grasp/plan.py
```

两层之间的唯一契约 — planner 不关心 IK，executor 不关心物体语义。

```python
@dataclass
class GraspPlan:
    grasp_pos: np.ndarray          # 抓取点 (world frame)
    grasp_quat: np.ndarray         # EE 姿态 (wxyz)
    pre_grasp_pos: np.ndarray      # 预抓取悬停点
    pre_grasp_quat: np.ndarray
    retreat_pos: np.ndarray        # 撤退点（抓起后）
    retreat_quat: np.ndarray
    finger_open: float             # 接近时 finger width
    finger_closed: float           # 夹紧时 finger width
    quality: float = 1.0           # 抓取质量评分 (0–1)
    metadata: dict                 # 来源信息 (template/sampler/model)
```

### `GraspPlanner` (ABC) + 实现

```
robotsmith/grasp/planner.py            — GraspPlanner ABC
robotsmith/grasp/template_planner.py   — TemplateGraspPlanner + GRASP_TEMPLATES
```

| 实现 | 方法 | 状态 |
|------|------|:---:|
| `TemplateGraspPlanner` | per-category 人工模板 → `GraspPlan` | ✅ |
| `SamplerGraspPlanner` | mesh antipodal sampling → N 个 candidates | 📋 |
| `LearnedGraspPlanner` | AnyGrasp / model prediction | 📋 |

```python
class GraspPlanner(ABC):
    @abstractmethod
    def plan(self, asset: Asset, object_pos, object_quat, rng) -> list[GraspPlan]: ...
```

**TemplateGraspPlanner** 从 `GRASP_TEMPLATES[category]` 查找模板，将 template + 运行时 object pose 转为 `GraspPlan`。

### `GraspTemplate`

```
robotsmith/grasp/template_planner.py
```

Per-category 参数化模板，定义一个品类的抓取姿态、finger width、关键 Z 高度：

```python
@dataclass
class GraspTemplate:
    category: str
    grasp_type: str                  # "top_down", "side", "rim"
    approach_axis: np.ndarray        # unit vec, EE approach direction
    ee_quat: np.ndarray              # EE quaternion at grasp (wxyz)
    finger_open: float
    finger_closed: float
    grasp_z: float = 0.135           # absolute Z of grasp
    hover_z: float = 0.25            # absolute Z of pre-grasp hover
    retreat_z: float = 0.30          # absolute Z of post-grasp retreat
    place_z: float = 0.15            # absolute Z of place point
    requires_scale: bool = False
    scale_range: tuple[float,float] = (1.0, 1.0)
```

已注册 template：

| Category | grasp_type | ee_quat | finger_open | 状态 |
|----------|-----------|:---:|:---:|:---:|
| block | top_down | `[0,1,0,0]` | 0.04 | ✅ |
| cube | top_down | `[0,1,0,0]` | 0.04 | ✅ |

待注册 template（Phase 3.1）：

| Category | grasp_type | ee_quat | 说明 |
|----------|-----------|:---:|------|
| fruit | top_down | `[0,1,0,0]` | 小球体 |
| bowl | top_down | `[0,1,0,0]` | `requires_scale=True` |
| can / bottle | side | `[0.5,0.5,-0.5,0.5]` | EE 旋转 90°，抓中段 |
| mug | side | `[0.5,0.5,-0.5,0.5]` | grasp_offset 偏向 handle |
| plate | top_down | `[0,1,0,0]` | rim pinch, `finger_closed` 极小 |

不同物体的差异体现在 `GraspPlanner` 输出不同的 `GraspPlan`，`MotionExecutor` 保持通用：

```
碗:   TemplateGraspPlanner → GraspPlan(top-down, scaled)  → MotionExecutor.pick()
杯子: TemplateGraspPlanner → GraspPlan(side, handle)      → MotionExecutor.pick()
瓶子: SamplerGraspPlanner  → GraspPlan(side, mid-body)    → MotionExecutor.pick()
玩具: LearnedGraspPlanner  → GraspPlan(best candidate)    → MotionExecutor.pick()
                                                              ↑ 同一个执行器
```

## Motion Execution Layer

### `MotionParams`

```
robotsmith/motion/params.py
```

纯运动时间参数（step counts），不含抓取决策。

### `MotionExecutor`

```
robotsmith/motion/executor.py
```

给定 `GraspPlan` + `MotionParams` + IK solver，生成 joint-space 轨迹。
对所有物体通用 — 只关心 "从 pre_grasp 到 grasp_pos 用什么姿态"。

| 方法 | 轨迹结构 | 等价旧代码 |
|------|---------|-----------|
| `pick()` | home → pre_grasp → grasp (close) → retreat | `PickStrategy.plan()` |
| `pick_and_place()` | pick + transport → pre_place → place (open) → retreat | `PickAndPlaceStrategy.plan()` |

旧 `StackStrategy` 不再是独立 class — 调用方循环 N 次 `pick_and_place()`。

## Part 2：Scene + Data Engine

### `SceneConfig` / `ObjectPlacement`

```
robotsmith/scenes/config.py
```

| 类 | 说明 |
|---|---|
| `ObjectPlacement` (dataclass) | 单类资产的摆放规格：asset_query, count, position_range, rotation_range |
| `SceneConfig` (dataclass) | 仿真器无关的场景配置：物体列表、桌面参数、机器人、工作空间、重力、相机 |

### `ResolvedScene` / `PlacedObject`

```
robotsmith/scenes/backend.py
```

| 类 | 说明 |
|---|---|
| `PlacedObject` (dataclass) | 具体化的资产实例：Asset + 世界坐标 position/rotation/quaternion |
| `ResolvedScene` (dataclass) | 完全解析的场景：config + placed_objects + table_asset + plane_asset |

`SceneConfig` → `SceneBackend.resolve()` → `ResolvedScene` → 加载到仿真器。

### `SceneBackend` (ABC)

```
robotsmith/scenes/backend.py
```

| 实现 | 说明 | 状态 |
|------|------|:---:|
| `ProgrammaticSceneBackend` | 碰撞感知随机摆放（stable pose + FCL collision） | ✅ |
| `SceneSmithBackend` | LLM 驱动场景生成（预留） | 📋 |

### `GenesisSceneHandle`

```
robotsmith/scenes/genesis_loader.py
```

`ResolvedScene` 加载到 Genesis 后返回的句柄，持有 franka、objects、cameras、table 等 Genesis 实体引用。

### `ValidationResult`

```
robotsmith/validate/pybullet_check.py
```

PyBullet URDF 加载 + 稳定性检查的结果。用于 Part 1 → Part 2 之间的质量关口。

## DataBackend：可插拔数据生产

```
robotsmith/tasks/data_backend.py  (planned)
```

| 类 | 说明 | 状态 |
|---|---|:---:|
| `DataBackend` (ABC) | 数据生产接口：`collect(task, scene, n_episodes) → LeRobotDataset` | 📋 接口 |
| `IKScriptedBackend` | 开环 IK waypoint 回放（当前 `collect_data.py` 逻辑） | ✅ |
| `DARTBackend` | IK + noise 注入 + IK 重求解 | 📋 |
| `DAggerBackend` | Policy rollout + IK expert relabel | 📋 |
| `OnlineRLBackend` | Policy exploration + reward | 📋 |

所有后端共享同一 `TaskSpec` + `GraspPlanner` + `MotionExecutor` + `SceneConfig`，产出统一 LeRobot 数据集。
后端选择对下游 VLA 训练透明 — 不同后端数据可自由混合。

## TaskSpec：全局枢纽

### 当前定义

```
robotsmith/tasks/task_spec.py    — TaskSpec dataclass
robotsmith/tasks/predicates.py   — PREDICATE_REGISTRY + evaluate_predicate()
robotsmith/tasks/presets.py      — TASK_PRESETS (pick_cube, mug_in_bowl, stack_blocks)
```

```python
@dataclass
class TaskSpec:
    name: str
    instruction: str
    scene: str
    contact_objects: list[str]
    success_fn: str                    # PREDICATE_REGISTRY key（非 Callable）
    success_params: dict               # 纯 dict，JSON 可序列化
    motion_type: str = "pick"          # "pick" | "pick_and_place"
    episode_length: int = 200
    dart_sigma: float = 0.0
    grasp_planner: str = "template"    # GraspPlanner 后端名
```

**状态**：✅ Phase 1 + Phase 3.0 架构重构完成。`motion_type` 取代了旧 `ik_strategy` 字段。

### 社区参考

**RoboLab**（NVIDIA, 120 tasks）— 最接近 RoboSmith 的模型：

```python
@dataclass
class Task:
    scene: InteractiveSceneCfg          # USD 资产
    instruction: str | dict[str, str]   # default/vague/specific 三种粒度
    terminations: TerminationCfg        # composable conditionals
    subtasks: list[Subtask]             # milestone tracking
    contact_object_list: list[str]
    episode_length_s: int
    attributes: list[str]               # 任务标签
```

关键设计：成功判定不用 lambda（`func=named_fn, params=dict`）、`@atomic/@composite` 装饰器、Subtask 内嵌 Task、instruction 多粒度。

**ManiSkill3**（UCSD）— OOP 继承模型，不可序列化。启示：**保持 dataclass 声明式**。

**vla-eval**（Allen AI）— 不定义 task，只定义协议。启示：**可插拔后端只需统一 eval result schema**。

### 缺陷分析与演进路线

| # | 缺陷 | 社区参考 |
|---|------|---------|
| 1 | ~~`success_fn: Callable` 不可序列化~~ | ✅ 已改为 `str + params dict` |
| 2 | milestone 与 TaskSpec 分离 | RoboLab: `Task.subtasks` |
| 3 | ~~`ik_strategy` 混入共享结构~~ | ✅ 已重构：`motion_type` + `grasp_planner` |
| 4 | `scene: str` 太弱 | 升级为 `SceneSpec` 结构体 |
| 5 | 缺少 obs/action spec | vla-eval: 由 benchmark 自描述 |

### 演进设计（渐进式）

**原则**：不一次性膨胀。每个缺陷在真正触碰时才修复。

**Phase 1（现在 → Part 2 S1）**：当前 TaskSpec 已采用 `success_fn: str + success_params: dict`。

**Phase 2（vla-eval 集成）**：

```python
@dataclass
class TaskSpec:
    name: str
    instruction: str | dict[str, str]      # 支持多粒度
    tags: list[str] = field(...)           # 任务标签
    scene: str | SceneSpec = ""
    contact_objects: list[str] = field(...)
    success_fn: str = ""
    success_params: dict = field(...)
    milestones: list[Milestone] = field(...)  # subtask tracking
    motion_type: str = "pick"
    grasp_planner: str = "template"
    dart_sigma: float = 0.0
    episode_length: int = 200              # eval 专用
    num_eval_episodes: int = 50

@dataclass
class Milestone:
    name: str
    predicate_fn: str
    predicate_params: dict = field(...)
    order: int = 0
```

**Phase 3（接入 vla-eval bridge）**：加入 `ObsSpec` / `ActionSpec`，升级 `SceneSpec`。

### 对比 RoboLab 的设计选择

| 设计点 | RoboLab | RoboSmith | 理由 |
|--------|---------|-----------|------|
| Task 定义 | 每个任务一个类 | TaskSpec 实例 | 任务量小，不需要 120 个类文件 |
| 成功判定 | `TerminationTermCfg(func=, params=)` | `success_fn: str + success_params: dict` | 等效，不依赖 Isaac Lab |
| 谓词组合 | `@atomic` / `@composite` 装饰器 | `PREDICATE_REGISTRY` + 组合函数 | 更轻量 |
| Subtask | `Subtask` dataclass | `Milestone` dataclass（更简） | Phase 2 按需升级 |
| 场景 | USD 文件路径 | SceneConfig preset 或 SceneSpec | URDF，非 USD |
| Instruction | `str \| dict[str, str]` | Phase 1 str → Phase 2 `str \| dict` | 先跑通再丰富 |

## Eval：vla-eval Benchmark Plugin

RoboSmith 不自建 eval engine。评估通过 [vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness) 完成。

RoboSmith 的唯一贡献是 `RoboSmithBenchmark` — 实现 vla-eval `Benchmark` ABC，将 Genesis scene 作为 benchmark plugin 接入：

```python
# robotsmith/eval/benchmark.py
class RoboSmithBenchmark(Benchmark):
    # 7 abstract methods: get_tasks, start_episode, apply_action,
    # get_observation, is_done, get_time, get_result
```

- Action: 7D EE delta → IK → joint control
- Observation: 8D EE state + overhead + wrist (eye-in-hand) images
- 10+ VLA models 自动可用（Pi0, StarVLA, OpenVLA, GR00T...），RoboSmith 不做 model serving
- MI300X 验证通过（smoke test）

## 可视化层

### `AssetBrowser` / `SceneViewer`

```
robotsmith/viz/asset_browser.py   # viser Web UI: 资产网格 + 元数据侧栏
robotsmith/viz/scene_viewer.py    # viser Web UI: ResolvedScene 3D 预览 + 机器人 URDF
```

> 调研内容、参考链接、实现笔记已移至 [study.md](study.md)。
