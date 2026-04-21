# RoboSmith — Design Document

> 具身数据基础设施：资产生成 + 数据生产。
> **定位：Data Infra，不是 VLA 训练框架。** 你定义任务，我们生成资产、生产数据。你来训练 VLA。
>
> Part 1: Sim-ready 3D gen (done)
> Part 2: Data Engine — TaskSpec + 可插拔数据生产后端 (IK ✅ / DART / DAgger / Online RL) + vla-eval benchmark plugin
>
> 相关文档：
> - [README.md](../README.md) — 项目概览与当前状态
> - [study.md](study.md) — 技术调研、前沿分析、实现笔记、参考链接
> - [part2.md](part2.md) — Part 2 Data Engine + Eval 总结
> - [background.md](background.md) — 技术背景（水密网格、URDF、凸包近似等）

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
Part 2 ── Data Engine + Eval ✅ (IK)
│
│   TaskSpec → SceneConfig → Genesis
│
│   Data Production Backends:
│     ├─ IK scripted (✅ default)
│     ├─ DART (IK + noise, planned)
│     ├─ DAgger (policy + IK, planned)
│     └─ Online RL (planned)
│   → LeRobot v3.0 数据集
│
│   Eval: vla-eval benchmark plugin (RoboSmithBenchmark)
│
────────────────────────────────────────────────────────────────────
│   DataBackend ABC                          │
│     .collect(task, scene, n_episodes)     │
│     → 统一 LeRobot 输出                    │
│   后端可插拔，下游无需感知数据生产方式         │
```

**验证状态**：

| 引擎 | 目标 | 状态 |
|------|------|:---:|
| **Data Engine — IK** | TaskSpec + IK solver: pick / place / stack → LeRobot 数据集 | ✅ |
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
| Task 定义系统 | ✅ | `robotsmith/tasks/`: TaskSpec + PREDICATE_REGISTRY (`object_above`, `object_in_container`, `stacked`) + IK_STRATEGIES |
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
    ik_strategy: str                   # IK waypoint 逻辑 ("pick", "pick_and_place", "stack")
    episode_length: int = 200          # 最大 step 数
    dart_sigma: float = 0.0            # DART 噪声 (0 = 纯 IK)
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
    ik_strategy="pick",
)

# pick-and-place: 把 mug 放进 bowl
TaskSpec(
    name="mug_in_bowl",
    instruction="Place the mug in the bowl",
    scene="tabletop_simple",
    contact_objects=["mug", "bowl", "table"],
    success_fn="object_in_container",
    success_params={"object": "mug", "container": "bowl"},
    ik_strategy="pick_and_place",
)

# stacking: 3 blocks 堆叠
TaskSpec(
    name="stack_blocks",
    instruction="Stack the red, green, and blue blocks",
    scene="tabletop_simple",
    contact_objects=["block_red", "block_green", "block_blue", "table"],
    success_fn="stacked",
    success_params={"objects": ["block_red", "block_green", "block_blue"]},
    ik_strategy="stack",
)
```

### 多任务 IK Solver

每种任务类型对应一套 waypoint 生成逻辑：

| 任务类型 | Waypoint 序列 | 状态 |
|---------|-------------|:---:|
| `pick` | reach → pre-grasp → grasp → lift (135 frames) | ✅ 20ep 100% |
| `pick_and_place` | pick + transport → pre-place → place → release → retreat (225 frames) | ✅ 20ep 100% |
| `stack` | N 轮 pick_and_place，每轮 place_z 递增 (N×225 frames) | ✅ 20ep 90% (N=3) |
| `push` | approach → contact → slide → release | 📋 |

```python
def generate_trajectory(task_spec, robot, scene_state):
    strategy = IK_STRATEGIES[task_spec.ik_strategy]
    waypoints = strategy.plan(robot, scene_state)
    if task_spec.dart_sigma > 0:
        waypoints = apply_dart_noise(waypoints, sigma=task_spec.dart_sigma)
    return waypoints
```

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

开环 IK waypoint 回放，per-task `IKStrategy` 生成 waypoints。
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
    ├── ik_strategy → IKStrategy → waypoints
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

**Phase 1（当前）**：抓握三件套 — 验证 TaskSpec + IK 数采框架

| 任务 | ik_strategy | 轨迹 | 状态 |
|------|-----------|:---:|:---:|
| `pick_cube` | `pick` | 135 frames | ✅ 20ep 100% |
| `place_cube` | `pick_and_place` | 225 frames | ✅ 20ep 100% |
| `stack_blocks` | `stack` | 675 frames (3×225) | ✅ 20ep 90% |

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

---

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

# Next Step：Grasp Affordance Gap 与演进路线

## 核心矛盾

RoboSmith 当前存在一个结构性 gap：

```
Part 1 能生成 custom assets (碗/杯/瓶...)
Part 2 只能操作 known geometry (cube/block)
    ↓
真正需要 RoboSmith 数据的场景（out-of-distribution assets），
恰恰是 RoboSmith 当前无法自动处理的
```

具体分析：

| 物体类型 | Part 1 价值 | Part 2 能力 | 矛盾 |
|---------|:-----------:|:-----------:|------|
| 标准件 (cube/block) | 低 — 任何 sim 都有 | ✅ 已验证 | 不需要 RoboSmith |
| Custom assets (碗/杯/瓶) | **高 — RoboSmith 差异化** | ❌ 不知道怎么抓 | **需要但做不到** |

**根因**：Part 1 → Part 2 之间缺少一个 **Grasp Affordance Layer** — 给定一个 custom mesh，如何确定 grasp point、approach direction、finger width、pre/post grasp waypoints。

```
Part 1: Asset → URDF + mesh + collision
           ↓
   [MISSING] Grasp Affordance: 从哪里抓？用什么姿态？
           ↓
Part 2: IK Strategy → trajectory → data
```

当前 IK strategy 的 grasp 参数全部 hardcoded（`grasp_z=0.135`, `grasp_quat=[0,1,0,0]`, `finger_closed=0.01`），
只适用于桌面上的小方块。碗（直径 14cm，超过 Franka 8cm 最大开口）、瓶（需要侧面抓取）等都无法处理。

## 社区参考：ShapeGen

[ShapeGen](https://wangyr22.github.io/ShapeGen/)（清华, 2026）解决的正是这个问题 — "给定新形状，怎么知道该怎么操作"：

| 概念 | ShapeGen 做法 | 对 RoboSmith 的启发 |
|------|-------------|-------------------|
| Shape Library | 每个 category 有 template shape + spatial warping | RoboSmith 已有 per-category assets (Part 1) |
| Functional correspondence | 同类物体间的功能性对应点 (handle, rim, spout) | 需要新增：per-category grasp template |
| Minimal annotation | 人工在 template 上标注一次 keypoints，warping 传播到 variants | 一次性 per-category annotation，非 per-object |
| Data generation | 基于对应关系自动修正新形状上的 action trajectory | IK strategy 从 template 读取参数 |

ShapeGen 是 simulator-free（用真实数据 + 3D warping），RoboSmith 可以做 sim-based 简化版。

## 两条演进路线

### 路线 A：场景多样性 — 标准件 + 丰富场景（短期可行）

不解决 HOI gap，而是从另一个角度提供数据价值：

- 同样是 pick cube，但 table layout、distractor objects、lighting、camera angle 每次不同
- Custom assets 作为 **distractor**（不操作，只作为视觉干扰），而非 **target**
- 价值：VLA robustness to visual diversity

```
TaskSpec: pick_cube
Scene: tabletop_simple (mug + bowl + 3 blocks as distractors)
数据价值: 相同任务 × 不同视觉上下文 → VLA 鲁棒性
```

**优点**：立即可做，不改架构
**缺点**：故事弱 — "我们生成了漂亮的碗，但只是用来当背景"

### 路线 B：Per-Category Grasp Template（中期，ShapeGen-inspired）

在 Part 1 和 Part 2 之间引入 **Grasp Affordance Layer**：

```python
@dataclass
class GraspTemplate:
    """Per-category grasp affordance definition."""
    category: str                    # "bowl", "mug", "bottle"
    grasp_type: str                  # "top_down", "side", "rim", "handle"
    approach_dir: np.ndarray         # approach direction (EE frame)
    grasp_offset: np.ndarray         # grasp point relative to object center
    grasp_quat: np.ndarray           # EE orientation at grasp
    finger_width: float              # required finger opening
    pre_grasp_offset: np.ndarray     # hover position offset
    min_scale: float                 # asset must be scaled to fit gripper
    max_scale: float
```

每个 category 人工定义 **一次** grasp template：

| Category | Grasp Type | 说明 | 一次性 annotation |
|----------|-----------|------|:---:|
| bowl | top-down (scaled) | scale 0.5x → 7cm, 顶部抓取 | 1 template |
| mug | handle | 侧面抓 handle, finger width ~2cm | 1 template |
| bottle | side | 侧面抓瓶身, 需要倾斜 EE | 1 template |
| can | side | 类似 bottle | 复用 bottle template |
| fruit | top-down | 小球体, 直接顶部抓 | 1 template |
| plate | rim/edge | 抓盘沿, 需要精确 Z | 1 template |

IK strategy 从 GraspTemplate 读取参数，替代 hardcoded `TrajectoryParams`：

```
Asset (Part 1) → match category → GraspTemplate → TrajectoryParams → IK plan
```

**优点**：解决核心矛盾，Part 1 assets 真正可操作
**缺点**：需要 per-category 人工定义（但只做一次），scale 可能影响物理真实性

### 推荐：A → B 渐进

1. **短期 (A)**：用 custom assets 做 distractor，标准件做 target，先出一个 demo
2. **中期 (B)**：逐步加 grasp template，从 bowl (top-down, scaled) 开始，最简单
3. **长期**：接入 GraspNet/AnyGrasp 自动预测 grasp pose，去掉人工 annotation

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
          ┌────────────────────────┼────────────────┐
          │                        │                │
   ┌──────▼──────┐         ┌──────▼──────┐  ┌──────▼──────────┐
   │ Part 1      │         │ Part 2      │  │ Eval            │
   │ Asset Layer │         │ Data Engine │  │ (vla-eval plugin)│
   └─────────────┘         └─────────────┘  └─────────────────┘
   Asset                    DataBackend (ABC)  RoboSmithBenchmark
   AssetMetadata             ├ IKScripted ✅    (vla-eval Benchmark ABC)
   AssetLibrary              ├ DART 📋
   GenBackend                ├ DAgger 📋
                             └ OnlineRL 📋
                            SceneConfig
                            IKStrategy
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

所有后端共享同一 `TaskSpec` + `IKStrategy` + `SceneConfig`，产出统一 LeRobot 数据集。
后端选择对下游 VLA 训练透明 — 不同后端数据可自由混合。

## TaskSpec：全局枢纽

### 当前定义

```
robotsmith/tasks/task_spec.py    — TaskSpec dataclass
robotsmith/tasks/predicates.py   — PREDICATE_REGISTRY + evaluate_predicate()
robotsmith/tasks/ik_strategies.py — IK_STRATEGIES + PickStrategy
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
    ik_strategy: str                   # IK_STRATEGIES key
    episode_length: int = 200
    dart_sigma: float = 0.0
```

**状态**：✅ Phase 1 已实现。`collect_data.py` 通过 `--task pick_cube` 使用 TaskSpec dispatch。
有 5 个结构性缺陷需要在扩展前解决。

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
| 3 | `ik_strategy` 混入共享结构 | 分离为 `CollectionConfig` |
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
    ik_strategy: str = "pick"              # Part 2 专用
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
