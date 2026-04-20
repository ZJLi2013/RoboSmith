# RoboSmith

具身数据基础设施：从 3D 资产到机器人操作数据的端到端锻造管线。
**Gen2Sim → Task 定义 → IK 数采 → VLA 训练 → 标准化评估 → 闭环数据飞轮**。

> Part 2 **Data Engine**：Task 定义系统 + 多任务 IK solver + DART 噪声增强 → 数据采集。
> Part 3 **Eval Engine**：Server-Client policy 接口 + 并行评估 + subtask tracking → 评估诊断。
> 两者共享同一套 TaskSpec + composable predicates，数采与评估在统一体系下闭环（AMD ROCm + Genesis）。

## 工作流程

```
┌─────────────────────────────────────────────────────────────────────┐
│ Part 1: Sim-Ready 资产                                              │
│                                                                     │
│  文本查询 → AssetLibrary.search()                                    │
│               │                                                     │
│           命中 / 未命中 → T2I (SDXL-Turbo) → TRELLIS.2-4B → URDF    │
│               │                                                     │
│               ▼                                                     │
│  Sim-Ready 资产 (URDF + PBR GLB + 碰撞凸包 + stable_poses)          │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                ┌───────────────▼───────────────┐
                │  TaskSpec (统一任务定义层)       │
                │  composable predicates 成功判定 │
                │  一次定义 → 数采 + 评估同时生效   │
                └───────┬───────────────┬───────┘
                        │               │
┌───────────────────────▼───┐   ┌───────▼───────────────────────────┐
│ Part 2: Data Engine (数采) │   │ Part 3: Eval Engine (评估)        │
│                           │   │                                   │
│ SceneConfig → 碰撞感知摆放  │   │ Server-Client policy 接口         │
│ → Genesis 仿真             │   │ → 任意 VLA 接入                   │
│                           │   │                                   │
│ IK Solver (per-task)      │   │ Genesis 并行评估 (--num-envs N)   │
│ + DART 噪声 (--dart-sigma)│   │ + subtask progress tracking       │
│ → LeRobot 数据集           │   │ → 标准化评分 + 诊断报告            │
│ → VLA 训练                │   │                                   │
└───────────┬───────────────┘   └──────────────┬────────────────────┘
            │                                  │
            └──────────── 闭环 ────────────────┘
            评估诊断 → 针对性补采 → 重新训练 → 循环
```

<p align="center">
  <img src="images/scene_overview.png" width="600" alt="tabletop_simple scene — Franka + random objects">
  <br>
  <em>tabletop_simple 场景：Franka 机械臂 + 碰撞感知随机摆放 (Genesis, MI300X)</em>
</p>

## 快速上手

```bash
pip install -e .

# 导入 Objaverse 高质量资产（10 品类 24 变体，按需下载 ~50 MB）
pip install objaverse
python scripts/part1/import_objaverse.py

robotsmith list                    # 列出全部资产
robotsmith search "cup"            # 搜索
robotsmith scene tabletop_simple   # 解析场景预设

# 生成新资产（需 GPU，≥24 GB VRAM）
robotsmith generate "red ceramic mug" --image reference.png                     # 默认 TRELLIS.2, 1K PBR
robotsmith generate "red ceramic mug" --image reference.png --quality fast      # 512 PBR, 100K faces (RL 批量)
robotsmith generate "red ceramic mug" --image reference.png --quality high      # 4K PBR, 1M faces (展示/论文)
robotsmith generate "red ceramic mug" --image reference.png --backend hunyuan3d # Hunyuan3D 备选
robotsmith generate "red ceramic mug"   # 无图：自动 T2I → 3D

# 验证全部资产
pip install pybullet
robotsmith validate
```

## 3D 生成

**资产策略**：10 品类桌面操作物品（24 变体，~60 MB），几何拓扑多样性最大化。
主力来源 [Objaverse](https://objaverse.allenai.org/) 按需导入，搜索未命中时自动调用 TRELLIS.2-4B @512 生成。可插拔后端架构（`GenBackend` ABC）。

**默认品类**（按几何覆盖选品，非按生活用品分类）：

| 品类 | 几何 | 变体 | 来源 |
|------|------|:---:|------|
| 马克杯 (mug) | 圆柱+把手 | 3 | Objaverse |
| 碗 (bowl) | 凹半球 | 2 | Objaverse |
| 积木 (block) | 长方体 | 3 | Primitive |
| 易拉罐 (can) | 短圆柱 | 2 | Objaverse |
| 瓶子 (bottle) | 高圆柱+窄颈 | 2 | Objaverse |
| 水果玩具 (fruit) | 球/椭球 | 3 | Objaverse |
| 动物玩具 (figurine) | 不规则凸包 | 3 | Objaverse |
| 盘子 (plate) | 扁圆盘 | 2 | Objaverse |
| L 形块 (L-block) | 非凸体 | 2 | Primitive |
| 小盒子 (box) | 扁长方体 | 2 | Primitive |

| 后端 | 模型 | PBR | VRAM | ROCm | 状态 |
|------|------|:---:|------|------|------|
| **`trellis2`** | [TRELLIS.2-4B](https://github.com/ZJLi2013/TRELLIS.2/tree/rocm) | ✅ | ≥24 GB | ✅ MI300X | **默认** — 1K PBR (可选 512/4K), 无底座 artifact, 无 bpy 依赖 |
| `hunyuan3d` | [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | ✅ | ≥29 GB | ✅ MI300X | 备选 |

默认管线（TRELLIS.2-4B）：

```
参考图 → TRELLIS.2-4B (4B, ~275s, ≥24GB) → O-Voxel → remesh + PBR bake → GLB
```

**纹理分辨率预设**（`--quality`）：

| 预设 | 纹理 | 面片 | GLB | 场景 |
|------|:----:|:----:|:---:|------|
| `fast` | 512 | 100K | ~2 MB | RL 批量训练、快速迭代 |
| `balanced` (默认) | **1024** | **200K** | **~8 MB** | sim 实验、demo、抓取测试 |
| `high` | 4096 | 1M | ~38 MB | 论文 figure、展示 |

> Collision mesh 由 trimesh 凸包生成，与纹理分辨率无关。
> 1K PBR 在仿真视口（640×480 ~ 1024×768）中肉眼无差异，GLB 从 38 MB→~8 MB，仿真加载更快。
> 权重：[microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B)。ROCm fork: [ZJLi2013/TRELLIS.2@rocm](https://github.com/ZJLi2013/TRELLIS.2/tree/rocm)。

### 资产目录结构

```
assets/
├── objects/                              # 默认资产（metadata.json git tracked，大文件 git ignored）
│   ├── mug_01/
│   │   ├── model.urdf                    # 引用 visual + collision mesh
│   │   ├── visual.glb                    # PBR mesh (git ignored)
│   │   ├── collision.obj                 # 碰撞凸包 (git ignored)
│   │   ├── metadata.json                 # 物理属性 + tags + stable_poses (git tracked)
│   │   └── provenance.json              # Objaverse UID / 来源 (git tracked)
│   ├── block_red/                        # Primitive: 只有 URDF + metadata (全 git tracked)
│   ├── table_simple/
│   └── plane/
├── generated/                            # 管线生成资产 (全 git ignored)
│   └── red_ceramic_mug_trellis2/         # TRELLIS.2 4K 参考 (保留)
└── catalog.json                          # 轻量索引 (git tracked)
```

## 场景预设

| 场景 | 描述 |
|------|------|
| `tabletop_simple` | 桌子 + 杯子 + 碗 + 3 个积木 |

## 可视化

### 3D 场景预览（viser）

```bash
pip install -e ".[viz]"
robotsmith view tabletop_simple            # 浏览器打开 http://localhost:8080
robotsmith view --asset mug_red            # 预览单个资产

# 远端 GPU 节点
ssh -L 8080:localhost:8080 user@gpu-node
robotsmith view tabletop_simple
```

### 资产画廊（零依赖 HTML）

```bash
python scripts/part1/browse_assets.py            # 生成 gallery.html 并自动打开
python scripts/part1/browse_assets.py --no-open  # 仅生成
```

- Built-in 显示 SVG 几何预览，Generated 显示 T2I 参考图
- 支持 All / Built-in / Generated 筛选
- 自包含 HTML，可离线查看

## Sim-ready 成熟度

```
层级   要求                          状态
────   ────                          ──────
L0     能加载到仿真器                 ✅
L1     碰撞生效                       ⚠️ 凸包近似（凹面丢失）
L2     物理属性精确                   ❌ 待实现
L3     视觉逼真（PBR）               ✅ TRELLIS.2 PBR (1K 默认, 可选 512/4K)
```



## 依赖

**核心（无需 GPU）：** `trimesh >= 4.0`, `numpy >= 1.24`

**可选：**
- `viser >= 1.0` — 3D 可视化 (`pip install -e ".[viz]"`)
- `pybullet >= 3.2` — 物理验证
- `torch >= 2.0` — 3D 生成（ROCm / CUDA）
- [TRELLIS.2](https://github.com/ZJLi2013/TRELLIS.2/tree/rocm) — **默认 3D 后端** (ROCm fork, 1K PBR 默认)
- [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) — 备选 3D 后端
- `genesis-world >= 0.2` — Genesis 仿真器（Part 2）

## 多任务数采管线

Part 2 的核心是在 sim 中为不同操作任务生成训练数据。每个任务通过 `TaskSpec` 定义，
IK solver 根据任务类型生成轨迹，DART 噪声增强作为可选参数提升数据鲁棒性。

**Task 定义**（借鉴 [RoboLab](https://github.com/NVLabs/RoboLab) composable predicates）：

| 任务 | 场景 | 成功判定 | IK 逻辑 |
|------|------|---------|---------|
| pick | cube on table | `object_above(cube, table, z=0.05)` | reach → grasp → lift |
| pick_and_place | mug + bowl | `object_in_container(mug, bowl)` | reach → grasp → lift → move → place |
| stacking | 3 blocks | `stacked(blocks, order)` | 多轮 pick-and-place |

```bash
# 数据采集（IK solver + 可选 DART 噪声增强）
python scripts/part2/collect_data.py --task pick --n-episodes 100 --save
python scripts/part2/collect_data.py --task pick --n-episodes 100 --dart-sigma 0.005 --save
python scripts/part2/collect_data.py --task pick_and_place --n-episodes 100 --dart-sigma 0.005 --save

# VLA 验证
python scripts/part2/train_smolvla.py --dataset-id local/franka-pick-100ep --n-steps 10000
python scripts/part2/eval_policy.py --policy-type smolvla --checkpoint outputs/smolvla/final
```

## 项目路线

| 阶段 | 任务 | 核心工作 | 验证模型 | 状态 |
|:---:|------|---------|---------|:----:|
| **1** | pick-cube (unseen 80%+) | TaskSpec + IK 数采 + DART 增强，单任务跑通 | SmolVLA (450M) | 🔄 |
| 2 | pick (多物体泛化) | Task 定义系统 + gen2sim 物品变体 + **Eval Engine v1** | SmolVLA (450M) | 📋 |
| 3 | pick-and-place, stacking | 多任务 IK solver + 并行评估 + subtask tracking | SmolVLA / StarVLA | 📋 |
| 4 | 长程任务 + 闭环 | sim-DAgger / 自动数据补采飞轮 | StarVLA (Qwen3-VL 4B) | 📋 |

## 项目结构

```
robotsmith/                      # Python 库（pip install -e .）
├── assets/                      #   Part 1: 资产管理 (schema, library, search, builtins)
├── gen/                         #   Part 1: 3D 生成 (TRELLIS.2 / Hunyuan3D 后端, mesh_to_urdf)
├── scenes/                      #   Part 2: 场景构建 (SceneConfig, genesis_loader, presets)
├── tasks/                       #   Part 2: 任务定义 (TaskSpec, predicates, IK strategies)
├── validate/                    #   物理验证 (PyBullet)
├── viz/                         #   可视化 (Viser asset browser + scene viewer)
└── cli.py                       #   CLI 入口 (robotsmith list/search/generate/view/...)

scripts/
├── part1/                       # Part 1: 资产生成 & 管理
│   ├── import_objaverse.py      #   导入 Objaverse 高质量资产
│   ├── compute_stable_poses.py  #   计算稳定放置姿态
│   ├── browse_assets.py         #   生成 HTML 资产画廊
│   └── render_mesh_local.py     #   本地渲染 mesh → PNG
├── part2/                       # Part 2: 数采 + VLA 训练验证
│   ├── collect_data.py          #   IK 数采 (+ --dart-sigma DART 增强)
│   ├── snapshot_scene.py        #   场景布局截图验证
│   ├── train_smolvla.py         #   SmolVLA fine-tune (数据质量验证)
│   └── eval_policy.py           #   闭环策略评估
└── tools/                       # 通用工具
    ├── sync_assets.py           #   远端 GPU 节点资产同步
    └── patch_genesis_rocm.py    #   Genesis ROCm 兼容补丁

assets/                          # 资产存储 (objects/ + generated/ + catalog.json)
docs/                            # 文档
tests/                           # 测试
```

## 更多文档

- [docs/design.md](docs/design.md) — 架构设计（Part 1/2/3 + 核心抽象）
- [docs/study.md](docs/study.md) — 调研笔记（RoboLab、3D 生成、场景工具等）
- [docs/stage-1.md](docs/stage-1.md) — Stage 1 实验计划
- [docs/background.md](docs/background.md) — 技术背景（水密网格、URDF、凸包近似等）
- [docs/experiments.md](docs/experiments.md) — 实验记录汇总
