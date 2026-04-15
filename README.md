
[EN](./README-en.md) | [CN](./README.md)

# RoboSmith

面向机器人操作研究的仿真就绪（sim-ready）数字资产库，搜索不到时自动调用 3D 生成模型补全。

## 工作流程

```
用户查询: "红色杯子"  (文本)
       │
       ▼
 AssetLibrary.search()
       │
   ┌───┴───┐
  命中    未命中
   │         │
   ▼         ▼
 返回      T2I → 参考图 → TRELLIS.2-4B (默认) → URDF → 入库
 URDF      SDXL-Turbo        image → PBR GLB (1K 默认)     │
   │         ┌────────────────────────────────────────────┘
   ▼         ▼
 仿真就绪资产 (URDF + 碰撞凸包 + PBR 纹理 + 元数据)
```

> 当前所有 SOTA 3D 生成模型均为 image-to-3D，文本查询未命中时先经 T2I (SDXL-Turbo) 生成参考图。

## 快速上手

```bash
pip install -e .

# 导入 Objaverse 高质量资产（10 品类 24 变体，按需下载 ~50 MB）
pip install objaverse
python scripts/import_objaverse.py

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
python scripts/browse_assets.py            # 生成 gallery.html 并自动打开
python scripts/browse_assets.py --no-open  # 仅生成
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

## 项目目标

教学/PoC 项目，分阶段验证 **Synthetic Data → VLA Post-Training → Sim Evaluation** 闭环。

| 阶段 | 目标 | 核心新增维度 | VLA 模型 | 状态 |
|:---:|------|------------|---------|:----:|
| **1** | 单物体 + 位姿泛化 (unseen 80%+) | 闭环 DART 数据 + 训练调优 | SmolVLA (450M) | 🔄 **← re-baseline** |
| 2 | 多物体泛化 | gen2sim 物品变体 | SmolVLA (450M) | 📋 |
| 3 | 中程多步任务 (stacking) | 多步推理 | [StarVLA](https://github.com/starVLA/starVLA) (Qwen3-VL 4B) | 📋 |
| 4 | 长程任务 | 长序列规划 | StarVLA (Qwen3-VL 4B) | 📋 |

**当前：Stage 1 re-baseline**。lerobot 项目中 V5 达到 60%（unseen），需要通过闭环 DART 数据 + 训练调优达到 **80%+** 后再进入 Stage 2。

### Sim-to-Policy 管线

```bash
# 数据采集（Genesis + Franka IK）
python pipeline/collect_data.py --n-episodes 100 --save          # 开环 baseline
python pipeline/collect_data_dart.py --n-episodes 100 --save     # 闭环 DART

# SmolVLA 后训练
python pipeline/train_smolvla.py --dataset-id local/franka-pick-100ep --n-steps 2000

# 闭环评估
python pipeline/eval_policy.py --policy-type smolvla --checkpoint outputs/smolvla/final
```

## 更多文档

- [docs/background.md](docs/background.md) — 技术背景（水密网格、URDF、凸包近似等）
