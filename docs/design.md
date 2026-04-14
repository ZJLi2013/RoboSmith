# RoboSmith — Design Document

> 从 3D 资产到机器人行为数据的端到端锻造管线。
> Sim-ready 3D gen → 仿真环境 → 轨迹采集 → 策略训练验证，一条可复现的闭环。
>
> 当前状态请参考 [README.md](../README.md) · [experiments.md](../experiments.md)
> 背景知识（水密网格、URDF、凸包近似等）请参考 [background.md](background.md)

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
────────────────────────────────────────────────
│
Part 2 ── 场景多样性 + Sim 管线 ◀ 当前核心
│
│   SceneConfig → ProgrammaticSceneBackend (碰撞感知摆放 + stable pose)
│   → ResolvedScene → genesis_loader → Genesis 仿真
│   轨迹采集 (IK scripted) → LeRobot 数据集
│   策略训练 (SmolVLA / ACT) → 闭环评估
│
────────────────────────────────────────────────
│
扩展 ── 更强 VLA (π0.5) / World Model 替代采集
```

**MVP 验证标准**：pipeline 生成的场景 + 采集轨迹 → 训出 policy → unseen 配置成功率 > 50%。

## 当前状态

| 模块 | 状态 | 说明 |
|------|:----:|------|
| 资产库 (`AssetLibrary`) | ✅ | 26 资产 (17 Objaverse + 7 primitive + 2 scene)，`objects/` + `generated/` 双目录扫描 |
| Stable poses | ✅ | 全部 26 资产预计算（trimesh, Linux MI300X），存入 `metadata.json` |
| 资产持久化 & 同步 | ✅ | `catalog.json` 索引，`.gitignore` 排除大文件，只 track json/urdf |
| 碰撞感知摆放 | ✅ | `ProgrammaticSceneBackend` + workspace_xy + stable pose 采样 + AABB/FCL 碰撞检测 |
| Genesis scene loader | ✅ | `genesis_loader.py`: ResolvedScene → Genesis entities (Plane + Table + Objects + Franka) |
| 场景预设 | ✅ | `tabletop_simple` (mug + bowl + 3 blocks, collision-aware) |
| 3D 生成后端 | ✅ | **TRELLIS.2-4B (默认, 1K PBR balanced, 可选 512/4K)** + Hunyuan3D-2.1 (备选)，MI308X 验证 |
| mesh → URDF 转换 | ✅ | trimesh 凸包 + 物理属性估算 |
| T2I 桥接 (text→image) | ✅ | SDXL-Turbo 默认，3D 友好 prompt 优化 (§1.4) |
| 静态场景可视化 | ✅ | viser SceneViewer |
| `collect_data.py` | ✅ | 支持 `--scene tabletop_simple` 模式（ResolvedScene）+ 原始 hardcoded cube 模式 |
| 轨迹采集 (Genesis E2E) | 🔜 | 待远端 GPU 验证 `--scene` 模式端到端 |
| 策略训练 | 📋 | SmolVLA/ACT baseline 已验证（单任务 cube），多物体场景待验证 |
| 环境外观随机化 | 📋 | P1: 颜色/纹理/光照随机化 (§2.8 Step 4) |

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
  (URDF)      │ T2I 桥接      │  text → 参考图 (SDXL-Turbo, 3D 友好 prompt)，有图则跳过
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │ TRELLIS.2-4B │  image → O-Voxel → remesh → PBR GLB (1K 默认, 可选 512/4K)
              └──────┬───────┘
              (备选: Hunyuan3D-2.1 Shape + PBR Paint)
                     ▼
              ┌──────────────┐
              │ mesh_to_urdf │  GLB visual + OBJ 凸包碰撞 + 物理属性 + 尺度标准化
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │ 自动入库       │  tag + metadata.json + catalog.json
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │ 资产同步       │  remote GPU → local (scripts/sync_assets.py)
              └──────────────┘
```

### 核心接口

```python
from robotsmith.assets import AssetLibrary

lib = AssetLibrary("./assets")

# 检索（objects/ 和 generated/ 都会被扫描）
cup = lib.search("红色水杯")

# 生成兜底（有参考图：直接 I2-3D；无图：自动 T2I → I2-3D）
if cup is None:
    cup = lib.generate("红色水杯", image_path="mug.png")

cup.urdf_path     # assets/generated/red_ceramic_mug_20260408_100000/model.urdf
cup.metadata      # {"mass": 0.25, "friction": 0.6, "source": "generated", ...}

# 列出所有 pipeline 生成的资产
lib.list_generated()

# 持久化索引（add 时自动调用，也可手动触发）
lib.save_catalog()  # -> assets/catalog.json
```

### 资产存储与同步

```
assets/
├── objects/                          # 内置/人工策划资产（git tracked）
│   ├── mug_red/
│   │   ├── model.urdf
│   │   └── metadata.json
│   └── ...
├── generated/                        # 管线生成资产（git ignored, 仅本地）
│   ├── red_ceramic_mug_20260408_100000/
│   │   ├── model.urdf
│   │   ├── visual.obj
│   │   ├── collision.obj
│   │   ├── metadata.json
│   │   └── reference.png             # T2I 生成的参考图
│   └── ...
└── catalog.json                      # 轻量索引 (git tracked)
```

| 问题 | 方案 |
|------|------|
| 重启后生成资产丢失 | `_load_catalog()` 同时扫描 `objects/` 和 `generated/` |
| 大模型文件不入 git | `.gitignore` 排除 `assets/generated/*`，仅保留 `catalog.json` |
| 远端节点随时切换 | `scripts/sync_assets.py` 从任意 GPU 节点 scp 回本地 |
| 索引持久化 | `library.add()` 自动写 `catalog.json`；手动 `save_catalog()` |

```bash
# 从远端 GPU 节点同步生成资产到本地
python scripts/sync_assets.py banff-sc-cs41-29.amd.com \
    --remote-dir /data/robotsmith/assets/generated

# Docker 容器内的资产
python scripts/sync_assets.py banff-sc-cs41-29.amd.com \
    --docker rocm_dev --remote-dir /data/robotsmith/assets/generated
```

## 1.2 内置资产

### 资产来源策略

| 来源 | 用途 | 质量 | 底座 artifact |
|------|------|:---:|:---:|
| **Objaverse 策划** | 高频类目（10 种桌面物品） | ★★★★☆ 人工建模/3D 扫描 | **无** |
| **TRELLIS.2 生成** ← 默认兜底 | 搜索未命中的长尾类目 | ★★★★☆ PBR (1K 默认) | **无** |
| Hunyuan3D 生成 ← 备选 | 备选后端 | ★★★☆☆ | 有（需 trim） |
| 程序化原语 ← 已废弃 | 仅供 table / ground plane | ★☆☆☆☆ 几何体 | 无 |

> TRELLIS.2 生成质量接近策划资产水准（PBR 纹理、无底座 artifact），作为"搜索未命中"的默认兜底。
> 默认 1K PBR（sim 实验最佳平衡），可通过 `--quality` 或 `--texture-size` 调整。

### 物品 — 10 品类桌面操作物品

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

**几何覆盖**：

```
球体/椭球 ─── fruit          对称，滚动
圆柱体   ─── can, bottle     高/矮，光滑曲面
长方体   ─── block, box      尺度变化（3cm → 15cm）
凹形     ─── bowl, mug       内凹面，边缘/把手
扁平     ─── plate           极端长宽比，pinch
不规则   ─── figurine        无规律，考验泛化
非凸     ─── L-block         重心偏移，非对称
```

**与旧选品对比**（fork/spoon/knife/pan → fruit/figurine/L-block/box）：

| 替换 | 原因 |
|------|------|
| ~~fork/spoon/knife~~ | 极细长，Franka 平行夹爪成功率低，不适合 pick 任务初期 |
| ~~pan~~ | 28cm 超出常规抓取范围，需双手/特殊策略 |
| + fruit | 近球体几何 + 滚动挑战，Objaverse 大量现成 |
| + figurine | 不规则凸包，训练泛化核心品类（GraspVLA 验证有效） |
| + L-block | 非凸形状，重心偏移，简单 primitive 组合即可 |
| + box | 扁长方体（与 block 不同长宽比），primitive 即可 |

**排除品类**（初期不引入）：

| 品类 | 原因 |
|------|------|
| 布料/毛绒 | 柔体，Genesis rigid body 不适用 |
| 刀具/剪刀 | 铰接 + 安全，不适合初期 |
| 纯球体 | fruit 已覆盖，纯球无法稳定放置 |
| 透明物体 (glass) | 渲染/感知 sim2real gap 过大 |

### 资产尺寸预算

> **目标：全部资产 < 500 MB**（初期尽量控制在 ~100 MB 内）。

**10 品类 × 24 变体，按来源估算**：

| 来源 | 数量 | 单个大小 | 小计 |
|------|:---:|:---:|:---:|
| Objaverse 导入 (mug×3, bowl×2, can×2, bottle×2, fruit×3, figurine×3, plate×2) | 17 | ~3 MB | ~51 MB |
| TRELLIS.2 @ 512px 补缺（Objaverse 找不到合适变体时） | ~0-3 | ~2 MB | ~6 MB |
| Primitive URDF (block×3, L-block×2, box×2) | 7 | ~1 KB | ~0 MB |
| **合计** | **24** | | **~60 MB** |

**分辨率对总尺寸的影响**：

| 纹理分辨率 | 单个 (TRELLIS.2) | 24 asset 全量估算 | 50 asset 扩展 |
|:---:|:---:|:---:|:---:|
| 4K ❌ | ~130 MB | 超标 | 超标 |
| **1K** | ~8 MB | ~200 MB | ~400 MB |
| **512 ← 推荐初期** | ~2 MB | **~60 MB** | ~100 MB |

> **决策**：TRELLIS.2 生成统一使用 **512px（fast 预设）**。
> Objaverse 原始 GLB 天然 1-5 MB/个，无需额外降质。
> 现有 4K 马克杯（131 MB）需用 512 重新生成。
> 512px 在仿真器视口（通常 640×480）下视觉无明显差异，collision mesh 不受影响。

**变体数量原则**："同品类内变体 > 跨品类数量"（见 §2.7 Insight 5），但初期每品类 **2-3 个即可**——
先验证 pipeline 跑通，再按需扩展。扩展到 50 个仍在 200 MB 以内。

### 资产来源策略（修订）

| 来源 | 适用品类 | 视觉质量 | 单个大小 | 说明 |
|------|---------|:---:|:---:|------|
| **Objaverse 按需导入** ← 主力 | #1-8 (17 个变体) | ★★★★ | 1-5 MB | `objaverse` 包按 UID 下载，不需要完整数据集 |
| **TRELLIS.2 @ 512** ← 补缺 | Objaverse 找不到好的变体时 | ★★★★ | ~2 MB | 512px fast 预设，特定颜色/形态长尾变体 |
| **Primitive URDF** ← 仅限几何体 | #3 block, #9 L-block, #10 box | ★★ | ~1 KB | 几何形状本身就是方块的品类 |

> **Primitive URDF 不适用于视觉策略训练**：VLA 依赖视觉 encoder，纯色几何体导致 policy 学到 color/shape shortcut 而非真正的物体理解。
> Primitive 仅保留给"物体外观就是纯色方块"的品类（L-block, box），以及 table/plane 等场景基础设施。

**Objaverse 按需导入流程**（不需要下载完整数据集）：

```
pip install objaverse     # Python 包，按需下载
    │
    ▼
objaverse.load_annotations()     # 首次下载注释索引 (~200MB, 缓存到 ~/.objaverse/)
objaverse.load_lvis_annotations() # LVIS 类别标注
    │
    ▼
find_candidates("mug", ...)      # LVIS + keyword 检索 → 候选 UID 列表
_score_candidate(ann)            # 面数/评分/许可证排序
    │
    ▼
objaverse.load_objects(uids=[best_uid])   # 只下载选中的 GLB (~几 MB/个)
    │
    ▼
trimesh.load → cleanup_mesh → mesh_to_urdf → metadata.json
```

每品类 2-3 个变体（24 个总计），Objaverse 下载量 ~50 MB。

**导入命令**：

```bash
pip install objaverse trimesh numpy
python scripts/import_objaverse.py              # 导入全部品类
python scripts/import_objaverse.py --category mug  # 导入单类
python scripts/import_objaverse.py --dry-run       # 仅预览候选
```

**Sim-ready 标准**: `model.urdf` + `visual.glb` + `visual.obj` + `collision.obj` + `metadata.json` + `provenance.json`

### 场景预设

固定臂（Franka）短/中程任务（3~8 step）场景泛化意义有限，物品泛化更关键。
保留单一场景预设，泛化重心放在 object variation（见 §2.7 数据多样性策略）。

| 场景 | 描述 | 包含物品 |
|------|------|---------|
| tabletop_simple | 1 桌 + 3-5 物品随机摆放 | 杯、碗、积木 |

```python
scene_config = {
    "table": {"size": [0.8, 0.6, 0.05], "height": 0.75},
    "objects": [
        {"type": "mug",   "count": 1, "pos_range": [[0.3, -0.2, 0], [0.6, 0.2, 0]]},
        {"type": "bowl",  "count": 1, "pos_range": [[0.2, -0.15, 0], [0.5, 0.15, 0]]},
        {"type": "block", "count": 3, "pos_range": [[0.25, -0.2, 0], [0.55, 0.2, 0]]},
    ],
    "robot": "franka_panda",
}
```

## 1.3 3D 生成管线

当资产库检索未命中时，调用 3D 生成模型自动补全。

### 当前默认：TRELLIS.2-4B

| 项目 | 值 |
|------|-----|
| 模型 | [TRELLIS.2-4B](https://github.com/microsoft/TRELLIS.2) (Microsoft, 4B params, CVPR'25 Spotlight) |
| ROCm fork | [ZJLi2013/TRELLIS.2@rocm](https://github.com/ZJLi2013/TRELLIS.2/tree/rocm) |
| 输入 | **单张参考图片** (image-to-3D) |
| 输出 | **GLB (PBR 纹理嵌入)**，visual.glb + collision.obj |
| 速度 | ~275s gen + ~85s export = **~423s 总计** (MI308X, 512³, 4K) |
| VRAM | **≥24 GB** |
| ROCm | **✅ 已验证** — MI308X, ROCm 6.4, flash_attn Triton, cumesh, flexgemm, nvdiffrast |
| 优势 | **无底座 artifact** · PBR 纹理 · mesh 质量高 · 无 bpy 依赖 · 原生 GLB 导出 |

#### 纹理分辨率预设（`--quality` / `--texture-size`）

默认 **balanced (1K)**，兼顾视觉质量与 sim 性能。

| 预设 | `--quality` | 纹理分辨率 | 面片数 | GLB 大小 (估) | 导出耗时 | 适用场景 |
|------|:-----------:|:---------:|:------:|:------------:|:--------:|---------|
| **fast** | `fast` | 512 px | 100K | ~2 MB | ~20s | 批量轨迹生成、RL 训练、快速迭代 |
| **balanced** | `balanced` (默认) | **1024 px** | **200K** | **~8 MB** | **~40s** | 日常 sim 实验、demo、抓取测试 |
| **high** | `high` | 4096 px | 1M | ~38 MB | ~85s | 展示/论文、视觉质量评估 |

> **为什么默认 1K 而非 4K？**
>
> - **Collision mesh 不受影响**：碰撞体由 trimesh 凸包生成，与纹理分辨率无关。1K/4K 的 collision.obj 完全一致。
> - **Visual mesh 足够 sim 使用**：1K PBR 纹理在仿真器视口（通常 640×480 或 1024×768）中肉眼无差异。
>   200K 面片的几何细节对于桌面操作物品绑绑有余。
> - **性能收益显著**：GLB 从 38 MB→~8 MB，仿真器加载/渲染速度更快，GPU 显存占用更低。
>   批量场景（100+ 物体）时差异尤为明显。
> - **4K 仅在需要时开启**：论文 figure、showcase demo 可用 `--quality high`。

```python
asset = lib.generate("red ceramic mug", image_path="mug.png")
# → TRELLIS.2-4B → O-Voxel → remesh → 1K PBR bake → GLB → mesh_to_urdf → URDF

# 快速模式（RL 批量训练）
asset = lib.generate("red ceramic mug", image_path="mug.png",
                     texture_size=512, decimation_target=100_000)
```

### 备选：Hunyuan3D-2.1

| 项目 | 值 |
|------|-----|
| 模型 | [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) (Tencent, 3.3B shape + 2B PBR) |
| 速度 | ~60s (shape) + ~30-60s (paint) = **~90-120s 总计** |
| VRAM | **~29 GB** |
| ROCm | **✅ 已验证** — MI300X/MI308X |
| 限制 | 内壁纹理破碎 · 底座 artifact (需 mesh_cleanup) · bpy 依赖 (需 lazy-import patch) |

```bash
robotsmith generate "red ceramic mug" --backend hunyuan3d --image mug.png
```

### 其他后端 (stub)

| 后端 | 模型 | PBR | VRAM | ROCm | 状态 |
|------|------|:---:|------|------|------|
| `triposg` | [TripoSG](https://github.com/VAST-AI-Research/TripoSG) 1.5B | ⚠️ | ≥6 GB | 待验证 | MoE Transformer, VRAM 低 |

#### TRELLIS.2 vs Hunyuan3D E2E 对比（red ceramic mug, MI308X, 512³）

| 指标 | Hunyuan3D-2.1 PBR | TRELLIS.2-4B |
|------|------|------|
| **总耗时** | 531s (shape 90s + paint 440s) | 423s (load 63s + gen 275s + export 85s) |
| **Shape 顶点** | 492,749 | 5,405,042 → 652,279 (decimated) |
| **Shape 面** | 985,500 | 10,834,068 → 992,967 (decimated) |
| **GLB 大小** | 1.1 MB | 38.6 MB |
| **纹理** | PBR Paint (diffuse map, 153 KB) | O-Voxel PBR (4K texture, 13.6 MB) |
| **纹理质量** | 杯体外部尚可，内壁纹理破碎明显 | 4K 分辨率，PBR 完整 (base color + metallic + roughness) |
| **底座 artifact** | 有（需 mesh_cleanup 裁剪） | **无** ✅ |
| **bpy 依赖** | 需 lazy-import patch 绕过 | **无**，o_voxel 原生 GLB 导出 |
| **ROCm 安装** | 较简单（pip + 两个 C++ extension） | 较复杂（需 ROCm fork: cumesh, nvdiffrast, flexgemm, flash_attn） |

> **结论：TRELLIS.2 已替代 Hunyuan3D 成为默认后端。**
> 优势：PBR 纹理完整嵌入 GLB（默认 1K, 可选 512/4K），无底座 artifact，无 bpy 依赖，mesh 质量显著更高。
> Hunyuan3D 保留为 `--backend hunyuan3d` 备选。

> 3D 生成模型全景调研（三层分类、Layer 1/2/3 详细对比、升级路径）见 [附录 A](#附录-a3d-生成模型全景调研)。

### Mesh → Sim-ready 转换

| 步骤 | 工具 | 说明 |
|------|------|------|
| 底座清理 | `mesh_cleanup.py` | 移除底部平面 artifact（仅 Hunyuan3D 备选需要，TRELLIS.2 默认无此问题） |
| 去退化面 | trimesh | 移除零面积三角形和孤立顶点 |
| 尺度标准化 | trimesh | 归一到物理尺寸 (meters) |
| 碰撞体生成 | trimesh convex_hull | 凸包近似 (MVP)；后续 V-HACD / CoACD 凸分解 |
| 物理属性 | 自动估算 | 体积 → 质量 (默认 800 kg/m³)，惯性张量 (bbox fallback) |
| URDF 打包 | `mesh_to_urdf` | visual.glb + visual.obj + collision.obj + model.urdf + metadata.json |

## 1.4 Text-to-Image 桥接组件

### 为什么需要 T2I？

所有 SOTA 3D 生成模型（Hunyuan3D-2.1、TRELLIS.2、TripoSG 等）均为 **image-to-3D**，不支持纯文本输入。
RoboSmith 以文本查询驱动检索，未命中时需要参考图片才能调用 3D 生成。

```
text query: "红色水杯"
       │
       └→ search() 未命中
              │
              ▼
       T2I (SDXL-Turbo, 512×512, ~3s) ──→ reference.png ──→ Hunyuan3D-2.1 ──→ mesh ──→ URDF
       prompt: "a red mug, front orthographic view, symmetrical, solid white bg, sharp edges, ..."
```

### 行业对标

| 产品/项目 | T2I 方案 |
|-----------|----------|
| NVIDIA Omniverse | 不需要（资产库海量） |
| Rodin Gen-2 | 内置 T2I（自有多模态管线） |
| InstantMesh / PhysX-Anything | 需外部 T2I |

**结论**：资产库规模有限的项目，T2I 桥接是**必要组件**。

### 模型选型

| 模型 | 参数 | 质量 | ROCm 验证 | 许可 | 备注 |
|------|------|------|-----------|------|------|
| **[SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo)** ← 默认 | 3.5B | ★★★★☆ | ✅ RS-6 (0.3s/img, 8GB) | Apache-2.0 | 无需认证，4步推理 |
| [FLUX.1-dev](https://github.com/black-forest-labs/flux) ← 升级路径 | 12B | ★★★★★ | ⚠️ gated repo | Non-commercial | 需 HF Token |
| [SDXL Base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | 3.5B | ★★★★☆ | ✅ RS-6 (1.9s/img, 8GB) | Apache-2.0 | 25步推理 |

**当前默认：SDXL-Turbo** — 4步推理、0.3s/img、8GB VRAM、Apache-2.0、无需 HF 认证。
FLUX.1-dev 质量更高但为 gated repo（需 HuggingFace 认证），作为升级路径保留。

**分辨率：512×512** — Hunyuan3D-2.1 内部 DINOv2 encoder resize 到 518×518，更高分辨率不提升 3D 质量。

### Prompt 工程 — 3D 重建友好

T2I 生成的图片直接送入 Image-to-3D 模型，prompt 质量直接决定 3D 资产质量。

**核心理念：从"描述"到"约束系统"**

传统 prompt 是自然语言描述（"一个红色杯子的产品照片"），但 3D 重建友好的 prompt
本质是**生成约束规范** — 每个关键词都对应一条显式约束条件。

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

> **CLIP 77 token 限制**：SDXL 的 CLIP text encoder 最多处理 77 个 token，
> 超出部分被截断。上述模板约 65 tokens（含 object 描述），留 ~12 tokens 给物体名称。
> 原始完整版（93 tokens）导致 `no specular highlight, object-only, high detail` 被丢弃。
> 关键约束（`no surface, no ground, no table`）必须放在前 77 tokens 内。

| 约束关键词 | 对应的 3D 重建问题 | 解释 |
|-----------|------------------|------|
| `isolated object, object-only` | 多物体干扰 | 比 `single object centered` 更强的孤立语义 |
| `no environment, no surface, no ground, no table` | **底部多余几何** | 模型默认"物体必须放在某处"，必须显式打断 |
| `no cast shadow, no contact shadow` | 阴影被当成结构 | 仅 `no shadows` 不够，模型仍生成接触阴影 |
| `no directional light` | 明暗对比→假几何 | 比 `soft lighting` 更精确 |
| `no specular highlight` | 高光被当结构 | 补充 `no reflection` 的盲区 |
| `opaque, matte finish, solid material` | 透明/反光物体重建差 | 显式约束材质属性 |
| `clean silhouette, sharp edges` | 拓扑不干净 | 轮廓越清晰→凸包越准→碰撞体越好 |

> **注意**：`symmetrical shape` 在 SDXL-Turbo 某些 seed 下被字面理解为"物体应左右对称" →
> 导致杯子生成双把手。已从默认模板移除。如需对称约束，放在 negative prompt 中用
> `asymmetric, lopsided` 反向引导更安全。

> **禁用词**（实测确认会触发复杂场景）：
> - ~~`product photography`~~ — SDXL-Base 训练数据中此标签关联复杂场景/台座/布景
> - ~~`studio lighting`~~ — 触发戏剧性打光
> - ~~`orthographic view`~~ — 被解读为建筑/工业设计风格
> - ~~`1:1 square composition`~~ — 触发排版/构图模式

#### 反向 Prompt（CFG > 0 的模型必须使用）

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

> Negative prompt 对 SDXL-Turbo (`guidance_scale=0.0`) 无效。
> 对 SDXL-Base 等使用 CFG 的模型，negative prompt **比正向 prompt 更重要** —
> 它直接告诉模型"推离"哪些模式。

#### 参数约束（比 prompt 更重要）

| 参数 | SDXL-Turbo | SDXL-Base | 说明 |
|------|:-:|:-:|------|
| `guidance_scale` | **0.0** | **≤5.0** | Base 的 7.5 是崩坏根源（CFG 放大效应） |
| `resolution` | 512×512 | **≥768×768** | Base 在 512 极易崩坏（训练分辨率为 1024） |
| `num_steps` | 4 | 25 | |
| `negative_prompt` | 无效 | **必须使用** | |
| `prompt tokens` | ≤77 | ≤77 | **CLIP 硬限制**，超出部分被静默截断 |

> **为什么 SDXL-Base 在 512 崩坏？** SDXL 系列训练分辨率为 1024×1024，
> 512 远低于训练分布，模型的 positional encoding 和 attention 模式退化，
> 导致构图碎片化（条纹、重复纹理）。768 是可接受的最低值。
>
> **为什么 CFG=7.5 有害？** CFG 本质是 `output = unconditional + scale × (conditional - unconditional)`。
> 当 scale=7.5 时，模型把"product photography"的条件信号（复杂场景）放大 7.5 倍。
> SDXL-Turbo 通过 ADD 蒸馏把 guidance 烤进权重，不需要运行时 CFG。

#### 绝对禁止的关键词

| 禁忌 | 原因 |
|------|------|
| `transparent, glass, see-through` | 透明物体重建极差 |
| `fantasy, glowing, melting, abstract` | 几何会混乱 |
| `product photography, studio lighting` | 触发复杂场景联想（LAION 审美偏差） |
| `complex patterns, dense textures` | 重建会糊、飞面 |
| `watercolor, ink wash, soft focus` | 边缘模糊 → 拓扑脏 |
| `orthographic, 1:1 composition` | SDXL-Base 过度解读为排版模式 |

#### 常见踩坑与修复

| 3D 缺陷 | 原因 | Prompt 修复 |
|---------|------|------------|
| 杯口/碗口变尖 | 透视太斜 | `front view, slight top-down angle` |
| 模型侧面鼓包 | 原图不对称 | `symmetrical shape, centered in frame` |
| 手柄断裂 | 遮挡/角度侧 | 确保正面完整看到手柄 |
| **底部多余几何** | **桌面/地面被重建** | **`no surface, no ground, no table`** |
| 表面凹凸不平 | 高光被当结构 | `no specular highlight, matte finish` |
| **背景噪点变几何** | **渐变/纹理背景** | **`plain pure white background, no environment`** |

### 集成方案

```python
# robotsmith/gen/generate.py
if backend == "hunyuan3d" and "image_path" not in gen_kwargs:
    ref_image = text_to_image(
        prompt=T2I_PROMPT_TEMPLATE.format(obj=prompt),
        output_path=output_dir / "reference.png",
        negative_prompt=T2I_NEGATIVE_PROMPT,  # only effective when guidance_scale > 0
    )
    gen_kwargs["image_path"] = ref_image
```

### 实现路线

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 0 | 手动提供参考图 | ✅ 完成 |
| Phase 1 | 接入 SDXL-Turbo，自动生成参考图 (512×512) | ✅ **RS-6/RS-9 验证通过** |
| Phase 2 | 升级到 FLUX.1-dev（需 HF Token） | 🔜 可选升级 |
| Phase 3 | Prompt 工程：约束系统模板 + 参数调优 | ✅ **RS-8 完成** |
| Phase 4 | 多视角生成（T2I × N views → multi-view reconstruction） | 远期 |

## 1.5 资产检索

```
查询: "红色水杯"
   ├─ Tag 精确匹配: metadata.tags ∋ {"mug", "red"} → 命中
   └─ Embedding 语义匹配 (后续): CLIP cosine similarity > threshold → 命中
```

**MVP**：tag 匹配（已实现）。后续加 CLIP/SentenceTransformer embedding。

## 1.6 已知问题 & 计划解决方案

### 1.6.1 底部平面 artifact（Base Plane）

> **状态更新**：切换默认后端为 TRELLIS.2-4B 后，底座 artifact **不再出现**。
> 以下分析保留作为 Hunyuan3D 备选后端的参考。

Hunyuan3D（以及大多数 image-to-3D 模型）生成的 mesh 会附带虚假的底部平面/底座，
即使输入参考图不包含地面或桌面。

| 项目 | 说明 |
|------|------|
| **根因** | Shape 模型（3.3B）训练数据偏差 — 训练集中物体都放在平面上，模型学到"底部应有水平面"先验 |
| **影响范围** | Shape 和 PBR 模式都存在（Paint 只做纹理不改几何，底座来自 Shape） |
| **官方状态** | [GitHub Issue #191](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/issues/191)，open，无官方修复 |
| **仿真影响** | 底座干扰碰撞检测和抓取行为 |

**缓解方案 & 风险**：

| 步骤 | 方法 | 风险 |
|------|------|------|
| **Mesh trim**（计划中） | 移除法向量朝下 + 位于最底部 5% 高度的面 | 可能削掉杯底等合理薄壁几何 |
| **水密性退化** | trim 后 mesh 底部出现开放边缘，不再 watertight | 体积/质量退化为 bbox fallback（误差 < 20%，桌面抓取可接受） |
| **补洞修复**（暂不实现） | 边界环检测 → 三角化填充 → UV 插值 | 工程成本高，易引入新 artifact |

**当前策略**：trim 底座 + bbox fallback 物理估算，不补洞。
如后续需要 watertight（如流体仿真），考虑 [ManifoldPlus](https://github.com/hjwdzh/ManifoldPlus)
或 trimesh `fill_holes()`。

### 1.6.2 社区方案对比：为什么 GraspVLA 没有底座问题？

> 核心区别：**策划资产** vs **生成资产**。

| 维度 | GraspVLA / 大规模合成数据项目 | RoboSmith (当前) |
|------|------|------|
| **3D 资产来源** | [Objaverse](https://objaverse.allenai.org/) — 百万级社区资产库 | Hunyuan3D-2.1 实时生成 |
| **资产制作方式** | 人工建模 / 3D 扫描 / 社区上传 → 人工策划筛选 | Image-to-3D 模型自动生成 |
| **底座问题** | **无** — 艺术家/扫描流程不产生此 artifact | **有** — 生成模型训练数据偏差 |
| **资产数量** | 240 类，数万高质量 mesh | 按需生成，无数量限制 |
| **定制能力** | 受限于库存 — 要什么有什么但不可定制 | 完全定制 — 文字描述→新资产 |
| **sim-ready 程度** | 需转换（Objaverse 原始格式非 URDF） | 管线内直接输出 URDF + 碰撞体 |

**GraspVLA 具体流程**：
1. 从 Objaverse 选择目标类目（mug, bowl, bottle 等）
2. 人工/自动筛选高质量、拓扑干净的 mesh
3. 标准化尺寸 + 生成碰撞体 + 打包 URDF
4. 大规模并行渲染（Isaac Sim / PyBullet）

这些资产本身就是"干净"的 — 由真人建模或高精度扫描设备采集，不经过生成模型，
自然不存在训练偏差导致的 artifact。

**对 RoboSmith 的启示**：

| 策略 | 说明 | 优先级 |
|------|------|:---:|
| **混合策略（推荐）** | 高频类目用 Objaverse/GSO 策划资产，长尾类目用生成 pipeline | P0 |
| **Mesh 后处理** | 对生成资产执行 base trim + 几何清洗 | P0 |
| **PhysX-Anything 升级** | 直接输出 sim-ready URDF + 物理参数（CVPR'26） | P1 |
| **开源资产库集成** | 批量导入 Objaverse / GSO / PartNet-Mobility | P1 |

> 结论：底座 artifact 是 Hunyuan3D 等模型的固有限制，但 TRELLIS.2 不存在此问题。
> 切换默认后端为 TRELLIS.2 后，生成资产质量接近策划资产水准。
> RoboSmith 的"搜索不到再生成"策略现已在质量上基本可行。

### 1.6.3 Sim-ready 成熟度 & 改进路线

一个仿真就绪资产需要满足 4 个层级：

```
层级    要求                              状态
─────   ────                              ──────
L0      能加载到仿真器，不崩溃              ✅ 已完成
L1      碰撞生效（物体不互穿）              ⚠️ 凸包近似丢失凹面细节
L2      材质精确的物理属性                  ❌ 尚未实现
L3      视觉逼真（PBR、纹理、颜色）         ✅ TRELLIS.2 PBR (1K 默认, 可选 512/4K)
```

**视觉层 (L3) 缺失项**：

| 缺失项 | 当前状态 | 期望状态 | 影响 |
|--------|---------|---------|------|
| 颜色/材质 | URDF `<visual>` 无 `<material>` | 逐物体 RGBA / PBR 材质 | 策略无法区分物体 |
| 纹理 | ✅ TRELLIS.2 PBR (1K 默认, 可选 4K) | — | ✅ 已解决 |
| 网格质量 | ✅ TRELLIS.2 200K faces (默认, 可选 1M) | — | ✅ 已解决 |

**物理层 (L2) 缺失项**：

| 缺失项 | 当前状态 | 期望状态 | 影响 |
|--------|---------|---------|------|
| 密度 | 统一 800 kg/m³ | 按材质: 陶瓷 ~2400, 金属 ~7800 | 质量不准 → 动力学不真实 |
| 摩擦系数 | 统一 0.5 | 陶瓷 ~0.3, 橡胶 ~0.8 | 抓取打滑或异常粘连 |
| 碰撞几何体 | 单凸包 | 凸分解 V-HACD / CoACD | 杯柄、碗腔丢失 → 抓取失败 |
| 接触参数 | 未设置 | 刚度、阻尼 | 接触弹跳或发软 |

**改进路线**：

| 优先级 | 任务 | 方案 |
|--------|------|------|
| ~~P0~~ | ~~底座 artifact 去除~~ | ✅ 已解决 — 切换 TRELLIS.2 默认后端（Hunyuan3D 备选仍需 mesh_cleanup） |
| P0 | 策划资产库扩充 | 导入 Objaverse / GSO 高频类目（§1.6.2） |
| P0 | 材质感知密度+摩擦 | prompt 推断材质 → 查找表 |
| P1 | URDF 颜色 | 提取 prompt 颜色关键词 → `<material>` |
| P1 | 更好的碰撞体 | V-HACD / CoACD 凸分解 |
| P2 | Genesis 材质属性 | 接触刚度/阻尼 → URDF 或 Genesis 配置 |
| P2 | PhysX-Anything 升级 | Layer 2 直接生成 sim-ready URDF（§附录 A） |

### 1.6.4 bpy (Blender Python) 依赖与 PBR 纹理输出

Hunyuan3D-2.1 PBR Paint pipeline 在最后一步调用 `convert_obj_to_glb`（位于
`hy3dpaint/DifferentiableRenderer/mesh_utils.py`），该函数使用 Blender Python
模块 (`bpy`) 将纹理化的 OBJ 转为 GLB。但 **`bpy` 没有 Python 3.12 的 wheel**：

| bpy 版本 | 支持 Python | 状态 |
|----------|------------|------|
| 4.2.x ~ 4.5.x | `==3.11.*` only | ❌ ROCm 镜像均为 3.12 |
| 5.0 ~ 5.1 | `==3.13.*` only | ❌ ROCm 镜像均为 3.12 |

Blender 从 3.11 直接跳到了 3.13，完全跳过了 3.12。验证结果：

- `pip install bpy==4.5.8` → pip 拒绝（no matching distribution）
- 手动解压 cp311 wheel → C 扩展可加载，但 `__init__.so` 内有硬编码版本检查 → `ImportError`
- `conda create python=3.11` 可行但需重装全部 deps（PyTorch ROCm 等），代价极高

**关键发现**：PBR 绘制的核心（UV 展开 xatlas、多视图渲染 custom_rasterizer、
神经纹理烘焙、inpaint）**完全不需要 bpy**。bpy 仅用于最终的 OBJ→GLB 格式转换。

**解决方案（已实施）**：

1. **Lazy-import patch**：将 `mesh_utils.py` 顶层 `import bpy` 移至仅 7 个
   Blender 相关函数内部（`_setup_blender_scene`, `convert_obj_to_glb` 等）。
   这是标准 Python 模式，零风险。
2. **`save_glb=False`**：调用 paint pipeline 时不触发 `convert_obj_to_glb`，
   输出纹理化 OBJ（含 MTL + diffuse/metallic/roughness/normal map）。
3. **trimesh GLB 导出**：`mesh_to_urdf` 检测到纹理后自动导出自包含 GLB
   （纹理内嵌），无需 bpy。

```
Paint Pipeline 内部流程（bpy 依赖仅在最后一步）：

  white mesh → xatlas UV → multi-view render → diffusion enhance
            → bake texture → inpaint → save OBJ+MTL+maps ← 我们取到这里
                                                    │
                                         (save_glb=True 才走↓)
                                         convert_obj_to_glb (bpy) ← 已绕过
```

---

# Part 2：Sim-to-Policy 管线

> Part 1 的 URDF 资产 → 加载到物理仿真 → 采集轨迹 → 训练策略 → 评估。
> 复用 [lerobot-from-zero-to-expert](../lerobot_from_zero_to_expert/) 已有管线。

## 2.1 仿真平台

### 物理引擎选择

| 平台 | 优势 | 劣势 | AMD | 定位 |
|------|------|------|:---:|------|
| **MuJoCo** | 精确接触、关节丰富、URDF 直接加载、生态成熟 | 无原生 GPU 并行 | ✅ | **主力引擎** |
| **Genesis** | GPU 并行、速度快 | API 还在 beta，不支持 gsplat | ✅ | 大规模采集加速 |
| **PyBullet** | 零配置、pip install | 渲染质量低 | ✅ | 轻量回退 |

**决策**：MuJoCo 为主力（接触精度 + mjviser 可视化），Genesis 做大规模并行采集。

### URDF → MuJoCo 流程

```
RoboSmith Asset (URDF + OBJ)
       │
       ▼
  mujoco.MjModel.from_xml_path("model.urdf")
       │   MuJoCo 原生支持 URDF 加载
       ▼
  MjModel + MjData
       │
       ├── mj_step()   → 物理仿真
       └── mjviser      → Web 可视化 + 交互调试
```

MuJoCo 可直接加载 URDF（自动转 MJCF），因此 Part 1 产出的资产无需额外格式转换。
对于复杂场景，可导出 MJCF XML 统一管理多物体布局和机器人。

## 2.2 仿真可视化 — mjviser

[mjviser](https://github.com/mujocolab/mjviser) 是基于 viser 的 web MuJoCo viewer，与 RoboSmith Part 1 的 `SceneViewer` 共享同一渲染后端。

### 为什么选 mjviser？

| 维度 | Part 1 SceneViewer | Part 2 mjviser |
|------|-------------------|----------------|
| 底层 | viser (raw API) | viser + MuJoCo |
| 物理仿真 | ❌ 静态预览 | ✅ `mj_step` 实时仿真 |
| 关节控制 | 位置 slider | 关节 + 执行器 slider |
| 接触力 | ❌ | ✅ 可视化接触点、力方向 |
| 轨迹回放 | ❌ | ✅ timeline scrubber + speed control |
| 安装 | `pip install viser` | `pip install mjviser` |

### 用法

```python
import mujoco
from mjviser import Viewer

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
Viewer(model, data).run()  # 浏览器打开 http://localhost:8080
```

CLI 一键启动：

```bash
uvx mjviser scene.xml              # 直接运行，无需安装
mjviser scene.xml --port 7070      # 指定端口
```

### 扩展点

```python
from mjviser import Viewer

def my_step(model, data):
    """自定义 step：可注入 RL policy 或 IK 控制。"""
    data.ctrl[:] = policy(obs)
    mujoco.mj_step(model, data)

Viewer(model, data, step_fn=my_step).run()
```

三个回调：
- **`step_fn(model, data)`** — 每步仿真调用，注入控制策略
- **`render_fn(scene)`** — 每帧渲染调用，添加可视化叠加层
- **`reset_fn(model, data)`** — 重置时调用

进阶用法（自定义 GUI）：

```python
import viser
from mjviser import ViserMujocoScene

server = viser.ViserServer()
scene = ViserMujocoScene(server, model, num_envs=1)
scene.create_visualization_gui()

with server.gui.add_folder("RoboSmith Controls"):
    slider = server.gui.add_slider("Gripper Force", min=0, max=100, initial_value=50)
```

### 在 RoboSmith 中的角色

```
Part 1: SceneViewer (viser)          Part 2: mjviser (viser + MuJoCo)
────────────────────────             ──────────────────────────────────
资产入库后快速预览                     仿真环境调试
  └ URDF 渲染、位置调整                 └ 物理交互验证
  └ 场景布局检查                        └ 关节/接触力可视化
                                        └ 轨迹回放 & 审查
                                        └ Policy 在线可视化
```

### RoboSmith → mjviser 集成计划

| 阶段 | 内容 | 状态 |
|------|------|------|
| Step 1 | 单个 URDF 资产加载到 MuJoCo + mjviser 预览 | 📋 计划中 |
| Step 2 | 场景配置 → MJCF 导出 → mjviser 多物体仿真 | 📋 计划中 |
| Step 3 | IK scripted 轨迹 + mjviser 回放验证 | 📋 计划中 |
| Step 4 | Policy 在线推理 + mjviser 实时可视化 | 📋 计划中 |

## 2.3 轨迹采集

**整个 pipeline 最 tricky 的环节。**

### 从 lerobot 项目带来的经验

| 发现 | 教训 |
|------|------|
| Goal ambiguity → 0% 成功率 | observation 必须包含足够信息消除歧义 |
| State-Action 时序错位 → 0% | 先观测 s_t，再执行 a_t，顺序不能反 |
| MLP BC 天花板 30% | 简单架构在连续 goal space 不够 |
| SmolVLA 6min 达 60% | pretrained vision encoder 是关键 |
| IK scripted 100% → BC 60% | 数据质量不是瓶颈，policy 泛化才是 |

### 采集方法

| 方法 | 优势 | 劣势 | 适用 |
|------|------|------|------|
| **IK scripted** | 确定性高、可批量 | 每任务手写逻辑 | MVP 首选 |
| Motion planning (RRT) | 泛化复杂场景 | 慢、路径不自然 | 复杂场景 |
| RL 训练 policy | 可学复杂行为 | 需 reward 设计 | 长期 |
| Teleoperation | 数据最自然 | 不 scalable | seed data |

### 参数化轨迹生成

```python
def generate_pick_place_trajectory(
    robot, object_pose, target_pose,
    grasp_type="top", lift_height=0.15,
):
    # reach → grasp → lift → move → place → release
    ...
```

每加一种新任务就写一个新的轨迹生成函数。

## 2.4 策略训练

直接复用 lerobot 已有管线，不重写：

```bash
python lerobot/.../01_train_smolvla.py \
  --dataset-id ./pipeline_output/pick_place_100ep \
  --n-steps 2000 --batch-size 4
```

### 已验证 baseline

| 架构 | 数据量 | unseen 成功率 | 训练时间 |
|------|--------|-------------|---------|
| SmolVLA (450M) | 100 ep | 60% | 6 min |
| ACT (45M) | 100 ep | 60% | 42 min |

## 2.5 闭环评估

每个任务记录：数据量 → 训练配置 → unseen 成功率，作为 pipeline "质量报告"。

## 2.6 任务规划

| 任务 | 交互类型 | 难度 | 资产需求 |
|------|---------|:---:|---------|
| pick_place | 抓放 | 低 | 杯、碗、积木 |
| stack_blocks | 多步骤 | 中 | 积木 |
| open_drawer | 铰接操作 | 中 | 抽屉 |
| push_object | 连续接触 | 低-中 | 瓶、罐 |
| sort_by_color | 条件分支 | 中 | 多色积木、托盘 |

## 2.7 合成数据 VLA 生态

> 2025-2026 年多个项目已验证"纯合成数据 → VLA 训练"路线可行。
> RoboSmith Part 2 定位在 post-training 规模。

### 合成数据 → VLA 训练

| 项目 | 数据规模 | 训练方式 | 核心发现 |
|------|---------|---------|---------|
| [GraspVLA](https://github.com/PKU-EPIC/GraspVLA) CoRL'25 | 10亿帧，240类 | pre-train | 纯合成零样本迁移真机抓取 |
| [InternVLA-A1](https://github.com/InternRobotics/InternVLA-A1) | 63万轨迹，70任务 | pre-train | 合成数据 alone 匹配 π₀ 级别 |
| [EmbodiChain](https://github.com/DexForce/EmbodiChain) | 平台级 | 全流程 | 资产→仿真→数据→VLA 端到端 |
| [ET-VLA](https://arxiv.org/abs/2511.01224) | 少量合成 | SCP warm-up | 合成 warm-up + 真机 fine-tune +53% |

### 端到端仿真平台：Genie Sim 3.0 (AgiBot)

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
Genie Sim 3.0         5140资产 / 10000h+   pre-train (real2sim, 场景泛化, NVIDIA)
GraspVLA              240类 / 10亿帧       pre-train (Objaverse, 物品泛化)
InternVLA-A1          70任务 / 63万轨迹    pre-train
RoboSmith (计划)      10-30类 / 5k-10k     post-train (gen2sim, 物品泛化, AMD ROCm) ←
lerobot (已验证)       1类 / 100轨迹        fine-tune
```

### RoboSmith 的差异化价值与继续方向

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
TRELLIS.2 (MI308X) → URDF → MuJoCo/Genesis → LeRobot 训练。
这对 AMD 数据中心用户（高校、企业自建集群）有不可替代的价值。

→ **方向：保持 AMD ROCm 全栈可用性作为核心差异化**，
  Part 2 选型（MuJoCo + Genesis）天然 AMD 兼容，不要引入 NVIDIA 绑定依赖。

**Insight 3：轻量 Post-Training 数据 Toolkit 比全栈平台更实用**

Genie Sim 是 pre-train 级平台（10,000+ 小时数据），部署成本高（Isaac Sim + NVIDIA GPU）。
多数研究者/小团队的真实需求是：拿一个现有 VLA（π0.5、SmolVLA）→ 在特定任务上
fine-tune/post-train → 部署。RoboSmith 可以做这个环节的 **"数据定制工坊"**：

```
用户指定: "我要训练 Franka 抓取 5 种不同形状的杯子"
         │
         ▼
RoboSmith: 生成 5 种杯子变体 (TRELLIS.2)
         → 桌面场景 + 位置随机化
         → IK scripted 轨迹 (1000 ep)
         → LeRobot v3.0 数据集
         → 直接喂给 SmolVLA / π0.5 fine-tune
```

→ **方向：Part 2 的核心价值不是做"又一个仿真平台"，而是做"VLA post-training 数据生成器"**，
  一条命令从文本描述到可训练数据集。

**Insight 4：Genie Sim Sim2Real 数据验证了合成数据 ≥ 真实数据**

Genie Sim Benchmark 的 Sim2Real 实验（8 任务）显示：
纯合成数据训练的 policy 在真机上的成功率（82.8%）**超过**真实数据训练的 policy（75.2%）。
这从工业级实验彻底验证了合成数据路线的可行性，RoboSmith 的方向是正确的。

→ **方向：不需要怀疑合成数据路线本身**，关键是数据多样性和质量，而非数量。

**Insight 5：Object Variation > Scene Variation > Data Volume**

综合 Genie Sim、GraspVLA、InternVLA-A1 的经验，对固定臂桌面操作的优先级排序：

```
物品变体多样性 (形态/外观/尺寸)     >>> 最关键，直接决定 policy 泛化能力
  │
  ├── 同类别内变体 (10种不同的 mug)   >>> RoboSmith gen2sim 的核心优势
  ├── 跨类别覆盖 (mug + bottle + can)  >>> Objaverse 策划 + 生成兜底
  └── 物理属性多样性 (质量/摩擦)       >>> 需实现 material-aware 物理 (§1.6.3)
      │
位置/朝向随机化                       >> 已在 ProgrammaticSceneBackend 实现
任务类型广度                           >> pick/place/stack/push 覆盖
      │
光照/纹理/相机视角                     > P1，渲染层面 domain randomization
场景布局                               > 固定臂几乎不需要
```

→ **方向：RoboSmith Part 2 的数据采集应以 object variation 为第一随机化轴**。

**数据多样性 > 数据数量**（InternVLA-A1 关键经验）：

| 随机化维度 | 优先级 | 说明 |
|-----------|:---:|------|
| 物体位置/朝向 | P0 | 每次采集不同摆放 |
| 物体类别组合 | P0 | 同场景不同物体搭配 |
| 任务类型广度 | P0 | pick/place/stack/push 覆盖比单任务深采更重要 |
| 光照/桌面纹理 | P1 | 减少视觉 overfitting |
| 相机视角 | P1 | 至少 2-3 个视角变化 |

## 2.8 场景多样性 — 借鉴 GraspVLA-playground

> 参考 [GraspVLA-playground](https://github.com/MiYanDoris/GraspVLA-playground)（MuJoCo/Robosuite）。
> 其场景多样性方案中有三个维度的技术可直接迁移到 Genesis + `ProgrammaticSceneBackend`。

### 三维多样性拆解

| 维度 | GraspVLA 做法 | 核心依赖 | RoboSmith 迁移可行性 |
|------|-------------|---------|-------------------|
| **物体多样性** | Objaverse 预处理子集，`random.sample(candidates, N)` | 离线 mesh 简化 + stable pose 预计算 | ✅ 直接复用 AssetLibrary |
| **Layout 多样性** | trimesh `CollisionManager` 碰撞感知随机摆放 | 纯几何，与仿真器无关 | ✅ 可直接集成到 `ProgrammaticSceneBackend.resolve()` |
| **环境外观多样性** | 38 种地板 + 43 种墙壁纹理随机替换 | Robosuite MuJoCo texture | ⚠️ 需适配 Genesis `gs.surfaces` / texture API |

### 关键技术：碰撞感知的多物体摆放

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

### BDDL 的角色澄清

BDDL（Behavior Domain Definition Language）在 GraspVLA 中**仅用于向 Robosuite 声明"场景里有哪些物体、目标是什么"**，不负责 layout 计算。BDDL 本身不绑定 NVIDIA Omniverse（BEHAVIOR benchmark 绑 Omniverse，但 LIBERO benchmark 绑 MuJoCo）。

RoboSmith 使用 Genesis + URDF，**不需要引入 BDDL**。`SceneConfig` + `ProgrammaticSceneBackend` 已覆盖同等功能。

### RoboSmith 迁移方案

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

### 实施优先级

| 优先级 | 改动 | 状态 | 收益 |
|:---:|------|:---:|------|
| P0 | Step 1 碰撞感知摆放 | ✅ | 多物体场景不穿透 |
| P0 | Step 2 stable pose + workspace | ✅ | 物体物理合理摆放 |
| P0 | Step 3 genesis_loader bridge | ✅ | `collect_data.py --scene` 多物体场景 |
| P1 | Step 4 纹理/颜色随机化 | 📋 | 视觉 domain randomization |

### 实现笔记 (2026-04-14)

**已完成文件变更：**

| 文件 | 改动要点 |
|------|---------|
| `robotsmith/scenes/config.py` | +`workspace_xy`, `collision_margin`, `max_placement_retries` |
| `robotsmith/scenes/backend.py` | `_CollisionChecker` (FCL/AABB), `_pick_stable_pose`, `_load_collision_mesh`; `resolve()` 碰撞感知 |
| `robotsmith/scenes/genesis_loader.py` | 新文件：`load_resolved_scene()` → `GenesisSceneHandle` |
| `robotsmith/scenes/presets/tabletop_simple.py` | 简化：不再硬编码 position_range，依赖 workspace_xy |
| `robotsmith/assets/schema.py` | +`stable_poses` field |
| `robotsmith/assets/builtin.py` | 7 primitives + table + plane (清理旧 mug/bowl) |
| `scripts/compute_stable_poses.py` | 新脚本：convex_hull + `trimesh.compute_stable_poses(n_samples=500)` |
| `scripts/import_objaverse.py` | 按 10 品类 / 17 variants 下载 |
| `pipeline/collect_data.py` | +`--scene` / `--assets-root` 参数，集成 genesis_loader |
| `tests/test_scenes.py` | 37 tests passing (collision, stable pose, workspace, quat) |
| `tests/test_library.py` | 更新 BUILTIN_COUNT=9, 新品类 assertion |

**关键设计决策：**

1. **碰撞检测跨平台**：`python-fcl` 在 Windows 难装，实现 AABB fallback 保证本地单测通过。远端 Linux 使用 FCL 精确碰撞。
2. **Stable pose 远端计算**：`trimesh.compute_stable_poses()` 需要物理模拟，Windows 极慢。在 MI300X Linux 节点批量计算，`scp` 传回 `metadata.json`。
3. **`collect_data.py` 双模式**：`--scene` 模式用 ResolvedScene + genesis_loader；无参数保持原始 hardcoded cube 兼容，避免破坏已有工作流。
4. **资产体积控制**：Objaverse 资产走按需下载 (`scripts/import_objaverse.py`)，Git 仅 track json/urdf，`.gitignore` 排除 glb/obj/stl/png。

**下一步 (Next)：**

在远端 GPU (MI300X) 上用 `--scene tabletop_simple` 运行 `collect_data.py`，验证端到端 Genesis 场景加载 + 数据采集。

---

# 扩展方向

MVP 跑通后，pipeline 架构天然支持两个高价值扩展：

## 扩展 A：更强的 VLA 模型

数据格式不变（LeRobot v3.0），只换下游 policy：

| 模型 | 参数量 | 硬件 | 定位 |
|------|--------|------|------|
| **SmolVLA** | 450M | 单卡 3090 | ← MVP，快速迭代 |
| **OpenVLA** | 7B | 单卡 A100 (LoRA) | 中间档 |
| **π0.5** ([openpi](https://github.com/Physical-Intelligence/openpi)) | 更大 | 多卡 | SOTA，已集成 LeRobot |
| **GraspVLA** | — | L40s | 抓取专用，纯合成 pre-train |
| **InternVLA-A1** | 3B | — | 通用操作，合成数据匹配 π₀ |

## 扩展 B：World Model 替换轨迹采集

| 模型 | 开源 | 特点 |
|------|:---:|------|
| [DIAMOND](https://github.com/eloialonso/diamond) | ✅ | 纯 diffusion WM，部署简单 |
| [GigaWorld-0](https://github.com/open-gigaai) | ✅ | Video + 3D + 物理一体化 |
| [Kairos 3.0-4B](https://apnews.com/press-release/media-outreach/ace-robotics-open-sources-real-time-generative-world-model-kairos-3-0-4b-3a7b28af3090368478c26f4613504a6d) | ✅ | 原生具身 WM，72x 加速 |

---

# 附录

## 附录 A：3D 生成模型全景调研

> 行业正从"生成好看的 3D"过渡到"生成仿真可用的 3D"。
> 按 sim-ready 深度，开源模型可分三层：

### 三层全景

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

### Layer 1 详细对比

| 模型 | 团队 | PBR | VRAM | 速度 | ROCm | 要点 |
|------|------|:---:|------|------|------|------|
| [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) 4B | Microsoft | ✅ | ≥24 GB | ~3s (H100), ~275s (MI308X) | **✅ 已验证** | Mesh 质量天花板，CVPR'25 Spotlight。[ROCm fork](https://github.com/ZJLi2013/TRELLIS.2/tree/rocm) |
| [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | Tencent | ✅ | 10+21 GB | ~60s | **✅ 已验证** | **当前默认后端**。344K verts，AOTriton FA |
| [TripoSG](https://github.com/VAST-AI-Research/TripoSG) 1.5B | VAST-AI | ⚠️ | ≥6 GB | 快 | 待验证 | MoE Transformer，VRAM 低，纹理非 PBR |
| [TripoSF](https://github.com/VAST-AI-Research/TripoSF) | VAST-AI | 配合 SG | ≥12 GB | 中 | 待验证 | 1024³ 超高分辨率 mesh |
| [Pandora3D](https://github.com/Tencent/Tencent-XR-3DGen) | Tencent XR | ✅ | 中 | 中 | 可能 | 多阶段纹理 pipeline |
| [Rodin Gen-2](https://developer.hyper3d.ai/) 10B | Deemos | ✅ 2K | — | ~60s | — | **非开源** API only |

**ROCm 验证状态：**

| 模型 | 状态 | 备注 |
|------|------|------|
| **Hunyuan3D-2.1** | **✅ 已验证** | MI300X, ROCm 6.4, 60s, AOTriton FA |
| **TRELLIS.2** | **✅ 已验证** | MI308X, ROCm 6.4, [ZJLi2013/TRELLIS.2@rocm](https://github.com/ZJLi2013/TRELLIS.2/tree/rocm), ~275s@512³, 5.4M verts → 993K faces GLB |
| TripoSG | 待验证 | VRAM 低，可配合 Hunyuan3D-Paint 补 PBR |

### Layer 2 详细分析

**[PhysX-Anything](https://github.com/ziangcao0312/PhysX-Anything)** (CVPR'26) — 最值得关注：
- 单张图片 → URDF/XML + 关节 + 物理参数（质量、摩擦、质心）
- VLM-based，几何 token 压缩 193×
- 效果：几何误差 -18%，物理误差 -27%，机器人抓取成功率 +12%

**[SIMART](https://simart-mllm.github.io/)** (SIGGRAPH'26) — Layer 1 → sim-ready 的桥梁：
- 整体 mesh → 拆解铰接零件 + URDF（MLLM 统一框架）
- 意义：把 Hunyuan3D 的高质量 mesh 转成 sim-ready 铰接资产

**[Seed3D](https://github.com/Seed3D/Seed3D)** (ByteDance)：
- 单图 → 仿真就绪 3D，水密 mesh，Isaac Sim 验证。局限：绑 NVIDIA 生态

### RoboSmith 升级路径

| 路径 | 做法 | 视觉 | 物理 | 复杂度 |
|------|------|:---:|:---:|:---:|
| **A: Layer 1 + 转换** ← 当前 | T2I (SDXL-Turbo, 3D prompt) → Hunyuan3D-2.1 → mesh_to_urdf | ★★★★ | ★★☆ | 中 |
| **B: Layer 2 直接生成** | PhysX-Anything → URDF + 物理参数 | ★★★ | ★★★★ | 低 |
| **A+B 混合（推荐）** | Layer 1 visual mesh → SIMART/PhysX-Anything 补物理 | ★★★★ | ★★★★ | 高 |

## 附录 B：Scene-level 工具调研

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

## 附录 C：开源资产来源

| 库 | 数量 | 格式 | sim-ready | 备注 |
|---|---|---|---|---|
| [Objaverse](https://objaverse.allenai.org/) | **百万级** | mesh | ❌ 需转换 | **GraspVLA 等项目的核心资产来源**（§1.6.2） |
| [Google Scanned Objects](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research) | ~1000 | SDF | ✅ | 高质量 3D 扫描，无生成 artifact |
| [PartNet-Mobility](https://sapien.ucsd.edu/browse) | ~2000 | URDF | ✅ | 铰接物体，ManiSkill 使用 |
| [ManiSkill](https://github.com/haosulab/maniskill) | ~50 | URDF | ✅ | 预打包，可直接用 |
| [PyBullet](https://pybullet.org/) | ~20 | URDF | ✅ | 经典基础资产 |
| [ArtVIP](https://huggingface.co/datasets/X-Humanoid/ArtVIP) ICLR'26 | 206 铰接 | URDF/USD | ⚠️ | 铰接零件级别 |

> **策略建议（§1.6.2）**：高频类目（杯子、碗、瓶子等）优先从 Objaverse/GSO 策划导入，
> 长尾类目再走 Hunyuan3D 生成 pipeline。这是 GraspVLA 等大规模项目的验证路线。

## 附录 D：已有能力基线

来自 [lerobot-from-zero-to-expert](../lerobot_from_zero_to_expert/)：

| 能力 | 状态 | 复用方式 |
|------|:----:|---------|
| Franka IK 轨迹生成 | ✅ | → Part 2 轨迹采集 |
| LeRobot 数据集写入 | ✅ | → Part 2 数据管线 |
| SmolVLA 训练 (6min → 60%) | ✅ | → Part 2 策略训练 |
| 闭环评估 | ✅ | → Part 2 评估 |
| AMD GPU 全栈 | ✅ | → 全项目 ROCm 兼容 |

前期调研成果：

| 工具 | 节点 | 结果 |
|------|------|------|
| SceneSmith | MI308X | PARTIAL PASS — 核心 OK, bpy 需 Blender Docker |
| RoboGen | MI300X | PASS — PyBullet + VLM deps 全部 OK |

## 附录 E：风险

| 环节 | 预期困难 | 应对 | 详见 |
|------|---------|------|------|
| ~~3D gen → 底座 artifact~~ | ~~Shape 模型训练偏差~~ | ✅ 已解决 — 切换 TRELLIS.2 默认后端 | §1.6.1 |
| 3D gen → sim-ready | 碰撞体不精确 | 凸分解 V-HACD / CoACD | §1.6.3 |
| ~~PBR Paint → bpy 依赖~~ | ~~Blender 无 Python 3.12 wheel~~ | ✅ 已解决 — TRELLIS.2 无 bpy 依赖 | §1.6.4 |
| 生成物品尺寸不一致 | bbox 差异大 | 标准化尺度 + metadata 约束 | |
| URDF 导入 MuJoCo/Genesis | 格式兼容 | MuJoCo 原生支持 URDF；先 PyBullet 验证 | |
| 检索未命中兜底延迟 | 生成+转换耗时 | 预生成常见类目 + 缓存 | §1.6.2 |
| 生成资产质量参差 | 模型幻觉 / 拓扑脏 | 策划资产 + 生成后处理混合策略 | §1.6.2 |

## 附录 F：项目结构

```
robotsmith/
├── assets/                          # 资产数据根目录
│   ├── objects/                     # 内置资产 (git tracked)
│   │   ├── mug_red/                 #   model.urdf + metadata.json
│   │   └── ...
│   ├── generated/                   # 生成资产 (git ignored)
│   │   └── <name>_<timestamp>/      #   model.urdf + visual.obj + collision.obj + metadata.json
│   └── catalog.json                 # 全量索引 (git tracked, 轻量 JSON)
├── robotsmith/                      # Python package
│   ├── assets/
│   │   ├── library.py               # AssetLibrary: 检索 + 入库 + catalog.json 持久化
│   │   ├── schema.py                # Asset / AssetMetadata dataclass
│   │   ├── search.py                # tag 检索引擎
│   │   └── builtin.py               # 12 个内置 URDF 物品
│   ├── gen/
│   │   ├── backend.py               # GenBackend ABC + 注册表
│   │   ├── hunyuan3d_backend.py     # Hunyuan3D-2.1 后端 (默认)
│   │   ├── generate.py              # generate_and_catalog 入口
│   │   ├── catalog.py               # tags/naming + catalog_asset()
│   │   ├── mesh_cleanup.py          # 底座去除 + 退化面清理
│   │   └── mesh_to_urdf.py          # mesh → URDF 转换
│   ├── scenes/
│   │   └── backend.py               # 场景配置 → ResolvedScene
│   ├── viz/
│   │   └── scene_viewer.py          # viser 静态预览 (Part 1)
│   └── cli.py                       # CLI 入口
├── scripts/
│   ├── import_objaverse.py          # Objaverse → 10 类桌面资产导入
│   ├── sync_assets.py               # 远端 → 本地资产同步
│   ├── browse_assets.py             # 静态 HTML 资产画廊生成
│   └── render_mesh_local.py         # 本地 3D 渲染
├── demo/                            # 端到端复现脚本
├── docs/
│   ├── design.md                    # 本文件
│   └── background.md                # 技术背景
├── tests/
├── experiments.md                   # 实验记录
├── .gitignore                       # 排除 generated/ 大文件
└── pyproject.toml
```

## 附录 G：参考链接

**3D 资产生成 & 转换**
- [SceneSmith](https://github.com/nepfaff/scenesmith) — Sim-ready 场景生成
- [mesh-to-sim-asset](https://github.com/nepfaff/mesh-to-sim-asset) — Mesh → SDF 碰撞体
- [ArtVIP](https://huggingface.co/datasets/X-Humanoid/ArtVIP) — 铰接物体资产库 (ICLR'26)
- [URDF-Anything+](https://github.com/URDF-Anything-plus/Code) — 图片 → URDF (ICML'26)

**场景多样性 & 评估**
- [GraspVLA-playground](https://github.com/MiYanDoris/GraspVLA-playground) — GraspVLA 仿真环境（Objaverse 物体 + 碰撞摆放 + 纹理随机）
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) — 终身学习操作 benchmark

**仿真平台 & 可视化**
- [MuJoCo](https://mujoco.org/) — 精确接触物理引擎
- [mjviser](https://github.com/mujocolab/mjviser) — Web MuJoCo viewer (viser)
- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) — GPU 并行物理仿真
- [PyBullet](https://pybullet.org/) — 轻量级物理仿真

**策略训练 & 评估**
- [LeRobot](https://github.com/huggingface/lerobot) — 数据格式 + 训练框架
- [SmolVLA](https://huggingface.co/lerobot/smolvla_base) — 轻量 VLA
- [π0.5 / openpi](https://github.com/Physical-Intelligence/openpi) — SOTA VLA
- [OpenVLA](https://github.com/OpenVLA/OpenVLA) — 7B 开源 VLA

**World Model**
- [GigaWorld-0](https://github.com/open-gigaai) — Video + 3D + 物理 WM
- [Kairos 3.0-4B](https://apnews.com/press-release/media-outreach/ace-robotics-open-sources-real-time-generative-world-model-kairos-3-0-4b-3a7b28af3090368478c26f4613504a6d) — 具身 WM
- [DIAMOND](https://github.com/eloialonso/diamond) — Diffusion WM
