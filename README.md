
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
 返回      T2I → 参考图 → 3D Gen (Hunyuan3D / TRELLIS.2) → URDF → 入库
 URDF      SDXL-Turbo        image → PBR GLB mesh        │
   │         ┌────────────────────────────────────────────┘
   ▼         ▼
 仿真就绪资产 (URDF + 碰撞凸包 + PBR 纹理 + 元数据)
```

> 当前所有 SOTA 3D 生成模型均为 image-to-3D，文本查询未命中时先经 T2I 生成参考图。
> 详见 [docs/design.md § 1.4](docs/design.md#14-text-to-image-桥接组件)。

## 快速上手

```bash
pip install -e .

# 导入 Objaverse 高质量资产（10 类桌面物品，首次运行需下载）
pip install objaverse
python scripts/import_objaverse.py

robotsmith list                    # 列出全部资产
robotsmith search "cup"            # 搜索
robotsmith scene tabletop_simple   # 解析场景预设

# 生成新资产（需 GPU，≥24 GB VRAM）
robotsmith generate "red ceramic mug" --image reference.png                   # 默认 Hunyuan3D
robotsmith generate "red ceramic mug" --image reference.png --backend trellis2 # TRELLIS.2
robotsmith generate "red ceramic mug"   # 无图：自动 T2I → 3D

# 验证全部资产
pip install pybullet
robotsmith validate
```

## 3D 生成

**资产策略**：10 类高频桌面物品从 [Objaverse](https://objaverse.allenai.org/) 策划导入（无生成 artifact），
搜索未命中时自动调用 Hunyuan3D-2.1 生成（含底座去除后处理）。
可插拔后端架构（`GenBackend` ABC），默认 Hunyuan3D-2.1（PBR 开启）。

| 后端 | 模型 | PBR | VRAM | ROCm | 状态 |
|------|------|:---:|------|------|------|
| **`hunyuan3d`** | [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | ✅ | ≥29 GB | ✅ MI300X/MI308X | **默认** |
| **`trellis2`** | [TRELLIS.2-4B](https://github.com/microsoft/TRELLIS.2) | ✅ | ≥24 GB | ✅ MI308X ([ROCm fork](https://github.com/ZJLi2013/TRELLIS.2/tree/rocm)) | **已验证** — 纹理质量更高 |
| `triposg` | [TripoSG](https://github.com/VAST-AI-Research/TripoSG) 1.5B | ❌ | ≥6 GB | 🔵 待验证 | stub |

默认管线（Shape + PBR Paint）：

```
参考图 → Shape (3.3B, ~60s, 10GB) → 白模 → Paint (2B, ~30-60s, 21GB) → 带纹理 GLB
```

> 模型权重开源：[tencent/Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1)。
> 两阶段管线详解 & 已知问题见 [docs/design.md § 1.3](docs/design.md#13-3d-生成管线)。

### 资产目录结构

```
assets/
├── objects/                              # 内置资产 (git tracked)
├── generated/                            # 管线生成资产 (git ignored)
│   └── red_ceramic_mug_20260408/
│       ├── model.urdf                    # 引用 visual.glb + collision.obj
│       ├── visual.glb                    # PBR 纹理 mesh
│       ├── visual.obj                    # OBJ fallback
│       ├── collision.obj                 # 碰撞凸包
│       ├── metadata.json                 # 物理属性 + tags
│       └── reference.png                 # T2I 参考图
└── catalog.json                          # 轻量索引 (git tracked)
```

## 场景预设

| 场景 | 描述 |
|------|------|
| `tabletop_simple` | 桌子 + 杯子 + 碗 + 3 个积木 |
| `kitchen_counter` | 桌子 + 杯子 + 盘子 + 瓶子 + 叉子 + 勺子 |
| `sorting_table` | 桌子 + 彩色积木 + 盘子目标区 |

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

## Demo：端到端复现

```bash
# 远端 GPU 节点（MI300X + ROCm 6.4）
bash demo/setup_env.sh           # 安装 Hunyuan3D-2.1 + RoboSmith
python demo/run_pipeline.py      # image → 3D → URDF → 入库 → 可视化
```

详见 [demo/README.md](demo/README.md)。

## Sim-ready 成熟度

```
层级   要求                          状态
────   ────                          ──────
L0     能加载到仿真器                 ✅
L1     碰撞生效                       ⚠️ 凸包近似（凹面丢失）
L2     物理属性精确                   ❌ 待实现
L3     视觉逼真（PBR）               ✅ Hunyuan3D PBR Paint
```

> 缺失项详析、改进路线见 [docs/design.md § 1.6](docs/design.md#16-已知问题--计划解决方案)。

## 测试

```
30 passed in 1.35s
  - test_library.py:      13 tests
  - test_mesh_to_urdf.py:  7 tests
  - test_scenes.py:       10 tests
```

远端 MI300X 端到端测试全部通过。

## 依赖

**核心（无需 GPU）：** `trimesh >= 4.0`, `numpy >= 1.24`

**可选：**
- `viser >= 1.0` — 3D 可视化 (`pip install -e ".[viz]"`)
- `pybullet >= 3.2` — 物理验证
- `torch >= 2.0` — 3D 生成（ROCm / CUDA）
- [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) — 默认 3D 后端
- [TRELLIS.2](https://github.com/ZJLi2013/TRELLIS.2/tree/rocm) — 高质量 PBR 3D 后端 (ROCm fork)
- `genesis-world >= 0.2` — Genesis 仿真器（Part 2）

## 设计文档

完整设计（架构、已知问题、改进路线、Part 2 规划）见 [docs/design.md](docs/design.md)。
