# 背景知识

RoboSmith 涉及的 3D 仿真核心概念。

---

## 水密网格（Watertight Mesh）

水密网格指**完全封闭、无缝隙的三角面片表面**。

类比：把网格想象成一个容器——如果往里灌水，水不会从任何缝隙漏出来，就是水密的。

```
水密网格                    非水密网格

  ┌────────┐               ┌────────┐
  │        │               │        │
  │  封闭  │               │  有缝隙 ╳ ← 面片缺失
  │        │               │        │
  └────────┘               └───  ───┘
  体积可计算                体积未定义
```

### 为什么仿真需要水密网格

| 需求 | 水密网格 | 非水密网格 |
|------|---------|-----------|
| 体积计算 | 正确 → 准确推导质量 | 未定义 → 质量只能靠猜 |
| 转动惯量 | 由体积积分精确计算 | 退化为包围盒近似，误差大 |
| 碰撞检测 | 内/外判定明确，不穿透 | 可能出现物体穿过缝隙 |
| 凸包/凸分解 | 输入合法，结果稳定 | 可能产生退化几何体 |

### 在 RoboSmith 中

Hunyuan3D-2.1 生成的网格通常是水密的。`mesh_to_urdf` 在转换时会检查 `mesh.is_watertight`：
- 水密 → 用 `mesh.moment_inertia` 精确计算惯性
- 非水密 → 退化为包围盒近似惯性（精度降低但不崩溃）

---

## trimesh

[trimesh](https://github.com/mikedh/trimesh) 是 Python 3D 网格处理库，功能类似 3D 领域的 NumPy。

### 核心能力

| 功能 | API 示例 | 说明 |
|------|---------|------|
| 加载网格 | `trimesh.load("obj/stl/glb")` | 支持 OBJ、STL、GLB、PLY 等主流格式 |
| 凸包计算 | `mesh.convex_hull` | 返回包裹原始网格的最小凸体 |
| 体积/质量 | `mesh.volume`, `mesh.mass` | 需水密网格；可设置 `mesh.density` |
| 转动惯量 | `mesh.moment_inertia` | 3x3 惯性张量，直接写入 URDF `<inertial>` |
| 水密检查 | `mesh.is_watertight` | 判断网格是否完全封闭 |
| 缩放 | `mesh.apply_scale(factor)` | 将网格缩放到目标尺寸（如米为单位） |
| 导出 | `mesh.export("out.obj")` | 导出为 OBJ、STL、GLB 等格式 |
| 包围盒 | `mesh.bounding_box`, `mesh.extents` | 获取轴对齐包围盒和尺寸 |

### 在 RoboSmith 中的用途

1. **缩放到真实尺寸** — 生成模型输出的网格尺度不确定，trimesh 将其缩放到目标尺寸（米）
2. **计算碰撞凸包** — `mesh.convex_hull` 生成碰撞几何体
3. **估算物理属性** — 体积 × 密度 = 质量；`moment_inertia` = 惯性张量
4. **导出 OBJ** — 分别导出 visual mesh（原始）和 collision mesh（凸包）供 URDF 引用

---

## 凸包近似（Convex Hull Approximation）

### 凸 vs 凹

- **凸形状（convex）**：形状内任意两点连线，线段完全在形状内部。球、方块、椭球都是凸的。
- **凹形状（concave）**：杯子把手有洞、碗有内腔——两点连线可能穿出形状外。大多数真实物体是凹的。

### 什么是凸包

凸包 = 包裹住原始形状的**最小凸体**。

直觉：用保鲜膜紧紧裹住物体，保鲜膜的形状就是凸包。或者想象在 2D 中用橡皮筋套住一组钉子，橡皮筋的形状就是这组点的凸包。

```
原始杯子（有把手）         凸包近似

   ┌──┐                    ┌────────┐
   │  │  ╭──╮              │        │
   │  │  │  │              │        │
   │  │  ╰──╯              │        │
   │  │                    │        │
   └──┘                    └────────┘
   有空洞/凹面              全部填实
```

### 为什么物理引擎偏爱凸形状

物理仿真中最频繁的操作是**碰撞检测**——判断两个物体是否接触、接触点在哪里。

- **凸-凸碰撞**：有高效精确算法（GJK + EPA），时间复杂度低，数值稳定
- **凹体碰撞**：需要逐三角面片检测，计算量随面片数爆炸；或者做空间划分，实现复杂

所以几乎所有物理引擎（PyBullet、MuJoCo、PhysX、Genesis）都优先使用凸碰撞体。

### 代价

凸包把凹面细节全部"填平"：

| 场景 | 真实物体 | 凸包碰撞体 | 后果 |
|------|---------|-----------|------|
| 抓杯子把手 | 手指穿过把手环 | 把手环被填实 | 手指被挡住，无法执行环握 |
| 物体放入碗中 | 物体落入碗腔 | 碗腔被填平成凸面 | 物体从碗上滑落 |
| 抓取 L 形零件 | 手指扣入凹槽 | 凹槽消失 | 只能用两指夹取，不能扣取 |

### 改进：凸分解（Convex Decomposition）

将一个凹物体拆成多个小凸块拼接，每个凸块单独参与碰撞检测：

```
原始杯子           凸分解（4 块）

   ┌──┐              ┌──┐
   │  │  ╭──╮        │A │  ╭B─╮
   │  │  │  │   →    │  │  │  │
   │  │  ╰──╯        │  │  ╰──╯
   │  │               │C─│─D│
   └──┘               └──┘
```

常用算法：
- **V-HACD**（Volumetric Hierarchical Approximate Convex Decomposition）— 经典方案，PyBullet 内置
- **CoACD**（Collision-Aware Convex Decomposition）— 更新，针对碰撞场景优化，分解质量更高

RoboSmith 当前使用单凸包（L1），计划集成 V-HACD / CoACD 实现更精确的碰撞（改进路线 P1）。

---

## URDF（Unified Robot Description Format）

URDF 是 ROS 生态中描述机器人和物体的 XML 格式。一个最简 URDF 包含：

```xml
<robot name="mug">
  <link name="base_link">
    <inertial>
      <mass value="0.25"/>
      <inertia ixx="..." ixy="0" ixz="0" iyy="..." iyz="0" izz="..."/>
    </inertial>
    <visual>
      <geometry><mesh filename="visual.obj"/></geometry>
    </visual>
    <collision>
      <geometry><mesh filename="collision.obj"/></geometry>
    </collision>
  </link>
</robot>
```

| 元素 | 作用 |
|------|------|
| `<inertial>` | 质量 + 转动惯量 → 物理引擎用于动力学计算 |
| `<visual>` | 渲染用网格 → 相机看到的外观 |
| `<collision>` | 碰撞检测用网格 → 通常比 visual 简化（凸包/凸分解） |

visual 和 collision 分开的原因：渲染需要高精度外观（万级面片），碰撞需要简化几何体（快速检测）。

### Genesis 兼容约定

RoboSmith 生成的 URDF 遵循 Genesis 要求：
- 单位：米（m）
- 坐标系：Z-up
- 网格格式：OBJ（Genesis 原生支持）
