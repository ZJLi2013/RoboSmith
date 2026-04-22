# General Object Grasp Solution — 不规则物体抓取技术方案全景

> 面向非规则物品（碗、杯、瓶、工具、玩具等）的机器人抓取，
> 从 "哪里能抓" → "怎么动" → "怎么 scale + 泛化" 三个层次梳理当前主流技术路线，
> 并给出横向对比与 RoboSmith 集成建议。
>
> 相关文档：
> - [design.md — Grasp Planning 设计原理](design.md#25-grasp-planning-设计原理) — 规划与执行分离、品类覆盖
> - [design.md — 品类扩展](design.md#graspplanner-品类扩展) — 品类扩展路线

---

## 0. 三层框架总览

```
┌────────────────────────────────────────────────────────────────────────┐
│  Layer 3 — 系统 / SDG / Scale                                         │
│  "怎么 scale + 泛化？"                                                 │
│  T(R,O)-Grasp · CaP-X                                                 │
│  跨 embodiment · 关系建模 · code-as-policy · RL post-training          │
├────────────────────────────────────────────────────────────────────────┤
│  Layer 2 — Policy（行为生成）                                           │
│  "怎么动？"                                                             │
│  Diffusion Policy · AnchorDP3 · Spatial RoboGrasp                     │
│  visuomotor policy · 3D representation · action chunk                  │
├────────────────────────────────────────────────────────────────────────┤
│  Layer 1 — Grasp / Affordance 生成                                     │
│  "哪里能抓？"                                                           │
│  Contact-GraspNet · GraspGen · GraspLDP                                │
│  6-DoF grasp pose · point cloud · diffusion · latent prior             │
└────────────────────────────────────────────────────────────────────────┘
```

**信息流**：Layer 1 产出一个或多个 **候选抓取姿态**（6-DoF SE(3) pose + gripper width），Layer 2 将其作为 **goal/condition** 融入策略网络输出完整运动轨迹，Layer 3 在更高层面解决 **跨物体/跨任务/跨 embodiment** 的 scale 问题。

三层并非严格串联 — 部分方案横跨两层（如 GraspLDP 同时产出 grasp prior 并驱动 policy；AnchorDP3 内嵌 affordance anchor）。但这个分层在 **问题定义** 层面是清晰的，下面逐层展开。

---

## 1. Layer 1：Grasp / Affordance 生成 —— "哪里能抓？"

### 1.1 问题定义

给定一个（可能未见过的）物体的观测（点云 / depth / mesh），输出一组 **6-DoF 平行夹爪抓取姿态** $\{(R_i, t_i, w_i)\}$，其中 $R \in SO(3)$ 为夹爪朝向，$t \in \mathbb{R}^3$ 为抓取点，$w$ 为 gripper opening width，附带质量评分 $q_i \in [0,1]$。

核心挑战：
- **几何多样性**：碗、瓶、螺丝刀的可行抓取分布差异巨大
- **杂乱场景**：遮挡、叠放导致可用接触面减少
- **实时性**：闭环控制要求 inference > 10 Hz

### 1.2 Contact-GraspNet（NVIDIA, ICRA 2021）

| 维度 | 说明 |
|------|------|
| **核心思路** | 将点云中每个 3D 点视为**潜在接触点**，预测该点处的 grasp 参数（approach direction + baseline rotation + gripper width + confidence），把 6-DoF 降为 **4-DoF 回归 per contact** |
| **输入** | 单帧 depth → 场景点云（~20K points） |
| **输出** | 每个 point 的 grasp 参数 + confidence → 通过 NMS 得到 top-K grasps |
| **网络** | PointNet++ backbone → per-point head (4-DoF param + grasp width + success score) |
| **训练数据** | **17M 模拟抓取**（GraspNet-1Billion 变体），IsaacSim 仿真 force-closure 标注 |
| **关键创新** | ① 将 contact point 作为 grasp anchor 大幅降低搜索空间；② scene-level 端到端（不需要先做分割再 per-object） |
| **Sim2Real** | 直接从 depth → real robot（Panda）无 fine-tune，>90% 成功率 |
| **推理速度** | ~200ms / scene（包含点云预处理） |
| **局限** | ① 仅限平行夹爪（parallel-jaw），不支持灵巧手；② 点云质量敏感（深度传感器噪声）；③ 无 task-awareness（只管"抓稳"，不管"怎么用"） |

**技术要点**：Contact-GraspNet 的核心 insight 是 **"每个表面点就是一个候选接触"**，避免了在完整 SE(3) 空间做采样/搜索。训练时对 17M grasps 做 force-closure 评估标注成功/失败，网络学习 point-wise grasp probability。

### 1.3 GraspGen（NVIDIA, ICRA 2026）

| 维度 | 说明 |
|------|------|
| **核心思路** | 将 6-DoF grasp 生成建模为 **Diffusion-Transformer (DiT)** 的条件去噪过程：从高斯噪声 iteratively denoise 出 grasp pose 分布 |
| **输入** | 物体点云 + gripper 参数 |
| **输出** | 一组 6-DoF grasp pose + quality score |
| **网络** | DiT 生成器 + 轻量 discriminator (on-generator training) |
| **训练数据** | **53M+ 模拟抓取**（新发布数据集），覆盖多种 gripper 类型 |
| **关键创新** | ① **On-generator training**：discriminator 只在生成器输出上训练（不在真实数据上），解耦数据分布偏移；② 支持 **多种夹爪**（parallel-jaw + industrial pinch + suction）；③ diffusion 天然建模多模态 grasp 分布 |
| **性能** | FetchBench SOTA，相比 prior art **+17%**；推理 ~20 Hz（TensorRT 前）；内存 **21× less** 于前代 discriminator |
| **Sim2Real** | 真实 Franka 上验证，noisy visual observation 下有效 |
| **局限** | ① 需要大规模仿真数据集（53M grasps 生产成本高）；② 纯 stability-oriented，无 task semantics |

**技术要点**：GraspGen 将 grasp 生成从 **回归问题** 转为 **生成问题**。Diffusion 的优势在于天然建模多模态分布 — 一个碗可能有"抓边缘"和"抓内壁"两种模式，回归模型倾向输出二者的平均（无效 grasp），diffusion 能保留两个 mode。On-generator training 让 discriminator 专注于生成质量排序，而非拟合真实 grasp 分布，减少 distribution shift。

### 1.4 GraspLDP（CVPR 2026）

| 维度 | 说明 |
|------|------|
| **核心思路** | 在 **latent diffusion policy** 中注入 **grasp prior**，使去噪过程不仅产出 action chunk，还隐式保证轨迹末端收敛到可行抓取构型 |
| **输入** | 腕部 RGB 相机 + robot state |
| **输出** | action chunk（EE delta 序列），末端隐式对齐到 grasp pose |
| **网络** | VAE encoder → latent space → conditional DDPM (decoder) + grasp prior injection + self-supervised image reconstruction |
| **关键创新** | ① **Grasp prior injection**：将离线 grasp detector 的预测（6-DoF pose）作为条件注入 latent diffusion 的每一步逆扩散；② **Self-supervised reconstruction**：在逆扩散中间步恢复 wrist-camera image，强制中间表征保留空间结构；③ 桥接了 Layer 1（grasp 在哪里）和 Layer 2（how to move） |
| **定位** | **跨 Layer 1/2** — 既包含 grasp affordance（prior）也包含 policy（diffusion） |
| **性能** | 仿真 + 真实机器人均超越 baseline diffusion policy，尤其在 **unseen object 泛化** 和 **spatial generalization** 上提升显著 |
| **局限** | ① 依赖预训练 grasp detector 的质量（prior 质量 → policy 上界）；② wrist camera 视角受限 |

**技术要点**：GraspLDP 最有价值的 insight 是 **"grasp prior 不应只在开始时提供一次目标，应在每一步 denoising 中持续引导"**。这与 classifier-guided diffusion 思路一致 — grasp prior 扮演 classifier guidance 的角色，将 diffusion 的 action 分布 pull 向 feasible grasp 子空间。Self-supervised reconstruction loss 则是防止 latent representation 退化的正则化手段。

### 1.5 Layer 1 横向对比

| 维度 | Contact-GraspNet | GraspGen | GraspLDP |
|------|:---:|:---:|:---:|
| **年份** | 2021 (ICRA) | 2025/2026 (ICRA'26) | 2026 (CVPR) |
| **范式** | 回归 (per-point) | 生成 (diffusion) | 生成 + policy (latent diffusion) |
| **输入** | 场景点云 | 物体点云 | wrist RGB + state |
| **输出** | grasp pose set | grasp pose set | action chunk (含 grasp) |
| **多模态** | 隐式（多 point 产出不同 grasp） | 显式（diffusion） | 显式（latent diffusion） |
| **训练数据量** | 17M grasps | 53M grasps | demonstration 数据 |
| **多 gripper** | 仅 parallel-jaw | parallel + pinch + suction | 依赖 prior detector |
| **Task-aware** | 否 | 否 | 部分（通过 policy conditioning） |
| **推理速度** | ~5 Hz | ~20 Hz | real-time (action chunk) |
| **跨 Layer** | 纯 Layer 1 | 纯 Layer 1 | Layer 1 + 2 |
| **Sim2Real** | 直接迁移 | 验证通过 | 验证通过 |
| **开源** | ✅ (NVlabs) | ✅ (NVlabs) | 代码未公开 |

**选型建议**：
- 只需 "给我一个抓取 pose" 的 SDG 场景 → **Contact-GraspNet**（成熟、开源、部署简单）或 **GraspGen**（更高性能、多 gripper）
- 需要端到端 policy（直接输出轨迹）→ **GraspLDP**（Layer 1+2 一体）
- 不想训练模型，想要 template/rule-based → RoboSmith **TemplateGraspPlanner**（见 [design.md — GraspTemplate](design.md#grasptemplate)）

---

## 2. Layer 2：Policy（行为生成）—— "怎么动？"

### 2.1 问题定义

给定当前观测（image / point cloud / state）和任务指令，输出一段 **robot action 序列**（EE delta / joint target / waypoints）完成操作。Layer 1 给出"抓在哪里"，Layer 2 解决"手怎么过去、过程中怎么避障、抓到后怎么运输放置"。

核心挑战：
- **多模态动作分布**：同一个 "pick up mug" 可能有多种合理路径
- **长时序依赖**：pick-and-place 需要协调 approach / grasp / transport / release
- **泛化**：场景变化（布局、光照、遮挡）下策略不崩

### 2.2 Diffusion Policy（Columbia / TRI, RSS 2023, IJRR 2024）

| 维度 | 说明 |
|------|------|
| **核心思路** | 将 visuomotor policy 表示为**条件去噪扩散过程**：给定观测，对 action sequence 做 iterative denoising 得到 action chunk |
| **输入** | RGB image(s) + robot state（关节角 / EE pose） |
| **输出** | action chunk $\{a_t, a_{t+1}, \ldots, a_{t+H}\}$，$H$ 为 prediction horizon |
| **网络** | 两种 backbone：① CNN-based U-Net；② **Diffusion Transformer (DiT)** |
| **训练** | Behavior Cloning — 从 demonstrations 学习 $p(a_{t:t+H} \mid o_t)$ |
| **关键创新** | ① **Action chunk**：一次预测多步动作，receding horizon 执行，减少 compounding error；② **Multimodal capacity**：diffusion 天然处理多模态，不像 GMM/BC 需要显式 mixture 建模；③ **Visual conditioning**：FiLM / cross-attention 将 image feature 注入 denoising |
| **性能** | 12 tasks × 4 benchmarks，平均 +46.9% vs prior SOTA |
| **局限** | ① 推理慢于 explicit policy（需多步 denoising，DDIM 加速后 ~10 steps）；② 纯 2D visual feature，缺乏 3D spatial reasoning；③ 对 demonstration 质量敏感 |

**技术要点**：Diffusion Policy 的核心贡献不在于 diffusion 本身，而是证明了 **action space 的 denoising diffusion 比显式回归/分类在 robot manipulation 中更有效**。关键 trick：(a) 使用 exponential moving average (EMA) 稳定训练；(b) DDIM 加速 inference（100 step → 10 step without quality loss）；(c) 观测 horizon vs action horizon 解耦。

### 2.3 AnchorDP3（arXiv 2025）

| 维度 | 说明 |
|------|------|
| **核心思路** | 在 DP3（3D Diffusion Policy）基础上，用 **affordance anchor** 替代 dense 轨迹预测：diffusion 只预测少数几个 **关键姿态**（pre-grasp, grasp, post-grasp），而非逐帧 action |
| **前置工作** | **DP3**（Ze et al., RSS 2024）：用稀疏点云作 3D representation 输入 diffusion policy，72 tasks +24.2% vs baselines |
| **输入** | 分割后的任务相关点云 + robot state |
| **输出** | sparse keyposes（关键姿态） + interpolated trajectory |
| **网络** | 3D point encoder (PointNet++) + task-conditioned feature encoder + diffusion action expert |
| **关键创新** | ① **Simulator-supervised segmentation**：用仿真 GT mask 自动分割 task-critical 物体点云（无需人工标注）；② **Task-conditioned encoder**：per-task 轻量 module，shared diffusion backbone → 高效多任务学习；③ **Affordance-anchored keypose diffusion**：预测 (pre-grasp, grasp, post-grasp) 这种 **语义锚点** 而非 dense trajectory，几何一致性约束帮助 diffusion 收敛 |
| **性能** | RoboTwin benchmark **98.7% 平均成功率**，极端随机化下（物体、杂乱、桌高、光照、背景） |
| **定位** | **Layer 1/2 融合** — affordance anchor 隐式包含 "抓哪里" 的信息 |
| **局限** | ① 依赖仿真器渲染 GT segmentation（sim-only）；② 双臂任务验证为主 |

**技术要点**：AnchorDP3 的最大贡献是将 diffusion policy 的预测从 **dense action sequence** 简化为 **sparse keypose set**。这有三个好处：(a) 降低 diffusion 的输出维度，收敛更快；(b) keyposes 有明确几何含义（pre-grasp = approach pose, grasp = contact pose），比 arbitrary action chunk 更易约束；(c) 插值 keyposes → dense trajectory 可以用 motion planner 保证运动学可行性。

**DP3 → AnchorDP3 的演进**：

| | DP3 (2024) | AnchorDP3 (2025) |
|---|---|---|
| 3D 输入 | raw 点云 | segmented 点云 |
| 输出粒度 | dense action chunk | sparse keyposes |
| 多任务 | per-task training | shared backbone + per-task head |
| Affordance | 无 | affordance anchor |
| 成功率 | 85% (4 real tasks) | 98.7% (RoboTwin) |

### 2.4 Spatial RoboGrasp（arXiv 2025）

| 维度 | 说明 |
|------|------|
| **核心思路** | 将 **spatial perception**（单目深度估计 + 6-DoF Grasp Prompt）融入 diffusion policy，构建 **depth-aware spatial representation** 作为统一观测 |
| **输入** | RGB image + estimated depth + 6-DoF grasp prompt (来自外部 grasp detector) |
| **输出** | action sequence (diffusion policy) |
| **关键创新** | ① **Monocular depth estimation** 补充 3D 信息（不依赖 depth sensor）；② **6-DoF Grasp Prompt**：将 Layer 1 的 grasp detection 结果编码为 visual prompt 叠加到 observation 上；③ **Domain-randomized augmentation**：光照/背景/遮挡随机化提升鲁棒性 |
| **定位** | **Layer 1 + 2 桥接** — 显式依赖外部 grasp detector 提供 spatial anchor |
| **性能** | 抓取成功率 +40%，任务成功率 +45%（vs raw RGB baseline），环境变化下泛化显著 |
| **局限** | ① 依赖单目深度估计质量（暗光/透明物体下降）；② grasp prompt 质量决定策略上界 |

**技术要点**：Spatial RoboGrasp 的核心 insight 是 **"perception 的质量决定 policy 的天花板"**。大多数 imitation learning 直接用 raw RGB 作输入，缺乏 3D spatial reasoning → 换个角度/光照就崩。通过 monocular depth + grasp prompt 构建 "spatially grounded" 观测，策略得到的 observation 本身就包含 3D 几何和抓取候选信息，大幅降低了 policy 的学习负担。

### 2.5 Layer 2 横向对比

| 维度 | Diffusion Policy | AnchorDP3 | Spatial RoboGrasp |
|------|:---:|:---:|:---:|
| **年份** | 2023 (RSS) | 2025 (arXiv) | 2025 (arXiv) |
| **3D 感知** | 无（纯 2D image） | 点云 (DP3 backbone) | 单目深度估计 |
| **Grasp Affordance** | 无 | 内嵌（affordance anchor） | 外部 grasp detector 提供 |
| **输出粒度** | dense action chunk | sparse keyposes | dense action chunk |
| **多任务** | per-task | shared + per-task head | per-task |
| **训练数据** | human demo | sim-generated (procedural) | human demo + domain randomization |
| **场景鲁棒性** | 中 | 极强（extreme randomization） | 强（depth + augmentation） |
| **Sim2Real** | real demo → real | sim → real (planned) | real demo → real |
| **硬件需求** | 中 | 高（点云 + GT seg） | 中 |

**选型建议**：
- 有足够 real demonstration → **Diffusion Policy**（最成熟、社区最大）
- 有仿真环境、需极高成功率 → **AnchorDP3**（sim-generated data + affordance anchor）
- 需 robust perception 但无 depth sensor → **Spatial RoboGrasp**（monocular depth + grasp prompt）

---

## 3. Layer 3：系统 / SDG —— "怎么 scale + 泛化？"

### 3.1 问题定义

Layer 1+2 解决的是 **单物体单任务** 的抓取。但真实应用要求：
- **跨物体**：从训练过的 mug 泛化到 unseen mug / bottle / tool
- **跨任务**：同一个 mug，"pick up" vs "pour water" 需要不同的 grasp + motion
- **跨 embodiment**：从 Franka parallel-jaw 迁移到 dexterous hand 或 industrial gripper
- **可扩展**：不能每个 (object, task, robot) 组合都人工标注/训练

Layer 3 的方案从 **关系建模** 和 **program-level abstraction** 两个方向解决 scale 问题。

### 3.2 T(R,O)-Grasp（arXiv 2025）

| 维度 | 说明 |
|------|------|
| **全名** | T(R,O) Grasp: Efficient Graph Diffusion of Robot-Object Spatial Transformation for Cross-Embodiment Dexterous Grasping |
| **核心思路** | 用 **图扩散模型** 建模 Robot (R) — Object (O) 之间的空间变换关系，而非直接预测 grasp pose |
| **输入** | 物体点云 + robot hand 模型 |
| **输出** | R-O 相对空间变换 $T(R,O) \in SE(3)$ → 反算出 grasp pose + finger configuration |
| **网络** | **T(R,O) Graph**：将 robot 和 object 分别作为 graph nodes，edges 编码空间关系 → Graph Diffusion Network 去噪得到最优空间变换 |
| **关键创新** | ① **关系建模**：不预测绝对 pose，而是预测 **robot-object 相对变换** — 天然 embodiment-agnostic；② **Graph structure**：robot links + object parts 作为 graph，GNN 捕获接触拓扑；③ **Cross-embodiment**：同一个 graph diffusion 可用于不同机器人手（改变 R 的 graph 结构即可） |
| **性能** | 94.83% 成功率，0.21s inference，41 grasps/sec (A100) |
| **跨 embodiment** | 验证 parallel-jaw → dexterous hand 迁移 |
| **局限** | ① 需要精确的 robot/object 几何模型（graph 构建依赖 mesh）；② 主要验证灵巧手抓取，平行夹爪场景优势不显著 |

**技术要点**：T(R,O)-Grasp 最核心的 insight 是 **"grasp 的本质是 R 和 O 之间的空间关系，而非 R 或 O 各自的绝对 pose"**。传统方法预测 "gripper 应该到 (x,y,z,R)" — 这是 embodiment-specific 的。T(R,O) 预测 "robot 和 object 应形成这样的相对构型" — 换一个 robot，只需重新解析 T(R,O) 为新 robot 的关节角。

**Graph Diffusion 的作用**：SE(3) 上的条件扩散，输入是 (R,O) graph 的初始 embedding（噪声），逐步去噪到 feasible 接触构型。Graph 结构自然编码了 "哪些 robot link 可能接触哪些 object surface" 的拓扑约束。

### 3.3 CaP-X（Microsoft, arXiv 2026）

| 维度 | 说明 |
|------|------|
| **全名** | CaP-X: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation |
| **核心思路** | **Code-as-Policy (CaP)**：让 LLM/VLM agent 生成 Python 代码组合感知和控制原语，而非学习端到端 policy |
| **组成** | ① **CaP-Gym**：39 tasks (RoboSuite + LIBERO-PRO + BEHAVIOR) 交互环境；② **CaP-Bench**：12+ frontier 模型评估；③ **CaP-Agent0**：training-free agentic framework (multi-turn visual diff + skill library + parallel reasoning)；④ **CaP-RL**：用环境 reward post-train LLM |
| **关键创新** | ① **Program-level abstraction**：robot 控制不是 "学一个神经网络"，而是 "写一个程序调用 skill primitives"；② **Visual differencing**：agent 执行→截屏→比较前后差异→修正代码；③ **CaP-RL**：用 sim reward post-training LLM（7B model 20% → 72%）；④ **Sim2Real transfer**：code-level policy 的 sim-to-real gap 极小（同一段代码，换 real API 即可） |
| **性能** | Frontier models >30% zero-shot；CaP-Agent0 = 18% on perturbed LIBERO-PRO（SOTA VLA = 13%）；CaP-RL = 72% after 50 iterations |
| **定位** | **最高抽象层** — 不关注 grasp pose / trajectory，关注 task decomposition + skill composition |
| **局限** | ① 依赖高质量 skill primitives（如果底层 pick() 失败，code 再好也没用）；② LLM inference 延迟高（~秒级），不适合 contact-rich 实时控制；③ 长程复杂任务中 code 正确性难保证 |

**技术要点**：CaP-X 代表了 **"agent paradigm"** 在机器人操作中的应用。与 Layer 1/2 的 "学习低级控制" 不同，CaP-X 假设低级技能已有（pick, place, push 等 API），问题是 **如何组合这些技能完成复杂任务**。

**CaP-RL 的关键**：传统 CaP 靠 LLM 的 zero-shot reasoning，效果受限于 LLM 对物理世界的理解。CaP-RL 在仿真中用 reward 做 RL post-training（类似 RLHF，但 reward 来自 sim），让 7B model 从 20% → 72%。这意味着 **小模型 + sim reward 可以超过大模型 zero-shot**。

### 3.4 Layer 3 横向对比

| 维度 | T(R,O)-Grasp | CaP-X |
|------|:---:|:---:|
| **年份** | 2025 (arXiv) | 2026 (arXiv, Microsoft) |
| **抽象层次** | Grasp geometry (SE(3)) | Task program (code) |
| **核心建模** | Robot-Object 空间关系 | Task decomposition + skill composition |
| **泛化维度** | 跨 embodiment (robot hand) | 跨 task (program reuse) |
| **是否需要 demonstration** | 仿真数据 | 可 zero-shot，CaP-RL 需 sim reward |
| **实时性** | 0.21s per grasp | 秒级 (LLM inference) |
| **底层依赖** | 精确 mesh model | 高质量 skill primitives |
| **适用场景** | 灵巧抓取、multi-embodiment 部署 | 多步任务、task-level planning |
| **开源** | 有 | ✅ (GitHub, MIT) |

---

## 4. 全局对比与技术选型

### 4.1 六方案一览

| 方案 | Layer | 核心范式 | 输入 | 输出 | 训练数据 | 推理速度 | 开源 |
|------|:---:|------|------|------|------|------|:---:|
| **Contact-GraspNet** | 1 | Point-wise regression | 场景点云 | grasp pose set | 17M sim grasps | ~5 Hz | ✅ |
| **GraspGen** | 1 | DiT diffusion | 物体点云 | grasp pose set | 53M sim grasps | ~20 Hz | ✅ |
| **GraspLDP** | 1+2 | Latent diffusion + grasp prior | wrist RGB + state | action chunk | demonstrations | real-time | ❌ |
| **Diffusion Policy** | 2 | Action diffusion | image + state | action chunk | demonstrations | ~10 Hz | ✅ |
| **AnchorDP3** | 1+2 | 3D affordance anchor diffusion | segmented 点云 | sparse keyposes | sim-generated | real-time | ✅ |
| **Spatial RoboGrasp** | 1+2 | Spatial perception + diffusion | RGB + est. depth + grasp prompt | action chunk | demo + aug | real-time | ❌ |
| **T(R,O)-Grasp** | 3 | Graph diffusion (R-O relation) | 点云 + robot model | R-O 空间变换 | sim data | ~5 Hz | 有 |
| **CaP-X** | 3 | Code-as-Policy agent | language + image | Python code | zero-shot / RL | 秒级 | ✅ |

### 4.2 按使用场景选型

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| **SDG pipeline（仿真数据生成）** | Contact-GraspNet / GraspGen + MotionExecutor | 只需 grasp pose → IK 执行，最轻量 |
| **Real robot imitation learning** | Diffusion Policy / GraspLDP | 从 teleoperated demo 学习，最成熟 |
| **Sim-to-Real extreme randomization** | AnchorDP3 | 98.7% 仿真 + sim-to-real planned |
| **跨 embodiment 部署** | T(R,O)-Grasp | 关系建模天然 embodiment-agnostic |
| **复杂多步任务** | CaP-X | program-level abstraction，skill composition |
| **无 depth sensor + 多变环境** | Spatial RoboGrasp | 单目深度 + grasp prompt + augmentation |

### 4.3 组合使用模式

上述方案并非互斥，实际系统往往 **跨层组合**：

```
模式 A: GraspGen (L1) → Diffusion Policy (L2)
  grasp candidate → policy conditioning → trajectory

模式 B: Contact-GraspNet (L1) → MotionExecutor (IK) → DART augmentation
  grasp pose → IK planning → noise injection → recovery data

模式 C: CaP-X (L3) → AnchorDP3 (L1+2) as skill primitive
  LLM decomposes task → each subtask calls AnchorDP3 policy

模式 D: T(R,O)-Grasp (L3) → GraspLDP (L1+2)
  relation-aware grasp → latent diffusion policy execution
```

---

## 5. 三层框架评价与补充

### 5.1 框架合理性分析

ChatGPT 给出的三层划分 **基本合理**，但有以下需要注意的点：

**合理之处**：
- **问题分解清晰**："哪里能抓" / "怎么动" / "怎么 scale" 是抓取研究的三个核心子问题
- **技术路线对应**：每层的方案确实在解决对应层的问题
- **渐进关系正确**：Layer 1 输出是 Layer 2 的输入/条件，Layer 3 在更高层面做组合

**需要修正 / 补充**：

| 点 | 说明 |
|------|------|
| **层间边界模糊** | GraspLDP、AnchorDP3、Spatial RoboGrasp 都横跨 L1+L2。实际趋势是 **层间融合**，而非严格分层 |
| **缺少 "感知层"** | Layer 1 隐含了 perception（depth → point cloud → grasp），但没有显式讨论 segmentation / depth estimation / scene understanding。Spatial RoboGrasp 的核心贡献恰恰在 perception，而非 policy |
| **Layer 3 内涵过广** | T(R,O)-Grasp 解决的是 cross-embodiment（同一个 grasp intent → 不同 robot），CaP-X 解决的是 task-level composition（多个 skill → 一个 task）。二者虽然都是 "scale"，但维度不同 |
| **缺少 sim-to-real** | 实际落地中 sim-to-real gap 是独立挑战，不属于任何一层但影响所有层 |
| **缺少 tactile / force** | 所有方案都是 vision-only，未涉及力/触觉反馈。对 deformable objects、fragile items 这是关键缺失 |

### 5.2 补充方案（值得关注）

| 方案 | 层 | 说明 |
|------|:---:|------|
| **AnyGrasp** (2022) | L1 | 清华 GraspNet 团队，大规模点云 grasp detection，SDK 最成熟 |
| **DexGraspNet** (2023) | L1 | 灵巧手大规模 grasp 数据集，5.3K objects × 1.3M grasps |
| **MimicGen** (2024) | L2/L3 | NVIDIA，少量 teleoperated demo → sim augmentation → 大规模数据 |
| **ShapeGen** (2026) | L1/L3 | 清华，functional correspondence warping，与 RoboSmith 最相关 |
| **TOG** (2025) | L1 | Training-free task-oriented grasp，VLM + grasp model 组合 |

---

## 6. RoboSmith 集成路线

基于上述分析，RoboSmith 的 Grasp Affordance Layer 应 **渐进集成**：

### Phase 3.0 — TemplateGraspPlanner（当前设计，无 ML）

```
Asset category → GRASP_TEMPLATES → GraspPlan → MotionExecutor → trajectory
```

对应 Layer 1 最简形态。优点：零依赖、可解释、不需要训练。
缺点：per-category 人工定义，对长尾物体无法覆盖。

### Phase 3.1 — SamplerGraspPlanner（mesh-based，无 ML）

```
Asset mesh → antipodal sampling → force-closure scoring → GraspPlan set
```

对应 Layer 1 经典方法（GPD 级别）。覆盖长尾物体，但不含语义。

### Phase 3.2 — LearnedGraspPlanner（集成 Contact-GraspNet / GraspGen）

```
Asset mesh → render point cloud → Contact-GraspNet / GraspGen → GraspPlan set
```

对应 Layer 1 SOTA。需要 GPU inference，但产出质量显著提升。
**推荐首选 Contact-GraspNet**：开源成熟、parallel-jaw 专用、RoboSmith 已有 Franka parallel-jaw。
后续可升级到 GraspGen（多 gripper + 更高性能）。

### Phase 3.3 — Diffusion Policy Integration（Layer 2）

```
GraspPlan (L1) → Diffusion Policy conditioned on grasp prompt → action trajectory
```

当 RoboSmith 的数据生产从 IK scripted 演进到 DAgger / Online RL 时，
Layer 2 的 diffusion policy 自然成为 data production 的核心。

### Phase 3.4 — CaP-X Style Task Composition（Layer 3）

```
TaskSpec (language) → LLM agent → skill sequence → per-skill GraspPlan + Policy
```

对应 CaP-X 的 program-level 组合。当 RoboSmith 支持多种 skill primitives 时，
可以用 LLM 组合实现长程任务的数据生成。

---

## 7. 结论

三层框架提供了一个 **有效的思考工具** 来理解不规则物体抓取的技术栈：

1. **Layer 1** 回答 "在哪里建立接触" — 从 per-point regression (Contact-GraspNet) 演进到 diffusion generation (GraspGen) 再到 grasp-prior-guided policy (GraspLDP)
2. **Layer 2** 回答 "整条轨迹怎么走" — Diffusion Policy 奠定了 paradigm，AnchorDP3 和 Spatial RoboGrasp 分别从 **3D affordance** 和 **spatial perception** 两个方向增强
3. **Layer 3** 回答 "如何超越单物体单任务" — T(R,O)-Grasp 通过关系建模实现 cross-embodiment，CaP-X 通过 code-as-policy 实现 task-level 组合

**趋势**：层间边界正在模糊化。最新方法（GraspLDP, AnchorDP3）倾向于在一个模型内同时解决 "抓哪里 + 怎么动"，而 CaP-X 则在更高层用 program abstraction 组合这些端到端模块。

**对 RoboSmith 的启示**：当前的 TemplateGraspPlanner (Phase 3.0) 是 Layer 1 最轻量的切入点，但中期应集成 Contact-GraspNet/GraspGen 以覆盖长尾物体。长期来看，整个 data gen pipeline 将从 "IK scripted" 演进到 "learned policy in the loop"（Layer 2），最终到 "agent-driven task composition"（Layer 3）。

---

> **附录 A：缩写表**
>
> | 缩写 | 全称 |
> |------|------|
> | DoF | Degrees of Freedom |
> | EE | End Effector |
> | SE(3) | Special Euclidean Group (3D rotation + translation) |
> | DiT | Diffusion Transformer |
> | DDPM | Denoising Diffusion Probabilistic Model |
> | DDIM | Denoising Diffusion Implicit Model |
> | BC | Behavior Cloning |
> | SDG | Synthetic Data Generation |
> | NMS | Non-Maximum Suppression |
> | VLM | Vision-Language Model |
> | VLA | Vision-Language-Action Model |
> | CaP | Code-as-Policy |
> | GNN | Graph Neural Network |
> | PBR | Physically Based Rendering |
>
> **附录 B：论文引用**
>
> | 方案 | 论文 | 会议 |
> |------|------|------|
> | Contact-GraspNet | Sundermeyer et al., "Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes" | ICRA 2021 |
> | GraspGen | Murali, Sundaralingam et al., "GraspGen: A Diffusion-based Framework for 6-DOF Grasping with On-Generator Training" | ICRA 2026 |
> | GraspLDP | "GraspLDP: Towards Generalizable Grasping Policy via Latent Diffusion" | CVPR 2026 |
> | Diffusion Policy | Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" | RSS 2023, IJRR 2024 |
> | DP3 | Ze et al., "3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations" | RSS 2024 |
> | AnchorDP3 | "AnchorDP3: 3D Affordance Guided Sparse Diffusion Policy for Robotic Manipulation" | arXiv 2025 |
> | Spatial RoboGrasp | "Spatial RoboGrasp: Generalized Robotic Grasping Control Policy" | arXiv 2025 |
> | T(R,O)-Grasp | "T(R,O) Grasp: Efficient Graph Diffusion of Robot-Object Spatial Transformation for Cross-Embodiment Dexterous Grasping" | arXiv 2025 |
> | CaP-X | Microsoft, "CaP-X: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation" | arXiv 2026 |
> | AnyGrasp | Fang et al., "AnyGrasp: Robust and Efficient Grasp Perception in Spatial and Temporal Domains" | T-RO 2023 |
