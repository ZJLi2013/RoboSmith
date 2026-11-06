# 操作：基本 agentic 操作证明任务可解

总体设计见 [system-design](system-design.md)。

## 要保证什么

agent 写步骤表（pick X、place X on Y…），技能执行，每步判定。**任务判定成立，且实际执行的每个动作都事先在仿真几何里验证过，才算证明可解。**
不可行就带原因失败，不靠物理推挤蒙混。同一套操作也是第一个被测方（基线）。

## 现在有什么

- 步骤层：agent 写步骤表，失败按计划层 / 物理层分流，同一 episode 内重新规划。
- 抓取：由物体几何生成候选，在仿真里试抓、按结果选；全部失败时 agent 修复环归因、补候选、复核后写回资产。
- 放置：在资产标注的目标面上放；抽屉开合。
- 判定：每步与整个任务共用同一组谓词，读仿真状态，可离线复判。
- 执行：技能、试抓、单资产验证只执行过了[执行门](#执行门的接口与契约)的轨迹；开合抽屉标 `ungated`。

## 迁移前的问题

执行门接入（下一步的第 3 步）之前，候选、规划、执行之间**没有共同且强制的合法性契约**：

- 没有统一的执行前检查：试抓只看能否抬起；自由段是不查碰撞的关节插值；接触段只查关节跳变；转基座、手指开合、修复工具、
  单资产验证都不经碰撞检查。
- 已删除的规划器（rocRobo）在自己的简化世界里规划，与仿真几何不一致，所测的一个可行位姿它也判失败；规划失败时退回不检查的
  直线，照常执行、照常判成功。

## 设计：一道执行门

所有执行入口（技能、试抓、修复、单资产验证）都只执行**经同一道门验证过的路径**：

1. **候选**：由资产几何生成抓取 / 放置 / 把手候选。
2. **验证**：在仿真场景里查 IK 能否到达、最终路径上有无不允许的接触（带被抓物）；接触段显式列出允许接触的对象。
3. **连接**：place 的搬运段由 `JointPlanner` 经门规划，其余关键构型之间关节插值；整条稠密轨迹交给门逐点检查，
   pick 不过就换下一个抓取候选，place 不过就零步拒绝。
4. **失败显式**：任一步不过就结构化失败（阶段、候选、原因、证据）；试下一个候选是显式搜索，不是放行。
5. **状态有唯一主人**：运行时持有实际的关节、物体与持物状态；试抓与正式执行共享同一道门。

### 执行门的接口与契约

门在 `robotsmith/execution/gate.py`。每个概念一个 owner：

| 概念 | Owner |
|---|---|
| Genesis 场景状态：快照、构型、状态指纹 | `SimEnv` |
| 「这条轨迹能不能执行」、持物生命周期与接触许可规则 | `ExecutionGate`（许可规则只在 `permitted` 一处；读实际物理接触的验收用同模块的 `certified`，见 [certified 读数](acceptance.md#被测链路的-certified-读数)） |
| 候选及其顺序（抓取）、place 的搬运目标 | 技能层（`run_pick` / `run_place`） |
| 航点模板与每段的持物 / 允许接触 | `MotionExecutor` |
| 关键位姿到稠密关节路径与段边界 | `motion.chain.execute_motion_chain` |
| 抓取的物理是否成立（抬得起） | `SimEnv.trial_from_here`，只在过门之后调用 |
| 执行 | `EpisodeRun`，只回放验证过的轨迹，不解 IK |

接口：

- `ExecutionGate.check_point(q, span, *, target, rest=(), held=None, placed=None, pose=None)` 是唯一的单点合法性检查，
  合法返回 `None`，否则返回 `{"reason", 证据}`：`invalid_value`（非有限值或维数不对）、`joint_limit`（超出
  `SimEnv.joint_limits`）、`fk_residual`（带 `pose` 时手部位姿与它相差超过 5 mm / 0.05 rad）、`self_collision`
  （机器人两连杆相碰且不在 `SimEnv.joined_links`，即非父子、非同父）、`collision`（其余接触不被 `permitted` 许可）。
  只读，返回或抛出时场景不变。后续的规划器与 `verify` 用同一个它。
- `ExecutionGate.verify(trajectory, spans, *, target, q_start, rest=(), holding=None, poses=None)` 对每个点调
  `check_point`，第一个不过的点抛 `MotionRejected(span, index, rejection)`（`rejection` 即上面的字典，接触对在 `.pair`）。
  `spans` 是 `Span(label, start, end, holding, allow)`，必须无缝铺满整条轨迹，门检查的是**会被执行的每一个稠密轨迹点**，
  不抽样；性能优化不得减少被检查的点。`poses` 是 `{轨迹下标: (pos, quat)}`，只含要到达某个手部位姿的点：运动链给每段记下
  航点位姿 `pose` 与到达它的下标 `reach`，`gate.poses_of(segments, offset)` 换算出来；自由插值的中间点、transit 与保持点
  不带期望位姿，所以不会因偏离终点被拒。`ungated(trajectory)` 只给开合抽屉用。
- 通过时返回只读的 `VerifiedMotion(trajectory, spans, fingerprint, checked, gated, holding)`；只有签发它的门的
  `check_current` 接受它，手工构造或别的门签发的凭证一律 `StaleMotion`。
- 凭证内容不可改：`VerifiedMotion` 与 `Holding` 只持有轨迹点和持物变换的 float64 `bytes` 副本与 shape，
  `trajectory` / `t_hand_object` 每次读取都从 `bytes` 构造新的视图。底层是只读缓冲区的数组不能用 `setflags(write=True)`
  重开写入（[NumPy `setflags`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.setflags.html)：只有自有内存
  或最终 owner 暴露可写缓冲区时才允许），视图、切片、memoryview 同样被拒；改读到数组的 `dtype` / `shape`
  （NumPy 2.5 起已弃用）只改调用方手里那份，下一次读取与回放仍是验证时的数值。调用方保留的输入数组与凭证不共享内存。
  另一种做法是签发时记内容哈希、`check_current` 时比对，它只能在执行前发现改动；不可变存储让改动无从发生，所以选它。
  限制：对 frozen dataclass 调 `object.__setattr__`、经 ctypes 按裸指针写内存能绕过 Python 层的保证，门不防这类刻意绕过。
- 段边界来自运动链：每段带 `span` / `holding` / `allow`，段后保持与尾部保持计入所在段；pick 的转基座段与 place 的规划搬运段由技能层作为第一段加入。
  pick 的手指闭合段是 `grip`、其后一段持物且 `rest`；place 的搬运持物，下放与松手持物且 `rest`，退回 `grip`。
- `SimEnv.contacts_at(qpos, held=None, placed=None)`、`SimEnv.hand_pose(qpos)`、`SimEnv.object_pose(name)` 在一个快照事务里
  摆构型（`held=(name, T)` 把物体放在手 · T，`placed=(name, T_world)` 放在世界位姿）、跑碰撞检测或正运动学，返回前还原快照。
  `SimEnv.state_fingerprint()` 是真实写入代次与全部刚体关节位置、速度的哈希：代次由 `SimEnv.step()` 与 `SimEnv.reset()` 递增，
  录制与 settle 窗口都经 `SimEnv.step()` 步进；快照事务内的还原不递增，所以同值 reset 也会让旧验证失效。

契约：

- **持物生命周期**：持物是调用方给的 `Holding(target, t_hand_object)`，有持物段而没有它时 `verify` 报错。pick 由抓取候选的
  手部位姿与被抓物当前位姿构造 T，place 由实测手部位姿与被抓物位姿构造。attach 处（首个持物点的前一点；轨迹从持物开始时取
  `q_start`）手 · T 与被抓物位姿相差超过 5 mm / 0.05 rad 时拒绝，记 `binding_mismatch`。持物段被抓物位姿 = 手位姿 · T；
  最后一个持物点之后，被抓物留在 手(最后持物点) · T，退回时手掌穿过它会被拒绝。被抓物就是场景里那个 entity，几何与执行一致。
- **接触许可**：
  - 手指碰目标，只在持物段或 `grip` 段合法；手的其它 link 碰到目标一律违规。
  - 被抓物碰支撑面或放置目标的 entity（`mat.top` 取 `mat`），只在 `rest` 段合法。
  - 其余凡涉及机器人或被抓物的接触都违规。
  - 第一版不判机器人自接触。
  - 必须拒绝的样例：两个端点合法、但中途穿墙的轨迹。
- **候选搜索**：pick 每个候选先过门，再试抓，抬起后在恢复的状态上对同一条轨迹重新过门；试遍全部候选时（`trial_all`，
  修复工具用），最后一次试抓之后再按候选顺序对抬起者的原轨迹重新过门，第一个通过的被选中；place 按 0、5、10、15、20 cm 的搬运
  高度取最低能过门的一条。没有候选在任何试抓之前过门时抛 `Refused("search_exhausted", evidence)`（零物理步，不证明不可解）；
  已有候选过门并试抓过、却没有一个成立时抛 `ExecutionFailed(evidence, steps)`，`steps` 是试抓步进的物理步数。
  `Refused` 只接受 `search_exhausted`、`unsupported`、`infeasible` 三个标签，`infeasible` 留给覆盖动作域的可检查证明，
  现在没有任何路径产出它。
- **状态事务**：只有 `SimEnv` 写 Genesis 状态。快照事务不改变状态指纹；试抓会步进物理，所以试抓之后旧的验证一律作废。
  真实步进与 `reset` 都会改变指纹。
- **失效与零执行**：执行前 `check_current` 比对 `VerifiedMotion.fingerprint` 与当前指纹，不等就抛 `StaleMotion`。
  `verify` 抛出时，没有任何一点进入录制；步骤抛 `Refused` 或 `ExecutionFailed` 时 `EpisodeRun` 记 `cause`（标签）、
  `evidence`（各候选的拒绝原因）与 `steps`，存进 `stopped`，不做正式回放（零帧）并停在这一步，入口不做任务判定的
  settle，直接判失败。`search_exhausted` 表示所有候选都在门前被拒（例如 F4），这一步在初始化之外零物理步；
  `execution_failed` 表示过门的候选已在试抓里步进过物理并恢复快照。settle 与保持只下发最后一个已验证点。
  初始 reset / settle 属于场景初始化，不在门的范围。
- **并发**：不支持。一个 `SimEnv` 同一时刻只能有一个调用者。

## 已定 / 不做

- rocRobo 与碰撞盲退回已删除，不保留为可选后端；删除后的运动层接口见[调用链](call-chain.md#运动层与执行门的接口)。
  启动脚本的 Docker socket 挂载只为进 rocRobo sidecar 容器，随之删除。自由段改由关节空间 RRT-Connect（`JointPlanner`，已实现、技能未接入）生成，
  状态与连边的合法性检查调用执行门同一份 `contacts_at` + `permitted`，不另建世界：「候选 × 关节插值」在 F2 上 0/20，
  失败全在写死的搬运路线（[读数](acceptance.md#被测链路的-certified-读数)）。rocRobo 的问题是它在自己的简化世界里规划，
  与仿真几何不一致（见「迁移前的问题」）。
- 关节体操作（开合抽屉）第一版不过门，标为 `ungated`；验收族里没有抽屉。
- 中间层用 Python 自建，不接 MoveIt（ROS / C++，跑不到 Genesis 上）；第一版不做通用框架、多后端、自动修复写回、扩原语、调参。
- 抓取不用学习模型；夹爪为位置控制，约 0.2 kg 以上的物体会滑脱。

## 实现须知

- Genesis 里判某个构型有无接触，要在设定关节后直接跑碰撞检测，不能先走一步仿真（一步内求解器会把手推开，读成无接触）。
- Panda 手指沿 hand 局部 Y 闭合；夹爪可用开口至少 7.5 cm。
- 同一个末端位姿换 IK 初值，接触结果会变：验证过的关节构型要带到执行，执行时不再重新解 IK。

- `SimEnv` 的三个查询已在 Genesis 0.4.5（MI300X、F1 场景）上实测，测试在 `tests/test_sim_queries_gpu.py`：
  - `set_dofs_position` 摆好构型后，`rigid_solver.detect_collision()` 直接反映新构型；交替查询相交 / 分离构型不沿用上一次结果。
    持物按 `set_pos` / `set_quat` 摆到手位姿 · T 后同样即时生效。
  - 快照事务 `scene.get_state()` / `scene.reset(state)` 还原后指纹不变，查询抛异常时也还原。指纹带 `SimEnv` 的真实写入代次，
    这样即使 `reset` 后状态与之前完全相同，旧验证也会失效。
  - 单点耗时在 10 ms 量级，几百个稠密点的轨迹过一次门要几秒；第一版接受，不抽样（读数在实验记录里）。
- 执行门在同一环境的真实场景上满足所列预期，测试在 `tests/test_gate_gpu.py`：两端合法、中途碰墙的轨迹在中间点被拒；
  持物进墙以被抓物–墙被拒、越过墙上方可过门，手 · T 摆出的被抓物与预期位置相差不到 1 mm；松手后手掌下到被抓物处以
  机器人–被抓物被拒、向上退回可过门；真实步进后旧凭证失效。
- 入口有界检查：F0、F4 各一个 seed 与单资产抓取验证都只经门执行；F0 完成 pick / place，F4 的 pick 在门前全部被拒、零帧，
  拒绝原因逐候选保留。这不是验收族的 certified 结果，certified 率见下一步。
- `check_point` 新增的关节限位、FK 残差、非邻接自碰撞与 attach 核对，以及 `SimEnv.joint_limits`（Genesis
  `get_dofs_limit`）、`joined_links`，在 CPU 受控样例上验证过（[e1 证据](../exps/r1/evidence/e1/README.md)）；在 Genesis 上
  只跑过 e3 的一个 F2 持物请求（两端带 FK 残差检查、对照路径的 attach 核对都通过）。IK 残差超过 5 mm 的抓取候选现在会以
  `fk_residual` 或 `binding_mismatch` 被拒，它对验收族读数的影响要等下一次 GPU 批次读出。`tests/test_gate_gpu.py` 已改为显式传 `Holding`。

## 下一版设计：规划给出的 meta action 组合（未实现）

上面的执行门已实现；本节是替换 `MotionExecutor` 航点模板的设计，迁移完成前现有实现仍按上文。原因：模板式链路在
F2 上 0/20，失败全在写死的搬运路线（[读数](acceptance.md#被测链路的-certified-读数)），给路线打补丁只适配一个场景。

**一句话：** 技能是少量 meta action 的组合；组合搜索在零物理步的状态查询里为整个组合绑定候选、用 RRT-Connect 连自由段、
整条时间线过一次执行门，拿到凭证才走第一个物理步。分工对照 MoveIt Task Constructor 的 Generate / Connect /
MoveRelative / ModifyPlanningScene。

### meta action

| 动作 | 运动由谁生成 | 前置 → 后置 | 附加失败标签 |
|---|---|---|---|
| `sample` | 几何采样 + IK（`GraspPlanner.plan / plan_place`），零运动 | 产出带多个 IK 解的抓取 / 放置候选绑定 | `no_candidate`、`ik_failed` |
| `move_to` | 关节空间 RRT-Connect | 持物状态不变，到目标构型 | `invalid_start`、`invalid_goal`、`search_exhausted` |
| `move_linear` | 笛卡尔直线逐点 IK，跳变即拒 | 持物状态不变，末端沿直线到达 | `ik_failed`、`discontinuity` |
| `gripper` | 手指位置目标，`allow=grip` | 张开到全宽；合爪到手指被物体挡住，宽度在候选 `closing_width` 容差内 | `grasp_unconfirmed` |
| `attach` | 状态事件，零运动 | 持物 = 刚体（`T_hand_object` 取自抓取绑定） | `binding_mismatch` |
| `detach` | 状态事件，零运动 | 持物为空，物体固定在放置位姿 | `unsupported_release` |

所有动作共有 `precondition_failed`、`collision`（门拒绝，带接触对）、`stale_state`。`push`（推移模型 + 推接触许可）与
`articulate`（把手约束 + 预测关节值）是扩展样例，需要门协议升级，第一版不实现；开合抽屉在那之前沿用 `drag_handle` +
`ungated`，不计入产品保证。

`pick`、`place` 是预定义组合：

```text
pick:  sample(grasp) · gripper(open) · move_to(g.pre) · move_linear(g.grasp) · gripper(close, target) · attach(g) · move_linear(+z)
place: sample(place | g) · move_to(p.pre) · move_linear(p.place) · detach(p.support) · gripper(open, target) · move_linear(p.retreat)
```

接触许可沿用 `Span`：合爪 / 松爪 `allow=grip`；抬起持物且 `rest`（只许离开起始支撑）；搬运持物、无许可；落座持物且
`rest`（只许碰放置支撑）；退回时目标固定在放置位姿。放置手位姿 = `T_world_object_goal · inv(T_hand_object)`，取代
`held_place_z`。一个组合至多一次 attach → detach。

扩展规则：新增动作 = 新模块 + 注册，实现参数 schema、零物理步的 `expand(state, args, budget)`（产出轨迹段 + span +
预测末态）与结构化失败；不调 `SimEnv.step`、不构造 `VerifiedMotion`。需要新许可、新持物类型或预测状态的动作属于
门协议升级。

### 提交边界与结果

- 同一目标的 pick + place 是一个组合：全部段在第一个物理步之前生成并一次过门，place 不可达时 pick 也不执行。
- 试抓只对已过门的组合做、只回放 pick part，由 `EpisodeRun` 发起；第一次试抓即提交，此后结果只有成功或
  `execution_failed`。门拒绝发生在任何试抓之前，所以 F4 仍是 reset 后零物理步。
- 结果：`search_exhausted`（声明的候选与预算内没有组合过门，含 F4，不证明不可解）、`unsupported`、`infeasible`
  （留给覆盖动作域的可检查证明，第一版不产出）三者都零物理步，由 `Refused(label, evidence)` 承载；`execution_failed`
  由 `ExecutionFailed(evidence, steps)` 承载，保留实际步数。这两个类型已实现，现有 pick / place 已经使用它们。
- 凭证：`gate.verify` 按切点签发 pick part、place part 两个只读 `VerifiedMotion`；任何物理步都改变指纹，旧 part 自然失效。
  pick part 执行后按实测关节角与实测 `T_hand_object` 重验 place part（实测起点偏差 > 0.01 rad 时先接一段 `move_to`），
  不过即 `execution_failed`。

### owner

| 概念 | owner |
|---|---|
| 动作展开 | 各 meta action 实现 |
| 候选搜索与组合内回溯 | `CompositionSearch`（替换 `pick._select`） |
| 自由段 | `JointPlanner`（已实现；place 的搬运段与 pick 的进场段已接入，替换了转基座、搬运高度与进场插值） |
| 合法性 | `ExecutionGate.check_point`：数值有效、关节限位、FK 残差（5 mm / 0.05 rad）、非邻接自碰撞、`contacts_at` + `permitted`；规划器与 `verify` 都调它（已实现，`verify` 已在用） |
| 持物 | `PlanState.holding`；`gate.verify` 接收 `Holding` 并在 attach 点核对（`verify` 一侧已实现，现由 pick / place 构造；`PlanState` 未实现） |
| 物理步 | `EpisodeRun`，含试抓、回放与 settle |

### 规划器

`robotsmith/motion/joint_planner.py` 的 `JointPlanner(env, gate, *, seed=0).connect(q_start, q_goal, *, target, rest=(),
holding=None, label="move_to")` 已实现，place 用它规划搬运段，pick 还没接入。它是仓库内的 lazy RRT-Connect（双树、固定种子），依赖只有 numpy：

- 合法性只问 `gate.check_point`，持物带调用方的 `Holding`；路径由 `gate.verify` 签发，返回
  `Connection(motion, queries, seconds, nodes, direct)`。查询与失败都零物理步、不改状态。
- 连边细分：`SimEnv.lever_arms(held)` 给每个关节的力臂上界 r（从关节锚点沿连杆原点到它带动的任一连杆碰撞 AABB、
  手指行程、持物 AABB 的最远距离），每步 Σ r_k |Δq_k| ≤ 5 mm，所以任一连杆或持物上的点每步位移不超过 5 mm。
  边内的点按二分顺序检查，碰撞早遇早停。
- 先试直连；不通则双树生长，新节点只查状态。两树相遇后从每个节点直接连到能连通的最远后继（捷径），树边失败就剪掉
  它下面的子树再长。
- 预算：每次 connect 至多 1500 次状态查询或 30 s，覆盖搜索与最终 `verify` 的逐点签发；`verify(..., budget=)` 在每点
  检查前扣一次查询、检查后与返回凭证前核对截止时间。返回的 `queries` 就是门实际检查的点数，重验同一点也计数。用尽或
  超时即 `Refused("search_exhausted", {"reason": "budget", "queries", "seconds"})`，不签发、零物理步；恰好用尽且按时签发
  完成算成功。起止点非法为 `invalid_start` / `invalid_goal`，带门的拒绝原因。组合共享 120 s 并按候选轮转（未实现）。
  做法同 MoveIt：`allowed_planning_time` 是规划与路径简化的总预算，简化只拿搜索剩下的时间（MoveIt 2 OMPL 接口
  [`model_based_planning_context.cpp`](https://github.com/moveit/moveit2/blob/main/moveit_planners/ompl/ompl_interface/src/model_based_planning_context.cpp)
  以 `allowed_planning_time - ptime` 调 `simplifySolution`，插值不计时）；本项目的签发就是对插值后
  路径的逐点重验，凭证是 connect 的产物，所以也算在内。两者的截止检查都在查询之间：正在运行的单点查询打断不了，
  仿真挂住由 `batch.supervise` 在进程外强杀并记 `error`。
- 起态装配：`SimEnv.hold(qpos, name, T)` 把机器人放到 qpos、物体放到手 · T，算一次真实写入，不步进。

F2 playground seed 0 的固定持物请求（retreat 位姿 → mat 正上方同高）在 Genesis 上一次连通：两端过门、直连在第 88/203
点以左手指碰 barrier 被拒、独立的抬高越障对照过门；`connect` 用 301 次查询、10.9 s、19 个节点，签发 247 点，最大每步
位移 4.995 mm，零物理步（[e3 证据](../exps/r1/evidence/e3/README.md)）。这是单个请求的持物 connect，不是任务级 certified。
那次的 301 次不含签发的 247 次检查；按全程预算计是 548 次，10.9 s 本就含签发，仍在预算内（[e4 证据](../exps/r1/evidence/e4/README.md)，
未重跑 GPU）。

限制：均匀采样的 7 维路径会绕远，CPU 受控样例里绕一道 0.2 rad 宽的墙要 0.6k–4k 次查询；比扩展步长（0.1 rad）还薄的
障碍会让两树隔着它反复相遇、剪枝，同类薄墙要 7k–160k 次，超出预算。OMPL 算法更成熟，但 Python 绑定未在 ROCm 镜像上
验证，接口留在 `JointPlanner` 可换；VAMP、cuRobo 自带碰撞表示，会形成与仿真几何不一致的第二份世界，不选。

### place 的规划搬运

`run_place` 用 IK 解 `plan_place` 的 standoff（以当前关节为初值，手指取当前值），`JointPlanner.connect` 带持物从当前构型
连到它，再由 `MotionExecutor.place(phase="descend")` 接下放、松手、退回；整条轨迹一次 `verify`、一张凭证。connect 被拒时
place 零步 `Refused("search_exhausted")`，证据带 connect 的 reason 与 queries。转基座段与搬运高度已删除。

在 playground seed 0–9 上与[逐片迁移的 dev 对照](acceptance.md#逐片迁移的-dev-对照)比较（[e7 证据](../exps/r1/evidence/e7/README.md)）：

| 族 | certified | 对照 |
|---|---|---|
| F0 空桌面 | 10/10 | 10/10 |
| F1 贴墙抓取 | 10/10 | 10/10 |
| F2 越障搬运 | 7/10 | 0/10 |
| F4 罩住 | 10/10 | 10/10 |

F2 的 10 个搬运全部连通。未 certified 的 3 个（seed 1、3、8）任务谓词成立，但正式回放的松手段里机器人 `link5` 实际碰到
barrier 5–8 个物理步，而门在签发时判这些点合法。这是已知问题，成因未定位（human 2026-11-06 决定不继续定位，先推进后续切片）。单实例耗时约为对照的两倍（F0 平均 85 s，对照约 50 s）：connect 约 8–10 s，
规划路径按每 5 mm 一个物理步执行，比原先 40 步的插值搬运长得多。

### pick 的规划进场

`_pick_request` 用 IK 解候选的首航点（初值取 home 转基座，或候选的 `ik_seed`，手指取当前值），`MotionExecutor.pick` 从那里起：
首航点锁臂，只在需要时张指，所以从当前构型到下探起点的手臂运动全部来自 `JointPlanner.connect`。抓取段先单独过门，被拒的
候选不花规划；connect 被拒记进该候选的试抓记录，换下一个候选。`_pick_seed_transit` 已删除。

在 playground seed 0–9 上逐实例与 place 的规划搬运（S1）相同（[e9 证据](../exps/r1/evidence/e9/README.md)）：F0 10/10、F1 10/10、
F2 7/10（另 3 个仍是松手段的已知接触）、F4 10/10 零步拒绝；die_01 单资产抓取 3/3，与之前相同。单实例耗时比 S1 长约 45%
（F0 平均 125 s）。

## 下一步

模板式链路的四步迁移已完成（rocRobo 删除、`SimEnv` 快照接口、执行门、[调用链](call-chain.md)重写），读数见
[certified 读数](acceptance.md#被测链路的-certified-读数)。下一版按以下顺序迁移，每步单独可合入：

1. 门加固：`check_point`、只读凭证、`Refused`、`verify` 接收 `Holding`（已实现，CPU 验证）；
2. `JointPlanner`，独立于技能，GPU 上测 F2 布局的持物 connect（已实现，F2 固定请求一次连通，见「规划器」）；
3. 模板链路逐片替换，每片只改一件事，在 playground seed 0–9 上与[逐片迁移的 dev 对照](acceptance.md#逐片迁移的-dev-对照)比较：
   1. place 的搬运段由 `JointPlanner` 生成，删转基座与搬运高度（已实现，结果见「place 的规划搬运」）；
   2. pick 的进场段由 `JointPlanner` 生成，删 `_pick_seed_transit`（已实现，结果见「pick 的规划进场」）；
   3. 同一目标的 pick + place 在第一个物理步前整组过门；
   4. `move_linear`、`gripper`、`attach`、`detach` 从 `MotionExecutor` 模板拆出，一次一个；
   5. `EpisodeRun`、单资产验证与 `certify` 走同一服务，删 `run_skills`、`plan_segment`；
4. plan 文件的 `actions` 与 LLM 可直接提交 meta action 列表，走同一搜索与门。

然后在组合式验收场景（含开发期不读的留出族）的 test seed 上读 certified 率。
