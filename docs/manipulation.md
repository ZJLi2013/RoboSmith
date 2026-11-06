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
3. **连接**：关键构型之间关节插值，整条稠密轨迹交给门逐点检查；不过就换下一个候选（抓取、搬运高度）。
4. **失败显式**：任一步不过就结构化失败（阶段、候选、原因、证据）；试下一个候选是显式搜索，不是放行。
5. **状态有唯一主人**：运行时持有实际的关节、物体与持物状态；试抓与正式执行共享同一道门。

### 执行门的接口与契约

门在 `robotsmith/execution/gate.py`。每个概念一个 owner：

| 概念 | Owner |
|---|---|
| Genesis 场景状态：快照、构型、状态指纹 | `SimEnv` |
| 「这条轨迹能不能执行」、持物生命周期与接触许可规则 | `ExecutionGate`（许可规则只在 `permitted` 一处；读实际物理接触的验收用同模块的 `certified`，见 [certified 读数](acceptance.md#被测链路的-certified-读数)） |
| 候选及其顺序（抓取、搬运高度） | 技能层（`run_pick` / `run_place`） |
| 航点模板与每段的持物 / 允许接触 | `MotionExecutor` |
| 关键位姿到稠密关节路径与段边界 | `motion.chain.execute_motion_chain` |
| 抓取的物理是否成立（抬得起） | `SimEnv.trial_from_here`，只在过门之后调用 |
| 执行 | `EpisodeRun`，只回放验证过的轨迹，不解 IK |

接口：

- `ExecutionGate.verify(trajectory, spans, *, target, q_start, rest=())` 返回
  `VerifiedMotion(trajectory, spans, fingerprint, checked)`；任何一个轨迹点违规就抛 `MotionRejected(span, index, pair)`。
  `spans` 是 `Span(label, start, end, holding, allow)`，必须无缝铺满整条轨迹，门检查的是**会被执行的每一个稠密轨迹点**，
  不抽样；性能优化不得减少被检查的点。`ungated(trajectory)` 只给开合抽屉用。
- 段边界来自运动链：每段带 `span` / `holding` / `allow`，段后保持与尾部保持计入所在段；转基座段由技能层作为第一段加入。
  pick 的手指闭合段是 `grip`、其后一段持物且 `rest`；place 的搬运持物，下放与松手持物且 `rest`，退回 `grip`。
- `SimEnv.contacts_at(qpos, held=None, placed=None)`、`SimEnv.hand_pose(qpos)`、`SimEnv.object_pose(name)` 在一个快照事务里
  摆构型（`held=(name, T)` 把物体放在手 · T，`placed=(name, T_world)` 放在世界位姿）、跑碰撞检测或正运动学，返回前还原快照。
  `SimEnv.state_fingerprint()` 是真实写入代次与全部刚体关节位置、速度的哈希：代次由 `SimEnv.step()` 与 `SimEnv.reset()` 递增，
  录制与 settle 窗口都经 `SimEnv.step()` 步进；快照事务内的还原不递增，所以同值 reset 也会让旧验证失效。

契约：

- **持物生命周期**：持物段的被抓物位姿 = 手位姿 · T，T 取自持物开始之前的构型（首个持物点的前一点；轨迹从持物开始时取
  `q_start`）与被抓物当时的位姿。最后一个持物点之后，被抓物留在 手(最后持物点) · T，退回时手掌穿过它会被拒绝。
  被抓物就是场景里那个 entity，几何与执行一致。
- **接触许可**：
  - 手指碰目标，只在持物段或 `grip` 段合法；手的其它 link 碰到目标一律违规。
  - 被抓物碰支撑面或放置目标的 entity（`mat.top` 取 `mat`），只在 `rest` 段合法。
  - 其余凡涉及机器人或被抓物的接触都违规。
  - 第一版不判机器人自接触。
  - 必须拒绝的样例：两个端点合法、但中途穿墙的轨迹。
- **候选搜索**：pick 每个候选先过门，再试抓，抬起后在恢复的状态上对同一条轨迹重新过门；试遍全部候选时（`trial_all`，
  修复工具用），最后一次试抓之后再按候选顺序对抬起者的原轨迹重新过门，第一个通过的被选中；place 按 0、5、10、15、20 cm 的搬运
  高度取最低能过门的一条。任何候选都不成立时抛 `Infeasible(reason, detail)`。
- **状态事务**：只有 `SimEnv` 写 Genesis 状态。快照事务不改变状态指纹；试抓会步进物理，所以试抓之后旧的验证一律作废。
  真实步进与 `reset` 都会改变指纹。
- **失效与零执行**：执行前 `check_current` 比对 `VerifiedMotion.fingerprint` 与当前指纹，不等就抛 `StaleMotion`。
  `verify` 抛出时，没有任何一点进入录制；步骤 `infeasible` 时 `EpisodeRun` 记 `cause: infeasible` 与各候选的拒绝原因，
  不做正式回放（零帧）并停在这一步，入口不做任务判定的 settle，直接判失败。判定之前，过了门的 pick 候选已在
  试抓里步进过物理并恢复快照；只有所有候选都在门前被拒（例如 F4）时，这一步才在初始化之外零物理步。settle 与保持只下发最后一个已验证点。
  初始 reset / settle 属于场景初始化，不在门的范围。
- **并发**：不支持。一个 `SimEnv` 同一时刻只能有一个调用者。

## 已定 / 不做

- rocRobo 与碰撞盲退回已删除，不保留为可选后端；删除后的运动层接口见[调用链](call-chain.md#运动层与执行门的接口)。
  启动脚本的 Docker socket 挂载只为进 rocRobo sidecar 容器，随之删除。下一步自由段改由关节空间 RRT-Connect 生成（未实现），
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
- 入口有界检查：F0、F4 各一个 seed 与单资产抓取验证都只经门执行；F0 完成 pick / place，F4 的 pick 判 `infeasible`、零帧，
  拒绝原因逐候选保留。这不是验收族的 certified 结果，certified 率见下一步。

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
  （留给覆盖动作域的可检查证明，第一版不产出）三者都零物理步，由 `Refused(label, evidence)` 承载，替代现在的
  `Infeasible`；`execution_failed` 保留实际步数。
- 凭证：`gate.verify` 按切点签发 pick part、place part 两个只读 `VerifiedMotion`；任何物理步都改变指纹，旧 part 自然失效。
  pick part 执行后按实测关节角与实测 `T_hand_object` 重验 place part（实测起点偏差 > 0.01 rad 时先接一段 `move_to`），
  不过即 `execution_failed`。

### owner

| 概念 | owner |
|---|---|
| 动作展开 | 各 meta action 实现 |
| 候选搜索与组合内回溯 | `CompositionSearch`（替换 `pick._select` 与 `CARRY_HEIGHTS` 循环） |
| 自由段 | `JointPlanner`（替换 `_transit_lerp`、`_pick_seed_transit`、`_place_transport_seed`） |
| 合法性 | `ExecutionGate.check_point`：数值有效、关节限位、FK 残差（5 mm / 0.05 rad）、非邻接自碰撞、`contacts_at` + `permitted`；规划器与 `verify` 都调它 |
| 持物 | `PlanState.holding`；`gate.verify` 接收它并在 attach 点核对，不再从首个持物点自推 `T_hand_object` |
| 物理步 | `EpisodeRun`，含试抓、回放与 settle |

### 规划器

仓库内 RRT-Connect（双树、固定种子），依赖只有 numpy；状态与连边都经 `check_point`，连边按任一连杆或被抓物上的点每步
位移 ≤ 5 mm 细分；先试直连。每次 connect 至多 1500 次状态查询或 30 s，组合共享 120 s 并按候选轮转。OMPL 算法更成熟，
但 Python 绑定未在 ROCm 镜像上验证，接口留在 `JointPlanner` 可换；VAMP、cuRobo 自带碰撞表示，会形成与仿真几何不一致
的第二份世界，不选。

## 下一步

模板式链路的四步迁移已完成（rocRobo 删除、`SimEnv` 快照接口、执行门、[调用链](call-chain.md)重写），读数见
[certified 读数](acceptance.md#被测链路的-certified-读数)。下一版按以下顺序迁移，每步单独可合入：

1. 门加固：`check_point`、只读凭证、`Refused`、`verify` 接收 `Holding`；
2. `JointPlanner`，独立于技能，GPU 上测 F2 布局的持物 connect；
3. 6 个动作、`CompositionSearch` 与 pick / place recipe 一步切换；`EpisodeRun`、单资产验证与 `certify` 三个入口改用
   同一服务，删除 `MotionExecutor.pick / place`、`run_skills`、`plan_segment` 与转基座 / 抬高补丁；
4. plan 文件的 `actions` 与 LLM 可直接提交 meta action 列表，走同一搜索与门。

然后在组合式验收场景（含开发期不读的留出族）的 test seed 上读 certified 率。
