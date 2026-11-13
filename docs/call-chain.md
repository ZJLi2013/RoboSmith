# 调用链：一次 scenario episode 怎么跑

开发文档，画 `main` 上的现状，粒度到函数；设计（要保证什么）在 [design](design.md) 与 [manipulation](manipulation.md)。改了这条链的 PR 同时改这里。

**每个被执行的关节目标都先经同一道执行门**：技能生成整条稠密轨迹，`ExecutionGate.verify` 对每个点调用
`check_point`（数值、关节限位、要到达的位姿、自碰撞与接触，接触在 Genesis 快照里查），`EpisodeRun` 回放前核对凭证是这道门
签发的、状态指纹没变，只回放签发过的原轨迹。门的接口与接触契约见
[manipulation](manipulation.md#执行门的接口与契约)。

## 四层

| 层 | 回答什么 | 组件（模块） |
|---|---|---|
| 编排层 | 做哪几步、失败后怎么办 | `scenarios/plans/`、`scenario_runtime/planner.py`、`scenario_runtime/runner.py` 的 `EpisodeRun` |
| 技能层 | 一步（pick / place / open / close）怎么变成一条过门的关节轨迹 | `skills/` |
| 底层：抓取与运动 | 抓哪里、航点是什么、每段怎么解出关节、哪个点合法 | `grasp/`、`execution/`、`motion/` |
| 仿真与判定 | 执行、录帧、判这一步和整个任务成没成 | `sim/`、`scenario_runtime/handoff.py`、`tasks/`；判定函数住在 `runner.py` |

每个框写 `模块.函数`，框按所属模块放进层里；同一层内的子框是可以单独替换的组件。

## 对象：一步 pick / place 由谁完成

```mermaid
classDiagram
    class EpisodeRun {
        +run_steps(skills, bindings)
        +result()
    }
    class SkillContext {
        skill
        scene_state
        held
        held_place_z
        phase
    }
    class SkillDef {
        primitive
        post
        settle_s
    }
    class RulePolicy {
        +pick(ctx, tools, place_follows, place_next, trial, try_all)
        +place(ctx, tools)
    }
    class ManipulationTools {
        +grasp_candidates(ctx)
        +plan_pick(ctx, plan, params)
        +verify(request)
        +trial(trajectory, target)
        +posed(qpos, objects)
        +plan_place(ctx)
    }
    class GraspPlanner {
        <<abstract>>
        +plan()
        +plan_place()
    }
    class GeometricGraspPlanner {
        +candidates(world_points)
    }
    class ExecutionGate {
        +verify(trajectory, spans)
        +check_point(q, span)
        +check_current(motion)
        +ungated(trajectory)
    }
    class JointPlanner {
        +connect(q_start, q_goal)
    }
    class MotionExecutor {
        +run(name, actions)
        +drag_handle()
    }
    class SimEnv {
        +contacts_at(qpos)
        +posed(qpos, objects)
        +trial_from_here(trajectory, target)
    }
    class VerifiedMotion

    EpisodeRun --> SimEnv
    EpisodeRun *-- ExecutionGate
    EpisodeRun --> MotionExecutor
    EpisodeRun *-- ManipulationTools
    EpisodeRun *-- RulePolicy
    EpisodeRun ..> SkillContext : 每个 phase 构造
    EpisodeRun ..> SkillDef : 查 SKILL_DEFS
    SkillDef ..> open_close : primitive
    RulePolicy ..> ManipulationTools : 调用
    ManipulationTools --> ExecutionGate
    ManipulationTools --> GraspPlanner
    ManipulationTools --> MotionExecutor
    ManipulationTools ..> JointPlanner : 每次规划构造
    SkillContext --> ExecutionGate
    SkillContext --> MotionExecutor
    GraspPlanner <|-- GeometricGraspPlanner
    SimEnv *-- GeometricGraspPlanner : build()
    ExecutionGate --> SimEnv
    JointPlanner --> ExecutionGate
    ExecutionGate ..> VerifiedMotion : 签发
```

实线箭头 = 持有（注入），菱形 = 自己创建，空心三角 = 继承，虚线 = 构造后交出或调用。`EpisodeRun.__init__` 创建
`ExecutionGate`、`ManipulationTools`（注入门、`env.planner`、执行器与 `GenesisBackend(env.solve_ik)`）与 `RulePolicy`；`env`、`executor` 由入口传入；
`SimEnv.build` 把 `GeometricGraspPlanner` 存为 `planner`。`SKILL_DEFS` 里 pick / place 只带完成判据与 `settle_s`，规划由策略做；
open / close 仍走自己的 `primitive`。

- `ManipulationTools` 是 pick / place 里调用抓取规划器、`gate.verify`、`JointPlanner` 与 `trial_from_here` 的唯一一处，
  每次调用执行一个显式请求，不选候选、不重试。门与试抓都在调用时从 `gate` / `env` 上取，certify 包在它们外面的监听照样生效。
- `RulePolicy` 只做选择：候选顺序、place 预检、试抓与试抓后重签。
- 跨步依赖由 `runner.plan_skill` 显式传给策略：下一步是不是 place（`place_follows`，决定抬升后是否保持）、
  它是不是单阶段 place（`place_next`，决定预检哪个 place）、试抓（`trial`，规划器是 `GeometricGraspPlanner` 时为 `tools.trial`）
  与 `try_all`。pick 返回的 `held_place_z` 由 `plan_skill` 交回 `EpisodeRun` 存下，下一步 place 从 `SkillContext` 读。

## episode：从入口到判定

```mermaid
flowchart TD
    CLI["scripts/datagen/run_generated_scenario.py<br/>main → load_scenario → run_scenario_smoke"]

    subgraph ORCH["编排：做哪几步"]
        T["--planner file（默认）<br/>scenarios/plans/ 下同名 YAML 的 steps<br/>runner.run_scenario_episode"]
        A["--planner agent --role certifier|baseline<br/>planner.run_agentic_episode<br/>render_request → call_llm → parse_steps"]
        RF["planner.route_failure<br/>plan 类失败（含零步拒绝）重写剩余步骤，最多 MAX_REPLANS = 2 次"]
        RUN["runner.EpisodeRun.run_steps<br/>每步按 skill_phases 分段；每段先 _live_scene_state 重读仿真"]
    end

    subgraph SKILL["技能：runner.plan_skill 分发，返回 VerifiedMotion"]
        PICK["rule_policy.RulePolicy.pick"]
        PLACE["rule_policy.RulePolicy.place"]
        OC["open_close.run_open / run_close<br/>gate.ungated"]
    end

    INF["Refused（search_exhausted：全在门前被拒，零物理步）<br/>或 ExecutionFailed（过门候选试抓后都没抬起，带 steps）<br/>pick 已执行后 place 被拒：记 execution_failed，evidence.refused 带原标签，steps 从 pick 起算<br/>EpisodeRun.stopped 记 cause、evidence、steps；零帧，停在这一步"]
    CUR["ExecutionGate.check_current<br/>不是本门签发、或凭证指纹 ≠ 当前指纹就抛 StaleMotion"]

    subgraph SIM["仿真与判定"]
        REC["sim/recorder.record_episode<br/>逐点下发 VerifiedMotion.trajectory 并录帧"]
        HO["handoff.check<br/>读手、持物、关节的逐帧采样，只记录"]
        JC["runner._judge_completion<br/>settle_window 保持末个已验证点 settle_s 后判这一步绑定的谓词"]
        TASK["runner.evaluate_scenario_task<br/>复用末步的窗口，否则静置 0.5 s；判任务的成功树"]
    end

    CLI --> T & A
    T --> RUN
    A --> RUN
    RUN -->|stop_on_fail 时返回失败的步| RF
    RF -->|route = plan| A
    RUN --> PICK & PLACE & OC
    PICK & PLACE -->|全部候选被拒| INF
    PICK & PLACE & OC -->|VerifiedMotion| CUR --> REC
    REC --> HO --> JC -->|下一段 / 下一步| RUN
    RUN -->|全部步骤结束| TASK
    INF -->|入口不做任务判定的 settle，直接判失败| CLI

    style ORCH fill:#e3f1e0,stroke:#2e7d32
    style SKILL fill:#e8eef9,stroke:#3b5b9a
    style SIM fill:#f2f2f2,stroke:#666
```

技能内部见下一张图。file 与 agent 两条路径共用 `EpisodeRun`，只是步骤表的来源不同：plan 文件的 `steps`，或 agent 回复里的
`steps`，都经 `scenario/load.steps_to_skills` 变成技能，pick 的 `lift` 进 `Skill.params["lift"]`。物体或放置目标不在场景里、pick 时手里
已有物体、place 时手里为空（agent 续跑从 `EpisodeRun.held` 推起）即加载时 `PlanInvalid`，消息以 `step <i>:` 开头
（[写法与报错](../scenarios/AGENT_PROMPT.md#plan-file-optional)）；关系求解的 `SolveError` 只在这个入口转成 `PlanInvalid`（保留为 cause），
YAML 与 schema 错误原样抛出。agent 路径的 `parse_steps` 把它包成 `PlanError`、消息不变，作为可修正的原因回给 agent。pick 已走过物理步之后组合即提交，place 仍从实测状态重新规划，这时它被拒就记 `execution_failed`。agent 路径上零步拒绝（`Refused` 的标签）按
plan 类失败重写剩余步骤，`execution_failed` 按 physical 失败上报；重写后的步骤全部执行完，episode 就不再算被拒。

### 一步的数据表示：从 plan 文件到凭证

```mermaid
flowchart LR
    Y["plan YAML steps"] -->|steps_to_skills| S["Skill<br/>params lift"]
    S -->|EpisodeRun._plan → plan_skill| C["SkillContext"]
    C -->|pick_actions / place_actions<br/>候选与抬升| A["RuntimeMotionWaypoint<br/>与 Attach / Detach"]
    A -->|execute_motion_chain| M["MotionChainExecution<br/>segments"]
    M -->|spans_of / poses_of| P["Span"]
    P -->|ExecutionGate.verify| V["VerifiedMotion"]
```

动作列表的顺序在 `meta_actions` 里固定为一种，步骤能换的是物体、放置目标与 pick 抬升的方向和距离；五个验收族的
`steps` 相同。plan 文件与 agent 回复都经 `load.steps_to_skills` 直接解析成 `Skill`。

### 一个场景的数据表示

一个场景在一个 seed 下的全部运行时输入是一个 `scenario.load.Scenario`，由 `load_scenario(path, *, seed, agent_plan=False)`
一次建成；每个概念只有一种类型、一个 owner。

```mermaid
classDiagram
    class Scenario {
        source_path
        scene
        task
        target_frames
    }
    class SceneConfig
    class ObjectPlacement
    class FrameRef
    class TaskSpec
    class SuccessNode
    class Skill
    Scenario --> SceneConfig : scene
    Scenario --> TaskSpec : task
    Scenario --> FrameRef : target_frames（plan 模式为步骤用到的，agent 模式为全部 place 目标）
    SceneConfig *-- ObjectPlacement
    TaskSpec *-- SuccessNode
    TaskSpec *-- Skill : skills（plan 模式为 plan 文件的步骤，agent 模式为空）
```

| 概念 | 类型 | 建它的 |
|---|---|---|
| 场景（物体、桌子、相机） | `scenes.config.SceneConfig` / `ObjectPlacement` | `load_scenario`（关系求解后的坐标）；单资产试抓入口 `asset_pick_validation._make_scene` |
| place 目标 | `Scenario.target_frames: dict[str, FrameRef]` | `load.place_target_frame` / `all_place_target_frames` |
| 任务与成功树 | `tasks.task_spec.TaskSpec` / `SuccessNode` | `load_scenario`（成功树由 `scenario.predicates` 的 builder 产出） |
| 一步 | `skills.base.Skill` | `load.steps_to_skills`（plan 文件与 agent 回复共用） |
| 步骤的完成条件 | `TaskSpec.step_bindings` | `adapters.to_step_bindings`（plan，按成功树）；`adapters.bindings_from_steps`（agent，按步骤与当前持物） |
| 加载期检查 | `load.ValidationError` | `load.check_object_ids`、`load.check_task`，JSON Schema，`steps_to_skills` 的步骤检查 |

- 运行时读 `Scenario`：`SimEnv.build(scene)`、`EpisodeRun(task, target_frames=...)`、`resolve_static_target_positions(env, target_frames)`、
  `evaluate_scenario_task(task)`、recorder 的 `task_spec.to_dict()`。planner 与 `run_generated_scenario.py` 从 `scene.objects` 读
  `name_override` / `asset_query` / `yaw_deg`。
- 每次 `load_scenario` 都从文件重建全部对象，两次调用（同模式或一 plan 一 agent）不共享场景、任务、技能参数或目标帧；建成后没有写者，
  certify 改起点写的是 `env.placed_map`。plan 文件在两种模式下都读并检查（坏的 plan 文件在 agent 模式也报错），只在 plan 模式用：
  任务填上它的技能与 `step_bindings`；agent 模式任务不预填，目标帧换成全部 place 目标。
- 加载失败抛 jsonschema `ValidationError`、`load.ValidationError`（物体 id 重复：`duplicate object names: [...]`；成功树叶子参数里的
  `object` / `objects` / `parent` / `reference` 不是场景物体、目标或 `table`：`success predicate references unknown names: [...]`）、
  `SolveError`（关系求解失败，或 plan 的 place 与成功树矛盾）、`PlanInvalid`（步骤与持物顺序）、`AssetNotAdmitted`。
- YAML 场景的物体都是 `fixed_position`，place 目标都是带锚点的 `FrameRef`；判定与诊断用的静态目标点只有
  `resolve_static_target_positions` 按 `target_frames` 对活世界求出的那些。按 region 随机摆放只有单资产试抓入口一处，它直接建带
  `position_range` 的 `SceneConfig`，每次 reset 重新采样位置。
- `ObjectPlacement.count` 在 YAML 场景恒为 1，但 `scenes/backend.py` 按它摆放多份（`tests/test_scenes.py` 用到 5 与 20）。

## 一个技能：候选、整条轨迹、过门

```mermaid
flowchart TD
    subgraph PK["RulePolicy.pick：按候选顺序"]
        GEO["ManipulationTools.grasp_candidates → grasp/geometric.GeometricGraspPlanner.plan<br/>写回的抓法在前，再按有向包围盒枚举 top_down / side"]
        PM["ManipulationTools.plan_pick<br/>当前构型张指；IK 解 pre-grasp（初值 = home 构型、基座关节转向抓取方位，或 ik_seed）<br/>从 pre-grasp 起的 pick 列表先单独 verify，再 JointPlanner.connect 从张指后的构型规划进场段<br/>MotionExecutor.run 从当前构型跑 meta_actions.pick_actions 构造的整张列表，进场段是带路径的 move_to 项<br/>plan_state.binding 取候选的抓取位姿"]
        V1{"ManipulationTools.verify → ExecutionGate.verify"}
        PRE{"rule_policy._place_check：place_next 不为空时<br/>在 ManipulationTools.posed(pick 末点, 目标在手中) 里以复制的 SkillContext 调 plan_place<br/>凭证与 trace 丢弃"}
        TR["ManipulationTools.trial → sim_env.SimEnv.trial_from_here<br/>重放、恢复快照；物体抬高超过 0.02 m 算抬起"]
        V2{"恢复后对同一条轨迹再 verify"}
    end

    subgraph PL["ManipulationTools.plan_place"]
        PP["manipulation_tools._resolve_place_plan → grasp/planner.plan_place<br/>手的落点 = 目标点 + 抓取相对高度，朝向取插入方向<br/>只有竖直放置时物体落在目标点上"]
        PC["ManipulationTools._connect：IK 解 standoff，手指取当前值<br/>JointPlanner.connect 带持物规划搬运段"]
        PX["MotionExecutor.run(meta_actions.place_actions(phase=descend))<br/>从 standoff 下放、松手、退回"]
        V3{"ExecutionGate.verify<br/>rest = 放置目标的 entity；plan_state.binding 取实测手与物体位姿"}
    end

    subgraph LOW["底层：执行与运动"]
        subgraph EX["执行 execution/"]
            EXE["executor.MotionExecutor.run：动作列表进运动链的唯一入口<br/>meta_actions 的 pick / place 列表与 drag_handle 的段：RuntimeMotionWaypoint（allow / linear）与 Attach / Detach"]
        end
        subgraph MO["运动 motion/"]
            CH["chain.execute_motion_chain → _solve_segment 逐段路由<br/>PlanState 逐段推进 q，Attach / Detach 项改持物、零运动<br/>_assemble_trajectory 给每段写 span / holding / allow / pose / reach"]
            F["只动手指：gripper.expand，手臂锁定"]
            C["直线段（最终接近、下放、拖拽、退回）<br/>move_linear.expand，逐步 IK；pick 遇跳变拒绝，place 与拖拽保留"]
            G["其余段：GenesisBackend.solve_ik + _interpolate，关节线性插值"]
        end
    end

    GEO --> PM --> V1
    V1 -- 拒绝，记 gate --> PM
    V1 -- 通过 --> PRE
    PRE -- place 被拒，记 place --> PM
    PRE -- 可达或不预检 --> TR
    TR -- 没抬起 --> PM
    TR -- 抬起 --> V2
    V2 -- 通过 --> OKP["签发 VerifiedMotion"]
    PP --> PC
    PC -- 拒绝 --> RF["Refused(search_exhausted)，零物理步"]
    PC -- 连通 --> PX --> V3
    V3 -- 拒绝 --> RF
    V3 -- 通过 --> OKL["签发 VerifiedMotion"]
    PM & PX --> EXE --> CH --> F & C & G

    style PK fill:#e8eef9,stroke:#3b5b9a
    style PL fill:#e8eef9,stroke:#3b5b9a
    style LOW fill:#fff4e0,stroke:#b9770e
```

同一个 pick 按调用顺序（certify 用的 file 规划器，`grasp_selection="trial"`）：

```mermaid
sequenceDiagram
    participant R as EpisodeRun
    participant P as RulePolicy
    participant T as ManipulationTools
    participant G as GeometricGraspPlanner
    participant Q as ExecutionGate
    participant J as JointPlanner
    participant E as SimEnv

    R->>R: _live_scene_state
    R->>P: plan_skill → pick(ctx, tools, place_follows, place_next, trial)
    P->>T: grasp_candidates(ctx)
    T->>G: plan(obj_pos, asset)
    G-->>P: 候选（写回的抓法在前）
    loop 每个候选，直到第一个通过
        P->>T: plan_pick(ctx, plan, params)
        T->>Q: verify（从 pre-grasp 起的 pick 段）
        T->>J: connect(opened, q_pre)
        T-->>P: request
        P->>T: verify(request)
        T->>Q: verify（整条 pick 轨迹）
        opt place_next 不为空
            P->>T: posed(q_end, 手里的物体)
            P->>T: plan_place(place_ctx)，凭证丢弃
        end
        opt place 可达或不预检
            P->>T: trial(trajectory, target)
            T->>E: trial_from_here(trajectory)
            E-->>P: lifted（抬高超过 0.02 m）
            opt lifted
                P->>T: verify(request)（快照恢复后再签一次）
            end
        end
    end
    P-->>R: VerifiedMotion 与 held_place_z
    R->>Q: check_current(motion)
    R->>E: record_fn 逐点回放
```

试抓步进了物理，之前签发的凭证随状态指纹失效，所以抬起后要再 verify 一次。

门检查的是会被执行的每一个点：pick 的规划进场段、手指段与段后保持（含抬升后的保持）都在 `spans` 里，`spans` 必须铺满整条轨迹。
pick 的手指闭合段允许手指碰目标（`grip`），其后的直线抬升持物且允许离开支撑面（`rest`）；place 的规划搬运段持物，
下放持物且 `rest`，下放之后 detach，松指与退回 `grip`。持物段把目标摆在 手 · T，T 由技能以 `Holding` 传入，门在持物开始之前的构型上核对
手 · T 与目标位姿（≤ 5 mm / 0.05 rad）；detach 之后目标留在 手(最后持物点) · T，松指与退回时手掌碰到它会被拒绝。
航点到达点带期望手部位姿（`poses`），门判 FK 残差；自由插值的中间点不带。

`trial_all`（试遍全部候选，修复工具用）在最后一次试抓之后再按候选顺序对抬起者的原轨迹重新过门，第一个通过的被选中。

## 执行门的边界

| 位置 | 在门里 | 不在门里 |
|---|---|---|
| pick / place 的整条轨迹 | 每个点，含 pick 的规划进场段、place 的规划搬运段、手指与保持段；持物与松手后的物体；非相连连杆的自碰撞 | 相连（父子、同父）连杆的接触 |
| 试抓 `SimEnv.trial_from_here` | 试抓前后对同一条轨迹各过一次门 | 试抓本身是物理步进，判抬起，不判接触 |
| settle 与任务判定 `settle_window` | 只下发最后一个已验证点 | 保持期间的物理演化 |
| 开合抽屉 `open_close` | — | `gate.ungated`，不查接触；验收族里没有抽屉 |
| 场景初始化 `reset_scene` | — | 初始 reset / settle 属于场景初始化 |
| `handoff.check` | — | 持物距离、首次脱手帧、可能原因，只记录 |

## 运动层与执行门的接口

- `execution/gate.ExecutionGate`：`check_point(q, span, *, target, rest, held, placed, pose)` 返回 `None` 或拒绝原因；
  `verify(trajectory, spans, *, target, q_start, rest, holding, poses, budget)` 对每个点调它（给了 `budget` 就在每点前后
  调它的 `spend()` / `check()`），返回只读
  `VerifiedMotion(trajectory, spans, fingerprint, checked, gated, holding)`（轨迹与 `Holding` 的变换存为 `bytes`，每次读取
  得到新的只读视图，所以每次读 `trajectory` 都是新 tuple，certify 按内容把回放段对回凭证），违规抛 `MotionRejected(span, index, rejection)`；
  `check_current(motion)` 不是本门签发或指纹不一致抛 `StaleMotion`；`ungated(trajectory)` 只给开合抽屉。许可规则在
  `permitted`。步骤级结果 `Refused(label, evidence)`（`search_exhausted` / `unsupported` / `infeasible`，零物理步）与
  `ExecutionFailed(evidence, steps)` 也在这个模块。
- `sim/sim_env.SimEnv`：`contacts_at(qpos, held, placed)`、`hand_pose(qpos)`、`object_pose(name)` 在快照事务里查询，
  返回前还原；`joint_limits`、`joined_links`、`lever_arms(held)`（每个关节的力臂上界，持物算在手上）读机器人模型；
  `hold(qpos, name, T)` 把机器人与物体摆成持物起态，是一次不步进的真实写入；`state_fingerprint()` 带真实写入代次，
  `step()`、`reset()` 与 `hold()` 递增。`posed(qpos, objects)` 是不步进的摆放事务：手臂与给定物体摆到给定位姿，退出时还原；
  Genesis 的快照还原会丢掉未步进的写入，所以事务里每次查询还原后都重放摆放。事务期间指纹带事务序号，里面签发的凭证在事务外不当前。
- `motion/joint_planner.JointPlanner(env, gate, *, seed, max_queries, max_seconds).connect(q_start, q_goal, *, target, rest,
  holding, label)`：关节空间 lazy RRT-Connect，状态与按 `lever_arms` 细分到每步 ≤ 5 mm 的连边点只经 `check_point`，路径由
  `verify(..., budget=)` 签发，返回 `Connection(motion, queries, seconds, nodes, direct)`。搜索与签发共用一个 budget：
  1500 次查询或 30 s，`queries` 是门实际检查的点数；超出抛 `Refused("search_exhausted")`，不签发。`place` 用它规划搬运段，
  `pick` 用它从张指后的构型规划到 pre-grasp 的进场段。
- `motion/planner.MotionPlanner` 只有 `solve_ik(pos, quat, finger_pos, *, init_qpos)`；唯一实现 `GenesisBackend` 包装
  `SimEnv.solve_ik`。
- `motion/move_linear.expand(MoveLinear(start_pos, end_pos, quat, finger_width, on_jump, label), q_start, solve_ik)`
  把一段笛卡尔直线展开成关节路径：`start_pos` 是上一航点的名义位置，`q_start` 是逐点 IK 的初值，每 1 cm 一点、至多 24 点；
  单步关节变化超过 0.8 rad 时，`on_jump="refuse"` 抛 `RuntimeError("<label> IK discontinuity: ...")`，`"keep"` 返回路径并标
  `continuous=False`。零物理步、无状态，不签发凭证。
- `motion/gripper.expand(Gripper(pos, quat, finger_width), q_start, solve_ik)` 给只动手指的段算目标构型：臂关节取
  `q_start`，指关节取 `solve_ik(pos, quat, finger_width)` 的指关节（不带初值），返回 `GripperSegment(q_target)`。
  零物理步、无状态，不签发凭证，不改持物状态。
- `motion/plan_state.PlanState(q, holding)` 是预测构型与持物绑定；`attach(state, Holding)` / `detach(state)` 是改变持物的
  唯一途径，保持 `q`，在动作列表里写成 `Attach(holding)` / `Detach()` 两项。`binding(target, hand_pose, object_pose)` 给出
  `Holding(target, hand⁻¹ · object)`，pick 传候选抓取位姿，place 传实测手位姿。
- `motion/chain.execute_motion_chain(actions, motion_planner, start: PlanState, params)` 逐项执行动作列表：
  运动段做「航点 → IK（`finger_only` 段交给 `gripper.expand`，`linear` 段交给 `move_linear.expand`）→ 关节插值」，
  带 `path` 的 `move_to` 段直接取规划器给的点（空路径不动），
  `Attach` / `Detach` 在所在位置改持物、不加轨迹点、不占段号。返回 `MotionChainExecution`，每段带 `span`、`holding`、`allow`、
  航点位姿 `pose` 与到达下标 `reach`；`spans_of` 把它们转成门的 `Span`，`poses_of` 转成 `verify` 的 `poses`。
  每段从 `state.q` 起、段末推进到终点构型，段的 `holding` 读运行时的状态；末态在 `.state`。
  跳变策略写在航点的 `linear` 上：pick 的最终接近 `refuse`，pick 的抬升、place 的下放、退回与拖拽各段 `keep`。
- `execution/meta_actions.pick_actions(plan, start_q, start_hand, approach, params, solve_ik, *, grasp, lift)` 与
  `place_actions(place_plan, params, *, phase)` 对一个候选按固定顺序构造动作列表：
  pick 是张指（起点已张开则 0 帧）→ 带规划路径的 `move_to` 到 pre-grasp → `move_linear` 到抓取位姿 → 合指（含 settle）→
  `Attach(grasp)` → 沿 `lift` 的 `{direction, dist}` 直线抬升（技能没给时是 `PICK_LIFT`，世界 +Z 0.10 m）；place 是搬运 →
  直线下放 → `Detach()` → 松指 → 直线退回。
  两张表只读候选的位姿与指宽，不读也不分类它的航点；
  `place` 的 `descend` 从当前构型先走一段到 standoff 的关节插值（规划搬运之后几乎不动），再下放、松手、退回。
- `MotionExecutor.run(name, actions, motion_planner, start: PlanState, params)` 是动作列表进运动链的唯一入口，
  把 trace 记为 `last_motion_trace`；pick 的 `lift_hold_steps` 是抬升项自己的 `hold_after_steps`。`drag_handle(..., action)` 也经它。
- `scenario/relations.rigid_moving_links`：关节子 link 加上与它固定连接的 link，给 `clear_of` 的扫掠占地用。
- `scenes/genesis_loader` 加载 Franka 时给 `gs.materials.Rigid(gravity_compensation=1.0)`：机器人连杆的重力被完全抵消，
  持物与其它物体照受重力，关节 PD 停在门检查过的命令构型上，不靠位置误差扛重力。门仍只判接触，回放仍是每点一次下发、
  一步物理、不确认到位。原因与限制见[机器人控制与回放](manipulation.md#机器人控制与回放)。
- 启动脚本 `scripts/setup/run_scenario.sh` 只挂仓库与 Hugging Face 缓存，不挂 Docker socket。

## 开关

| 开关 | 默认 | 作用 |
|---|---|---|
| `--planner` | `file` | 步骤表来自 plan 文件（`file`）还是 LLM（`agent`） |
| `--role` | `certifier` | agent 路径下是否看见成功树（`certifier` 看见，`baseline` 不看见） |
| `ROBOSMITH_LLM_*` | — | agent 路径调用的网关、模型与密钥 |

## 其他入口

- 单资产抓取验证（`scripts/datagen/validate_asset_picks.py` → `rollout/asset_pick_validation.run_asset_pick`）每个
  episode 建一个 `EpisodeRun`（`grasp_selection` 选 `trial` / `trial_all`），pick 带公开组合的 pick 段，走同一套技能、门与
  执行器；pick 停下时把 `cause`、`evidence`、`steps` 记进 `stopped`、零帧，不写 episode。
- 修复环工具（`grasp/diagnostics/tool.try_candidates`）把 agent 提出的抓法交给单资产入口，以 `trial_all` 试遍，
  所以同样只经门执行。单资产入口在第 0 个 episode 读 `GeometricGraspPlanner.last_plans`（最近一次 `plan()` 的全部候选），
  全部失败时写进诊断请求，这是它唯一的读者。

`GraspPlan` 的 `retreat_pos` / `retreat_quat` 只有 `plan_place` 填（放下后退回的位姿，`place_actions` 读）；pick 候选不带它们，
抬升由 pick 步骤的 `lift` 给（缺省世界 +Z 0.10 m）。

## 评测考场

外部策略经 Inspect Robots 的 `Embodiment` 协议进 RoboSmith 的场景；要保证什么、考生看到什么、怎么判分、已测结果与限制见
[evaluation](evaluation.md)。这里只画代码路径：`robotsmith/eval/` 与 `scripts/eval/run_exam.py`。

接触许可在代码里：`judge.exam_context` 从成功树取 target（唯一被搬动的物体）与 rest（`env` 加叶子的 `parent`），`judge.held`
判两个不同手指连杆碰 target，`judge.violations` 每步合成 `Span("exam", 0, 1, holding=held, allow="rest" if held else "grip")`
后对每对接触调 `gate.certified`。在线的 `ExamMonitor` 与离线的 `eval.judge.judge` 调同一组函数。

### 对象

```mermaid
classDiagram
    class Embodiment {
        <<interface>>
        info
        +reset(scene, seed)
        +step(action)
        +close()
    }
    class LLMAgentPolicy {
        +act(observation)
    }
    class RoboSmithEmbodiment {
        info
        +reset(scene, seed)
        +step(action)
        +close(ended)
    }
    class ExamMonitor {
        violating
    }
    class JudgeLog {
        +header(seed, attempt, scenario, success, target, rest)
        +physics(k, contacts, held)
        +control(step, action, world)
        +final(ended, success, state)
    }
    class Verdict {
        ended
        success
        certified
        violating_steps
        env_error
    }
    class SimEnv
    class Scenario
    LLMAgentPolicy ..> Embodiment : inspect_robots eval 驱动
    Embodiment <|.. RoboSmithEmbodiment
    RoboSmithEmbodiment --> Scenario : load_scenario(agent_plan=True)
    RoboSmithEmbodiment *-- SimEnv : reset 时建
    RoboSmithEmbodiment *-- JudgeLog
    RoboSmithEmbodiment *-- ExamMonitor
    ExamMonitor --> SimEnv : 唯一的 on_step 读者
    ExamMonitor --> JudgeLog : physics
    JudgeLog ..> Verdict : eval.judge.judge
```

| 概念 | owner | 复用 |
|---|---|---|
| 机器人层接口与观测 | `RoboSmithEmbodiment`（`eval/embodiment.py`） | `load.load_scenario`、`SimEnv.build`、`runner.reset_scene`、`SimEnv.solve_ik`、`GripperSpec` |
| 逐物理步监测 | `ExamMonitor`（同上） | `SimEnv.on_step`、`SimEnv.contacts_now`、`gate.certified` |
| 判卷日志 | `JudgeLog`（`eval/log.py`） | 与 LeRobot 录制分开、不可关；考场不调 `recorder.record_episode` |
| 终态判定 | `RoboSmithEmbodiment.close` | `runner.evaluate_scenario_task`（静置 0.5 s 后判成功树），判过的状态写进 `final` |
| 判卷与复判 | `eval.judge.judge`（`eval/judge.py`） | 纯函数，日志进、`Verdict` 出；`predicates.evaluate_success` |
| 运行与尝试 | `scripts/eval/run_exam.py` | `batch.run_batch` 给 attempt 编号，`batch.supervise` 做进程外超时（每次尝试 1800 s） |

`EpisodeRun`、`ExecutionGate.verify` 与技能不在考场路径上；它们照旧做出题人。

### 一次外部 episode：agent 走 `move_to` 到判卷

```mermaid
sequenceDiagram
    participant R as run_exam
    participant P as LLMAgentPolicy
    participant B as RoboSmithEmbodiment
    participant E as SimEnv
    participant M as ExamMonitor
    participant L as JudgeLog

    R->>B: 构造（scenario、seed、run_batch 给的 attempt、日志路径）
    R->>P: inspect_robots eval(task, policy, embodiment, before_scoring)
    P->>B: reset(scene, seed)
    B->>E: build → reset_scene
    B->>L: header（seed、attempt、成功树、target、rest、sha256）
    B->>M: 挂到 on_step
    B-->>P: Observation（eef_state、object_pos.*，无图像）
    loop 直到 done / give_up / 步数上限
        P->>P: LLM 调用 → move_to → 插值成 ActionChunk
        loop chunk 里每个 Action
            P->>B: step(action)
            B->>E: solve_ik → 关节目标 → step × 3
            E->>M: on_step（每个物理步）
            M->>L: physics（接触对、held）
            B->>L: control（动作、collect_world_state）
            B-->>P: StepResult（观测）
        end
    end
    P-->>R: 返回；before_scoring 记下结束原因
    R->>B: close(ended)
    B->>E: evaluate_scenario_task（静置 0.5 s，静置的物理步也进 physics）
    B->>L: final（ended、success、判过的世界状态）
    R->>R: eval.judge.judge(日志) → Verdict，写进 instances.jsonl 的这一行
```

### 模块契约

#### 日志布局与生命周期

- 一次运行一个目录（`--out`）：`run.json`（场景、seed、RoboSmith commit、被测方身份：策略名、两个包的版本与 Inspect Robots
  的 commit、模型、effort、调用与时长上限、特权观测）；`instances.jsonl`（每次尝试一行，带 `verdict`）；`report.txt`。
- 每次尝试一个目录 `trials/<seed>/attempt-<n>/`：`judge.jsonl` 与被测方自己的日志 `inspect/`（eval log、transcript、
  wire capture）。attempt 编号由 `batch.run_batch` 给出，与 `instances.jsonl` 里同一 attempt 的行对应；重试不追加、不覆盖。
  `JudgeLog` 以独占方式新建文件，路径已存在就报错。
- `judge.jsonl` 每行一个 JSON，写完即 flush：第一行 `header`，之后 `physics`（每个物理步，含终态静置）与 `control`
  （每个控制步）交错，最后一行 `final`。进程中途死掉时已写的行仍可读。
- 运行结束后日志不再改写；复判只读不写。运行目录不进 git。

#### 读写方

- 写：`RoboSmithEmbodiment`（`header`、`control`、`final`）与它创建的 `ExamMonitor`（`physics`），同一进程同一线程。
- 读：`eval.judge.judge`，在线由 `run_exam.py` 在 `close` 之后调，离线用 `python -m robotsmith.eval.rejudge <judge.jsonl>...`；
  它只 import 判卷所需的纯 Python 模块，不启动 Genesis、torch、Inspect Robots 或 LLM。被测方读不到判卷日志。
- `Verdict`：seed、attempt、ended、success、certified（成功且零违规步）、physics_steps、violating_steps、first_violation、
  env_error。`instances.jsonl` 另记 `ExamMonitor` 在线数出的违规步与被测方的 LLM 请求数（数 wire capture 的行）。

#### 不变量

| 不变量 | 测试 |
|---|---|
| 三个接触样例按 `gate.certified` 判：合指与正常松手合法，持物擦墙违规 | test_eval_judge.py::test_contact_samples_closing_scraping_a_wall_and_releasing |
| 被测方写的字段（动作、note）改成任何值，`Verdict` 不变 | test_eval_judge.py::test_evaluatee_fields_do_not_change_the_verdict |
| 四种正常结束（done、give_up、horizon、policy_error）都写 `final`，成败只看判过的状态 | test_eval_judge.py::test_endings_all_write_a_final_and_are_judged_by_the_physics |
| 缺 `final` 判 `env_error`，不算被测方失败 | test_eval_judge.py::test_a_missing_final_is_an_environment_error |
| 同 seed 两次尝试各自一个日志、各自判卷；`run_batch` 把 attempt 交给写入方 | test_eval_judge.py::test_attempts_keep_separate_logs、test_run_batch_names_the_attempt_it_runs |
| 离线入口与 `judge` 给出同一 `Verdict` | test_eval_judge.py::test_rejudge_prints_the_same_verdict |
| embodiment 过 Inspect Robots 的 conformance kit，被测方的工具里没有判卷入口 | test_eval_embodiment.py |

#### 结束路径

- `done` / `give_up`：agent 的终止工具；`horizon`：Inspect Robots 的步数上限（`--max-seconds` 仿真秒）；`policy_error`：
  试验出错（如 LLM 调用在 agent 自己的重试后仍失败），或 `before_scoring` 没被调到。四种都由 `close` 静置、判成功树、写 `final`。
- 仿真报错或进程被 `supervise` 超时杀掉：日志缺 `final`，`Verdict.env_error` 为 `no final line`，`run_batch` 重试一次，新尝试
  写新目录。
- 被测方调 `done` 不等于成功。
