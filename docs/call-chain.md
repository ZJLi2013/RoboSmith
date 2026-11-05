# 调用链：一次 scenario episode 怎么跑

开发文档，画 `main` 上的现状，粒度到函数；设计（要保证什么）在 [system-design](system-design.md) 与 [manipulation](manipulation.md)。改了这条链的 PR 同时改这里。

**默认路径上没有一处在执行前检查动作是否可行**：抓取候选只看能否抬起，自由段是碰撞盲的 IK 插值，开 rocRobo 时规划失败也退回碰撞盲直线。

## 四层

| 层 | 回答什么 | 组件（模块） |
|---|---|---|
| 编排层 | 做哪几步、失败后怎么办 | `scenarios/teacher/`、`scenario_runtime/planner.py`、`scenario_runtime/runner.py` 的 `EpisodeRun` |
| 技能层 | 一步（pick / place / open / close）怎么变成一串关节目标 | `skills/` |
| 底层：抓取与运动 | 抓哪里、航点是什么、每段怎么解出关节 | `grasp/`、`execution/`、`motion/` |
| 仿真与判定 | 执行、录帧、判这一步和整个任务成没成 | `sim/`、`scenario_runtime/handoff.py`、`tasks/`；判定函数住在 `runner.py` |

每个框写 `模块.函数`，框按所属模块放进层里；同一层内的子框是可以单独替换的组件。

```mermaid
flowchart TD
    CLI["scripts/datagen/run_generated_scenario.py<br/>main → materialize_candidate → run_scenario_smoke"]

    subgraph ORCH["编排层：做哪几步"]
        T["teacher（--planner 默认）<br/>scenarios/teacher/ 下同名 YAML 的步骤表<br/>runner.run_scenario_episode"]
        A["agent（--planner agent）<br/>planner.run_agentic_episode<br/>render_request → call_llm → parse_steps"]
        RF["planner.route_failure<br/>plan 类失败就重写剩余步骤，最多 2 次"]
        RUN["runner.EpisodeRun.run_steps<br/>每步按 skill_phases 分段<br/>每段先 _live_scene_state 重读仿真"]
    end

    subgraph SKILL["技能层：skills/registry.plan_segment 按 SKILL_DEFS 分发"]
        PICK["pick.run_pick<br/>_select_by_trial：按序试候选，取第一个抬起的<br/>_pick_seed_transit：转基座，关节插值"]
        PLACE["place.run_place<br/>_resolve_place_plan：落点<br/>_author_transport_hover：扫 standoff（仅 rocRobo）"]
        OC["open_close.run_open / run_close<br/>frames._resolve_handle_drag：把手与拖动方向"]
    end

    subgraph LOW["底层：抓取与运动"]
        subgraph GR["抓取 grasp/"]
            GEO["geometric.GeometricGraspPlanner.plan<br/>写回的抓法在前，再按有向包围盒枚举<br/>每个候选 4 个航点：pre → grasp → 闭合 → lifted"]
            PP["planner.plan_place<br/>放置的 pre / drop / retreat 航点"]
        end
        subgraph EX["执行 execution/"]
            EXE["executor.MotionExecutor<br/>pick / place / drag_handle：航点模板<br/>标注每段是手指、笛卡尔还是自由段"]
        end
        subgraph MO["运动 motion/"]
            CH["chain.execute_motion_chain<br/>_solve_segment 逐段路由"]
            F["只动手指：手臂锁定"]
            C["笛卡尔段：cartesian.solve_cartesian_ik_path<br/>逐步 IK，只查关节跳变"]
            FREE{"自由段：<br/>设了 ROCROBO_BACKEND？"}
            G["GenesisBackend.solve_ik + 关节线性插值<br/>碰撞盲（默认）"]
            R["RocRoboBackend.plan_motion<br/>rocrobo_world：只含固定件的盒子世界"]
            MISS{"规划成功？"}
        end
    end

    subgraph SIM["仿真与判定"]
        TRIAL["sim_env.SimEnv.trial_from_here<br/>重放候选轨迹、恢复快照<br/>只判物体抬高超过 0.02 m"]
        REC["recorder.record_episode<br/>Genesis 逐帧执行并录帧"]
        HO["handoff.check<br/>手、持物、关节的逐帧读数，只记录"]
        JC["runner._judge_completion<br/>静置后判这一步绑定的谓词"]
        TASK["runner.evaluate_scenario_task<br/>静置 0.5 s，判任务的成功树"]
    end

    CLI --> T & A
    T --> RUN
    A --> RUN
    RUN -->|失败的步| RF
    RF -->|route = plan| A

    RUN --> PICK & PLACE & OC
    PICK --> GEO
    PLACE --> PP
    PICK & PLACE & OC --> EXE
    PICK -->|每个候选试一次| TRIAL

    EXE --> CH
    CH --> F & C & FREE
    FREE -- 否 --> G
    FREE -- 是 --> R --> MISS
    MISS -- 否：warning 后退回 --> G

    RUN -->|技能返回的关节目标| REC
    REC --> HO --> JC -->|下一段 / 下一步| RUN
    RUN -->|全部步骤结束| TASK

    classDef blind fill:#fde2e2,stroke:#c0392b,stroke-width:2px
    class G,C,TRIAL blind
    style ORCH fill:#e3f1e0,stroke:#2e7d32
    style SKILL fill:#e8eef9,stroke:#3b5b9a
    style LOW fill:#fff4e0,stroke:#b9770e
    style SIM fill:#f2f2f2,stroke:#666
```

红框是不查碰撞的地方。另有两段关节插值同样不经检查：`pick._pick_seed_transit` 的转基座，以及 rocRobo 关闭时 `place._place_transport_seed` 的搬运前转向。

## 不做检查的地方

| 位置 | 层 | 现在判什么 | 不判什么 |
|---|---|---|---|
| `SimEnv.trial_from_here` | 仿真与判定 | 物体终点比起点高 0.02 m | 重放过程中手、臂与固定件的接触 |
| `chain._solve_segment` 自由段，默认 | 底层 | IK 解出终点 | 路径与终点的碰撞 |
| `chain._solve_segment` 自由段，rocRobo | 底层 | rocRobo 的盒子世界里无碰撞 | 松散物体；规划失败后退回的直线 |
| 笛卡尔段 | 底层 | 相邻步关节跳变（pick 超限即拒，place / drag 不拒） | 碰撞 |
| `handoff.check` | 仿真与判定 | 持物距离、首次脱手帧、可能原因 | 不中断 episode，只记录 |

## 开关

| 开关 | 默认 | 作用 |
|---|---|---|
| `--planner` | `teacher` | 步骤表来自 YAML（`teacher`）还是 LLM（`agent`） |
| `--role` | `certifier` | agent 路径下是否看见成功树（`certifier` 看见，`baseline` 不看见） |
| `ROCROBO_BACKEND` | 未设 | 设了才构建碰撞世界、自由段走 rocRobo；需要 sidecar 容器 |
| `ROBOSMITH_LLM_*` | — | agent 路径调用的网关、模型与密钥 |

## 其他入口

单资产抓取验证（`scripts/datagen/validate_asset_picks.py` → `rollout/asset_pick_validation`）经 `skills.run_skills` 走技能层以下的同一套组件，不经编排层；修复环工具（`grasp/diagnostics/tool`）直接用 `GeometricGraspPlanner` 和试抓。
