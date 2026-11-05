# 调用链：一次 scenario episode 怎么跑

开发文档，画 `main` 上的现状，粒度到函数；设计（要保证什么）在 [system-design](system-design.md) 与 [manipulation](manipulation.md)。改了这条链的 PR 同时改这里。

**链路上没有一处在执行前检查动作是否可行**：抓取候选只看能否抬起，自由段是碰撞盲的 IK 插值，笛卡尔段只查关节跳变。
要补上的执行门见 [manipulation](manipulation.md#执行门的接口与契约proposed尚未实现)。

## episode：从入口到判定

```mermaid
flowchart TD
    CLI["scripts/datagen/run_generated_scenario.py<br/>main → materialize_candidate → run_scenario_smoke"]

    subgraph ORCH["编排：做哪几步"]
        T["--planner teacher（默认）<br/>scenarios/teacher/ 下同名 YAML 的步骤表<br/>runner.run_scenario_episode"]
        A["--planner agent --role certifier|baseline<br/>planner.run_agentic_episode<br/>render_request → call_llm → parse_steps"]
        RF["planner.route_failure<br/>plan 类失败重写剩余步骤，最多 MAX_REPLANS = 2 次"]
        RUN["runner.EpisodeRun.run_steps<br/>每步按 skill_phases 分段；每段先 _live_scene_state 重读仿真"]
    end

    subgraph SKILL["技能：skills/registry.plan_segment 按 SKILL_DEFS 分发"]
        PICK["pick.run_pick"]
        PLACE["place.run_place"]
        OC["open_close.run_open / run_close"]
    end

    subgraph SIM["仿真与判定"]
        REC["sim/recorder.record_episode<br/>Genesis 逐帧执行关节目标并录帧"]
        HO["handoff.check<br/>读手、持物、关节的逐帧采样，只记录"]
        JC["runner._judge_completion<br/>静置 settle_s 后判这一步绑定的谓词"]
        TASK["runner.evaluate_scenario_task<br/>静置 0.5 s，判任务的成功树"]
    end

    CLI --> T & A
    T --> RUN
    A --> RUN
    RUN -->|stop_on_fail 时返回失败的步| RF
    RF -->|route = plan| A
    RUN --> PICK & PLACE & OC
    PICK & PLACE & OC -->|关节目标序列| REC
    REC --> HO --> JC -->|下一段 / 下一步| RUN
    RUN -->|全部步骤结束| TASK
```

技能内部见下一张图。teacher 与 agent 两条路径共用 `EpisodeRun`，只是步骤表的来源不同。

## 一个技能：候选、航点、逐段求解

```mermaid
flowchart TD
    subgraph PK["pick"]
        GEO["grasp/geometric.GeometricGraspPlanner.plan<br/>写回的抓法在前，再按有向包围盒枚举 top_down / side"]
        WP["每个候选 4 个航点：pre → grasp → 闭合 → lifted"]
        TR["pick._select_by_trial<br/>逐个候选：_pick_motion 生成整条轨迹<br/>→ sim_env.SimEnv.trial_from_here 重放、恢复快照"]
        OK{"物体抬高超过 0.02 m？"}
    end

    subgraph PL["place"]
        PP["grasp/planner.plan_place<br/>落点 = 目标点 + 抓取相对高度"]
    end

    DR["open / close<br/>executor._open_drag_waypoints / _close_drag_waypoints"]

    EXE["execution/executor.MotionExecutor<br/>pick / place / drag_handle：航点模板 → RuntimeMotionWaypoint"]
    CH["motion/chain.execute_motion_chain → _solve_segment 逐段路由"]

    F["只动手指：手臂锁定"]
    C["笛卡尔段（最终接近、下放、拖拽、退回）<br/>motion/cartesian.solve_cartesian_ik_path<br/>逐步 IK，只查关节跳变"]
    G["自由段：GenesisBackend.solve_ik + _interpolate<br/>关节线性插值，碰撞盲"]

    GEO --> WP --> TR --> OK
    OK -- 否 --> TR
    OK -- 是，执行这条轨迹 --> EXE
    PP --> EXE
    DR --> EXE
    EXE --> CH
    CH --> F & C & G

    classDef blind fill:#fde2e2,stroke:#c0392b
    class G,OK,C blind
```

红框是不查碰撞的地方。pick 在航点之前另有一段转基座的关节插值（`pick._pick_seed_transit`），place 也有一段（`place._place_transport_seed`），两者都不经任何检查。

## 不做检查的地方

| 位置 | 现在判什么 | 不判什么 |
|---|---|---|
| `SimEnv.trial_from_here` | 物体终点比起点高 0.02 m | 重放过程中手、臂与固定件的接触 |
| `chain._solve_segment` 自由段 | IK 解出终点 | 路径与终点的碰撞 |
| 笛卡尔段 | 相邻步关节跳变（pick 超限即拒，place / drag 不拒） | 碰撞 |
| `handoff.check` | 持物距离、首次脱手帧、可能原因 | 不中断 episode，只记录 |

## 开关

| 开关 | 默认 | 作用 |
|---|---|---|
| `--planner` | `teacher` | 步骤表来自 YAML（`teacher`）还是 LLM（`agent`） |
| `--role` | `certifier` | agent 路径下是否看见成功树（`certifier` 看见，`baseline` 不看见） |
| `ROBOSMITH_LLM_*` | — | agent 路径调用的网关、模型与密钥 |

## 其他入口

单资产抓取验证（`scripts/datagen/validate_asset_picks.py` → `rollout/asset_pick_validation`）经 `skills.run_skills` 走同一套技能与执行器，不经 `EpisodeRun`；修复环工具（`grasp/diagnostics/tool`）直接用 `GeometricGraspPlanner` 和试抓。
