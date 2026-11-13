# 评测：外部策略在 RoboSmith 场景里跑任务，按物理记录判卷

RoboSmith 把一个场景变成考场：外部的机器人策略（agentic 系统、code-as-policy 或 VLA）不改自己的代码进来跑任务，
**判卷只看仿真里实际发生了什么，不经执行门、不读被测方写的任何字段**；判卷日志事后可以离线重判，结果必须一样。
现在有最小版本：一个外部 LLM agent 在空桌面场景的一个实例上跑通并 certified（[已测结果](#已测结果)）。代码怎么走见
[call-chain 的评测考场](call-chain.md#评测考场)。

## 四个角色

| 角色 | 是谁 | 做什么 |
|---|---|---|
| 考生 | 外部策略，第一个是 [Inspect Robots](https://github.com/robocurve/inspect-robots) 的 `agent`（前沿 LLM 经工具调用驱动机器人），原样安装、不打补丁 | 读观测、发动作，自己决定什么时候交卷 |
| 考场 | `RoboSmithEmbodiment`：把一个场景的一个 seed 包成 Inspect Robots 的 `Embodiment` | 给考生观测，把考生的动作变成关节目标、步进物理 |
| 监考 | `ExamMonitor`：考场运行里唯一读每个物理步的观察者 | 每个物理步记下谁碰了谁 |
| 判卷 | `eval.judge.judge`：只读判卷日志的纯函数 | 判成功与违规，给出 `Verdict`；在线与离线调同一个 |

Inspect Robots 的 `Embodiment` 协议同时接 VLA、code-as-policy 与 LLM agent，所以 RoboSmith 只实现考场这一侧，被测方用它们
各自已有的适配层。选它作第一个被测方，是因为同类开源 agent（ORA、CaPtain、uni-robot-agent、RPent、ASPIRE、RoboRSI）都靠 RGB
加 SAM3 或 VLM 看图感知，无图像观测下要换掉感知模块，等于改策略本身；这个 `agent` 不看图也能跑。

## Inspect Robots 在 RoboSmith 里的位置

[Inspect Robots](https://github.com/robocurve/inspect-robots)（robocurve，MIT）是一个机器人策略评测框架：一个 benchmark 接任意
策略与任意真机或仿真。RoboSmith 只用它的两样东西：

| 用 | 不用 |
|---|---|
| `Embodiment` 协议，当考场的机器人层接口；RoboSmith 只实现考场一侧（`robotsmith/eval/embodiment.py`） | 它的评分与 benchmark：成败与违规全由 RoboSmith 读仿真判，它的 scorer 只挂一个计步数的占位 |
| `agent` 插件，当第一个被测方，原样安装、不打补丁 | 它的日志不进判卷，只作被测方自己的记录留在 `inspect/` 下 |

它是可选依赖（`pip install -e ".[eval]"`），核心包与 certify 都不依赖它。它还是 alpha、API 会变，所以版本钉在 `7720195`
（2026-11-04，`v1.0.0` 之后 5 个提交），升级要显式换版本。适配只在 `robotsmith/eval/embodiment.py` 一个文件里，换框架的影响面也
在这里。

## 考生看到什么、能做什么

- **任务**：一句指令，F0 是 `Put the die on the mat.`。成功判定（骰子落在垫子顶面多大范围内、容差多少）只在场景 YAML 里，考生看不到。
- **观测**：手的指尖中点位置与夹爪开度（`eef_state`），以及每个物体的中心位置（`object_pos.<物体>`）。物体位置是直接从仿真读给
  考生的特权信息，考生不必自己感知；`EmbodimentInfo.docs` 与报告都写明这一点。没有图像。
- **动作**：指尖中点的绝对目标 x / y / z 加夹爪开度（0 闭合、1 张开），世界系；手始终竖直朝下，yaw 固定。每个动作考场用 IK 变成
  关节目标，步进 0.1 s 物理（10 Hz 控制、30 Hz 仿真）。
- **工具**：`move_to`、`sleep`、`done`、`give_up`。它们是 Inspect Robots 的 agent 插件定义的，不是 RoboSmith 定义的：
  - 插件按 embodiment 声明的动作空间生成控制工具，绝对末端目标对应 `move_to`（关节目标对应 `move_joints`，增量对应 `move_by`），
    它把一次调用插值成一串逐步动作，限速为每维量程 10%/s；`sleep` 原地保持；`done` / `give_up` 交卷。声明相机且开
    `images=on_demand` 时还会有 `take_pic`，这里没有相机。
  - embodiment 可以经 `contribute_tools` 给考生加工具；RoboSmith 没有加。把 RoboSmith 的 pick / place 技能、抓取候选或带执行门
    的规划器开放成工具，就是「技能 / API 工具赛道」，还没做；现在考的是机器人层，抓法、轨迹、放置全由考生自己决定。

## 监考与判分

两层，粒度不同：

- **合不合法：每个物理步都判。** 考生没有执行门签发的凭证，所以考场从物理状态给出 `gate.certified` 需要的上下文，规则与
  RoboSmith 自己的技能轨迹用的是同一份：被搬动的物体是 target，桌面与目标面是 rest，两根手指都碰到 target 即算持物。机器人任何
  连杆碰垫子、桌面或墙，或持物碰到 rest 以外的东西，这一步就记违规。考生说「接触是有意的」不改变判定。

  | 样例 | 这一步的接触 | 持物 | 判定 |
  |---|---|---|---|
  | F0 合指：两指夹住桌上的 die | 左指–die、右指–die、die–桌面 | 是 | 合法：手指碰持物；持物碰桌面 |
  | 搬运中擦墙（F1 `acc_f1_wall_gap`，F0 没有墙） | 左指–die、右指–die、die–墙 | 是 | 违规：墙不在 rest |
  | F0 正常松手：die 落在垫子上，右指还挨着 | 右指–die、die–垫子 | 否 | 合法：未持物时手指可碰 target；自由的 die 只碰垫子 |

- **成没成：只看终态。** 考生交卷（或到时间上限）后，考场保持最后的指令静置 0.5 s，再按成功树判一次。考生调 `done` 不等于成功。
- **certified = 成功且全程零违规步。**
- **过程里程碑还没做。** 抓住、抬起、放上这些过程节点按时间线判「从未达到」还是「达到过又丢了」，是用来说明失败卡在哪一步，
  不改变 certified；设计见[还没做](#还没做)。

## 判卷日志

每次尝试一份 `judge.jsonl`，JSON 行：第一行题目头（seed、attempt、成功树、target、rest、场景文件 sha256），之后每个物理步一行
接触，每个控制步一行动作与世界状态（物体位置、AABB、接触、手指接触），最后一行终态（结束方式、静置后判过的世界状态）。
每行写完即落盘，进程中途死掉时已写的行仍可读。

- **不重**：已测的那次 episode（295 个控制步、900 个物理步）日志 391 KB，其中世界状态约 270 KB、接触约 115 KB；被测方自己的
  日志（transcript、LLM 请求记录）另有约 500 KB。
- **复判**：`python -m robotsmith.eval.rejudge <judge.jsonl>` 只凭日志重算 `Verdict`，不启动 Genesis、torch、Inspect Robots 或 LLM。
  重跑物理或重新调 LLM 不保证同结果，复现指的是日志复判。
- **结束方式**：`done`、`give_up`、`horizon`（时间上限）、`policy_error`（如 LLM 调用失败）四种都写终态、照常判卷；日志缺终态
  说明仿真或进程出了问题，判为环境错误，不算考生失败，重试一次，新尝试写新目录。

## 已测结果

一个实例：Inspect Robots `agent`（`7720195`）以 gpt-5.6-sol（effort low）在 `acc_f0_open` playground seed 0 上，经 11 次 LLM
调用把 die 放到 mat 上后自己调 `done`；物理判卷 success，900 个物理步零违规，certified；保存的日志离线复判，`Verdict` 与在线
逐字段相同。RoboSmith `29bec14`，单卡 MI300X（gfx942），ROCm 6.4.4 镜像加 Genesis 0.4.5。

边界：观测无图像，物体位置是特权信息，手竖直朝下、yaw 固定；这是单个实例、单个模型的一次运行，LLM 输出不保证可复现，不代表
F0 全族或其他场景的表现。

## 怎么跑

```shell
python -m pip install -e ".[eval]"     # 装钉住版本的 Inspect Robots 与 agent 插件
export ROBOSMITH_LLM_BASE_URL=... ROBOSMITH_LLM_MODEL=... ROBOSMITH_LLM_API_VERSION=... ROBOSMITH_LLM_KEY=...
python scripts/eval/run_exam.py --scenario scenarios/acceptance/f0_open.yaml --split playground --limit 1 --out <dir>
```

在 GPU 节点的 RoboSmith 镜像里跑（[Quick start](quick-start.md)）。`<dir>` 下有 `run.json`（场景、RoboSmith 与被测方版本、模型、
特权观测）、`instances.jsonl`（每次尝试一行，带 `Verdict`）、`report.txt` 与 `trials/<seed>/attempt-<n>/`（判卷日志与被测方
自己的日志）。网关变量与 agent planner 读的是同一组；LLM 端点要订阅头与 `api-version` 时，`run_exam.py` 在传给 agent 的 httpx
transport 里补上，不改 agent 代码。接别的 Inspect Robots 策略，换掉 `run_exam.py` 里构造的 policy 即可。

## 还没做

- **里程碑**：写在场景 YAML 的 `task.milestones`，按顺序，每项一个 `name` 加一个成功树叶子，另加 `held`（与持物同一判据）：

  ```yaml
  milestones:
    - {name: grasped, held: {object: die}}
    - {name: lifted, lifted: {object: die, height: 0.03}}
    - {name: "on", on: {object: die, surface: mat.top}}
  ```

  `on` 加引号：RoboSmith 的加载器把裸 `on` 读成字符串，通用的 YAML 1.1 读法读成布尔真。判法：第 i 个在它之前的都已达到、且首次
  成立的控制步记为达到；不成功的实例设最后达到的是第 k 个，一个都没达到记 `never: <第 0 个>`，第 k 个在终态仍成立记
  `never: <第 k+1 个>`，不再成立记 `lost: <第 k 个>`（抬起后掉回桌面是 `lost: lifted`）；成功的实例不记失败标签。
- **certify 也写判卷日志**：让 RoboSmith 自己的出题人运行能用同一个判卷器离线复判；试抓阶段的物理步不算里程碑进展，被拒绝或执行
  失败而跳过任务判定的运行也写终态。
- **多族报告**：按场景族汇总 `Verdict`，并列出同一批实例的 certify 结果。
- **更多接口**：流式动作模式与相机（给 VLA）、技能 / API 工具赛道。

不做：不对外部动作做执行门预检；一题只有一个被搬动的物体；不归因到责任方（never / lost 只描述失败过程）；不做真机结论。
