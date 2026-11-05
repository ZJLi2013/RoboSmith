# 验收场景：操作链路在什么上算「能证明可解」

总体设计见 [system-design](system-design.md)，当前调用链见 [call-chain](call-chain.md)。

操作链路（技能层以下：抓取、可行性、运动、执行）的验收只用 `scenarios/acceptance/` 下的场景族。一族只变一个约束，
可解性在构造时已知，所以一次失败能直接归到被测链路，而不是归到场景、资产或判据。

## 好的验收场景满足什么

1. **一族只变一个约束**：对照族之外，每族只加一种障碍（参考 LIBERO 按一类因素分套件）。
2. **程序化生成，dev / test 分开**：布局由关系 + seed 求出；`playground` seed 调参用，`test` seed 只读数
   （参考 MotionBenchMaker 的按族生成、RoboTwin 2.0 的域随机化）。
3. **可解性已知，含负例**：正例都有一条独立于被测链路的执行见证；负例在几何上不可解。
4. **难度是一个连续参数**：每个实例记下实现的参数值，读数按参数画曲线，不只报一个均值。
5. **判据是任务级的 certified**：任务谓词成立，且实际执行的轨迹零违规接触，且失败时给出正确的标签
   （负例应判 `infeasible` 且不执行）。
6. **不被其他层污染**：被抓物是已准入的简单刚体（`die_01`，5 cm），障碍是 URDF 图元固定件，场景先过静止检查，不渲染。
7. **单个 episode 便宜**：一次抓放，十几秒仿真。

## 场景族

| 族 | 文件 | 唯一变量 | 难度参数 | 预期 |
|---|---|---|---|---|
| F0 空桌面 | `f0_open.yaml` | 无（对照） | 物体到机器人基座的距离 | 可解 |
| F1 贴墙抓取 | `f1_wall_gap.yaml` | 物体一侧有一道比它高 1 cm 的矮墙（`wall_01`），手指垂直于墙会撞墙 | 物体到墙的间隙，5–40 mm | 可解：手指平行于墙 |
| F2 越障搬运 | `f2_over_barrier.yaml` | 物体与放置垫之间有一道 20 cm 高墙（`wall_02`） | 物体到墙的距离，30–150 mm | 可解：抬过墙再放 |
| F4 罩住 | `f4_caged.yaml` | 物体被无开口的罩子（`cage_01`）罩住 | — | 不可解 |

四族的任务相同：把 die 放到垫子（`mat_01.top`）上；步骤表在 `scenarios/acceptance/teacher/`。被测的是技能层以下，
所以不让 LLM 写步骤。split：`playground` 0–9，`test` 100–119。

### `offset_from`

物体相对障碍的摆位用关系 `offset_from: {of: <固定件 id>, x: [lo, hi], y: [lo, hi]}`，由
`robotsmith/scenario/relations.py` 求解：

- 在 `of` 的局部坐标系（含它的 yaw）里对 x、y 各自均匀采样，z 由 `on: support` 决定；
- 不做足迹避让，所以物体可以放在固定件里面（F4 的罩子需要这一点）；
- 求解顺序：带 `anchor_xy` 或 `on: <id>.<target>` 的先放，再放 `offset_from` 的，最后放 `near: robot` 的；
- `of` 不是已放置的固定件（还没放、或不是 fixture）时报 `SolveError`。

## 可解性见证

`scripts/acceptance/witness.py` 不调 `skills/`、`grasp/`、`execution/`、`motion/`。它按族写死俯抓朝向（F2 转 90°，
手指平行于墙）和搬运高度（F2 的 TCP 在桌面上方 0.34 m，其余 0.20 m），用 Genesis IK 求笛卡尔路点，再带物理执行。

```text
python scripts/acceptance/witness.py --family f2_over_barrier --seeds 100 119 --out out/f2.json
```

每个物理步读一次接触。正例通过 = 全程没有违规接触且末态的任务谓词成立；遇到第一处违规接触就判不通过并停下。
违规接触按下面的规则判，这就是读数的含义：

- **机器人**碰到任何东西（桌面、固定件、die）都算违规，唯一例外是闭合到张开之间手指（`left_finger` /
  `right_finger`）碰 die。机器人自身连杆之间的接触不检查。
- **die** 碰到桌面、垫子或手指都放行，**任何阶段都放行**；碰到其他固定件（墙、罩子）算违规。
  所以见证比「die 只在起止时碰桌 / 垫」更宽：它证明 die 没碰障碍、机器人没碰任何东西，不证明 die 在搬运中没擦过桌面。

负例（F4）不执行抓放，只确认 die 静止后仍在罩内（xy 偏离罩心 < 25 mm），它的 `witness_ok` 不表示任何执行成功。

### 约束真的起作用：对照

同一见证改掉该族所需的动作，在 test seed 上读到的结果（只对这些实例和这些动作成立）：

| 族 | 对照动作 | 实例 | 结果 |
|---|---|---|---|
| F1 | 手指垂直于墙（转 90°） | 100–119 | 20/20 违规，`hand-wall` 或 `left_finger-wall` |
| F2 | 搬运高度降到 0.20 m（低于墙顶） | 100–101 | 2/2 违规，`die-barrier` |

F2 只跑了 2 个实例：穿透接触让求解器变慢，批次在第 3 个实例前超时。

## 被测链路的 certified 读数

`scripts/acceptance/certify.py` 对每个实例用 teacher 步骤跑一次 `EpisodeRun`（技能、执行门、执行器），不渲染：

```text
python scripts/acceptance/certify.py --family f1_wall_gap --split test --out out/f1_test
```

- **每个物理步都判**：`SimEnv.step()` 每步回调一次，读该步解出的接触，按正在执行的 span 用
  `robotsmith.execution.gate.certified` 判。试抓、正式回放用已验证轨迹的 span，步骤与任务判定的 settle 用末个
  已回放的 span；reset 及其静止计步不判；不属于任何阶段的物理步算违规。试抓的快照恢复不抹掉它的接触计数。
- **规则**是执行门的 `permitted` 再加一条：die 没被抓着时只能碰桌面、放置目标与机器人手指。门判的是静态位姿，
  认证判的是实际物理，没被抓的 die 碰到墙说明它被推了。和见证的差别：搬运中 die 擦到桌面也算违规。
- **certified**：正例要任务谓词成立且零违规；负例要第一步判 `infeasible` 且 reset 之后零物理步。
  其余结果分别记 `infeasible`、`violation`、`task_failed`、`init_failed`（静止检查不过）与 `error`（抛异常，重试一次，
  两次尝试都留在 `instances.jsonl`，取后一次）。Genesis 偶尔在某个物理步上挂住，Python 中断不了它：一次尝试超过
  600 s 就记为 `error` 并以退出码 3 结束进程，`--resume` 接着已写的行重试它并跑完其余 seed。
- 每实例一行：布局、静止后位置、难度参数、各阶段物理步与违规步、首个违规（阶段、span、接触对、物理步号）、
  正式帧数、交接记录、任务判定。

test seed 100–119 上的读数（commit `7707f92`，每族 20 个实例各跑一次，无重试）：

| 族 | certified | 其余 | 失败在哪 |
|---|---|---|---|
| F0 空桌面 | 19/20 | 1 `violation` | seed 108：任务谓词成立，但 pick 正式回放的第一个 transit 点读到 die–mat 接触（此时 die 在 y=0.16、垫在 y=−0.22），之后的 settle 另有 13 个违规步；成因未查 |
| F1 贴墙抓取 | 19/20 | 1 `infeasible` | 间隙最小的实例（7.7 mm）：8 个抓取候选在下探段全被门拒，接触对 die–wall；其余 19 个（8.4–40.9 mm）全部 certified |
| F2 越障搬运 | 0/20 | 20 `infeasible` | 都在 place：5 个搬运高度全部在 transit 段的同一个点被门拒，接触对 die–barrier |
| F4 罩住 | 20/20 | — | 都在 pick 判 `infeasible`，reset 之后零物理步 |

F2 的拒绝点不随搬运高度变：place 先用一段固定的 transit 在抬起高度（约 0.14 m）转基座对准垫子，搬运高度只作用于
transit 之后，20 cm 的墙在转基座时就挡住了 die。所以这一族是 place 的搬运路线缺陷，不是门或场景的问题。
单实例 45–50 s（F0、F1），F2 约 30 s，F4 约 11 s。

## 不在这一版

- 侧插放置（格子、货架）：需要俯抓以外的抓取朝向，属于给链路加能力。
- 关节体（抽屉）、导入的房间、LLM 写步骤表、渲染。
