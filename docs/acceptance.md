# 验收场景：操作链路在什么上算「能证明可解」

总体设计见 [design](design.md)，当前调用链见 [call-chain](call-chain.md)。

操作链路（技能层以下：抓取、可行性、运动、执行）的验收只用 `scenarios/acceptance/` 下的场景族。一族只变一个约束，
可解性在构造时已知，所以一次失败能直接归到被测链路，而不是归到场景、资产或判据。

## 好的验收场景满足什么

1. **一族只变一个约束**：对照族之外，每族只加一种障碍（参考 LIBERO 按一类因素分套件）。
2. **程序化生成，dev / test 分开**：布局由关系 + seed 求出；`playground` seed 调参用，`test` seed 只读数
   （参考 MotionBenchMaker 的按族生成、RoboTwin 2.0 的域随机化）。
3. **可解性已知，含负例**：正例都有一条独立于被测链路的执行见证；负例在几何上不可解。
4. **难度是一个连续参数**：每个实例记下实现的参数值，读数按参数画曲线，不只报一个均值。
5. **判据是任务级的 certified**：任务谓词成立，且实际执行的轨迹零违规接触，且失败时给出正确的标签
   （负例应零步拒绝：现有链路记 `search_exhausted`，表示声明的候选与预算内无解，不声称已证明不可解；
   见[步骤结果](manipulation.md#执行门的接口与契约)）。
6. **不被其他层污染**：被抓物是已准入的简单刚体（`die_01`，5 cm），障碍是 URDF 图元固定件，场景先过静止检查，不渲染。
7. **单个 episode 便宜**：一次抓放，不渲染。当前链路上单实例墙钟约 150–190 s（含规划，每 5 mm 一个物理步），
   一个族的 10 个 playground 实例一张卡约半小时。

## 场景族

| 族 | 文件 | 唯一变量 | 难度参数 | 预期 |
|---|---|---|---|---|
| F0 空桌面 | `f0_open.yaml` | 无（对照） | 物体到机器人基座的距离 | 可解 |
| F1 贴墙抓取 | `f1_wall_gap.yaml` | 物体一侧有一道比它高 1 cm 的矮墙（`wall_01`），手指垂直于墙会撞墙 | 物体到墙的间隙，5–40 mm | 可解：手指平行于墙 |
| F2 越障搬运 | `f2_over_barrier.yaml` | 物体与放置垫之间有一道 20 cm 高墙（`wall_02`） | 物体到墙的距离，30–150 mm | 可解：抬过墙再放 |
| F4 罩住 | `f4_caged.yaml` | 物体被无开口的罩子（`cage_01`）罩住 | — | 不可解 |
| F5 留出（已解封） | `heldout/f5_clutter.yaml` | 两道矮墙（`wall_01`，一道转 45°、一道转 90°）围着转 45° 的 die，手指垂直于近墙会在下探时撞墙 | die 到近墙的间隙，test 100–119 为 5.7–38.9 mm | 可解：手指平行于近墙 |

五族的任务相同：把 die 放到垫子（`mat_01.top`）上；plan 文件在 `plans/`（留出族在 `heldout/plans/`），
`steps` 是技能步骤，`actions` 是六动作组合，文件有 `actions` 时运行时执行它。被测的是技能层以下，所以不让 LLM 写步骤。
split：`playground` 0–9，`test` 100–119。

### plan 文件的 `actions`：六动作组合

`actions` 是运行时执行的[六动作组合](manipulation.md#六个-meta-action-与-pick--place-组合)，五族逐字相同：
候选由场景几何决定，组合里没有场景名、见证轨迹或搬运高度。写法、参数、绑定规则与 `PlanInvalid` 报错见
[plan 文件用法](../scenarios/AGENT_PROMPT.md#plan-file-optional)；只有 `steps` 的文件展开成同样的两段。

`read_plan_file` 是读 plan 文件的唯一入口（`load_scenario` 与 `load_plan` 都经它），只认 `name` / `steps` / `actions` 三个键。
它只读文件，不回写、不跨调用缓存，YAML 与 schema 错误原样抛出。F5 的 plan 文件解封后从旧键 `teacher` / `plan` 迁到新键，
seed 100–119 加载出的技能与动作逐个不变；它的 sha256 因此不再等于封存 manifest 里的值。

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

每个物理步读一次接触，包括任务判定的 settle（经 `env.on_step`）。正例通过 = 全程没有违规接触且末态的任务谓词成立；
每行记下各阶段的物理步数与第一处违规（`phase_steps`、`first_illegal`）。违规接触按 `PHASES` 逐阶段判（规则
`phase-v2`，记在每行的 `rule`）：

| 阶段 | 手指可碰 die | die 还可碰 |
|---|---|---|
| approach、descend | 否 | 桌面 |
| close、lift | 是 | 桌面 |
| carry | 是 | 无 |
| lower、open | 是 | 垫子 |
| retreat、settle | 否 | 垫子 |

- **机器人**只能用手指（`left_finger` / `right_finger`）在可碰的阶段碰 die，碰其余任何东西都算违规。
- **自碰撞**：Franka 连杆之间只放行相连的一对（父子或同父的兄弟，例如两指），其余连杆对的接触算违规。
- **die** 碰到表中以外的东西（障碍、搬运中的桌面或垫子）算违规。

负例（F4）不执行抓放，只确认 die 静止后仍在罩内（xy 偏离罩心 < 25 mm），它的 `witness_ok` 不表示任何执行成功。
见证只证明实例可解，它的路点不进 plan 文件，也不作运行时的候选。

test seed 100–119 上 `phase-v2` 的读数（commit `0b5f981`，每族 20 个实例，无重试）：

| 族 | 见证 | 难度参数实际范围 |
|---|---|---|
| F0 空桌面 | 20/20 | 到基座 0.41–0.60 m |
| F1 贴墙抓取 | 20/20 | 间隙 7.7–40.9 mm |
| F2 越障搬运 | 20/20 | 到墙 36–150 mm |
| F4 罩住 | 20/20 在罩内 | — |

正例单实例 78–93 s，F4 约 2.5 s。

### 约束真的起作用：对照

同一见证改掉该族所需的动作，在 test seed 上读到的结果（只对这些实例和这些动作成立；下表是 `phase-v2` 之前的
宽松规则下的读数）：

| 族 | 对照动作 | 实例 | 结果 |
|---|---|---|---|
| F1 | 手指垂直于墙（转 90°） | 100–119 | 20/20 违规，`hand-wall` 或 `left_finger-wall` |
| F2 | 搬运高度降到 0.20 m（低于墙顶） | 100–101 | 2/2 违规，`die-barrier` |

F2 只跑了 2 个实例：穿透接触让求解器变慢，批次在第 3 个实例前超时。

## 留出族

留出族检验链路在没调过参的约束上是否仍成立，所以它的约束与参数在解封前只给允许的读者看。

- **位置**：族定义、plan 文件、见证参数（`witness.json`）与约束和选择理由（`README.md`，封存）在
  `scenarios/acceptance/heldout/`，逐实例结果只放在实验记录的封存目录里。公共 `witness.py` 只有
  `--heldout` 机制，不含任何留出族名或参数；`certify.py` 读 `--scenario` 给出的场景文件，留出族只在解封读数时以
  `--scenario scenarios/acceptance/heldout/<name>.yaml` 显式给出。
- **冻结**：只有构造它的 task 写这些文件；test 批次在已提交的输入上运行，manifest 记下 commit 与各文件的 sha256，
  之后只读。批次失败或输入未提交时不写 manifest，修正后整批重跑。
- **改动留出文件**：封存期间只允许不改字节的迁移（如移动路径），sha256 保持与 manifest 一致；改内容等于烧毁。
- **读者**：解封前只有 `witness.py --heldout` 与该 task 的 verifier；实现与调参不读它的布局、plan 文件、见证或逐实例结果。
- **解封**：只在最终评测，且新版实现与参数都已提交；解封时记下这个 commit，评测前对照 manifest 的 sha256。
- **烧毁**：解封前被实现或调参读取或使用，这批留出作废，记录保留，另建一个未用于调参的留出族。
- **已解封**：F5 于 2026-11-10 在 `464edf4` 上解封、读了 test seed 100–109（manifest 核对一致），之后它的 plan 文件改用新键。
  之后改动链路再要泛化读数，另建一个未读过的留出族，不再用 F5。

## 被测链路的 certified 读数

`scripts/acceptance/certify.py` 对每个实例用 plan 文件加载出的技能（有 `actions` 时按它）跑一次 `EpisodeRun`（技能、执行门、执行器），不渲染：

```text
python scripts/acceptance/certify.py --scenario scenarios/acceptance/f1_wall_gap.yaml --split test --out out/f1_test
```

它读任意带 `split` 的 scenario：每个实例要挪的是 plan 里 pick 的物体，不认物体名；难度读数只对 `witness.py` 里的族给出。

- **每个物理步都判**：`SimEnv.step()` 每步回调一次，读该步解出的接触，按正在执行的 span 用
  `robotsmith.execution.gate.certified` 判。试抓、正式回放用已验证轨迹的 span，步骤与任务判定的 settle 用末个
  已回放的 span；reset 及其静止计步不判；不属于任何阶段的物理步算违规。试抓的快照恢复不抹掉它的接触计数。
- **每组 seed 一个场景**：实例之间 certify 只把 plan 里 pick 的物体移到本 seed 的起点。`scene_groups` 按「除这些物体的起点外场景相同」（物体、资产、
  yaw、关节初值、其余物体位置、桌面）把 seed 分组，每组在第一次运行时建一次场景；换组前 `gs.destroy()`，因为 Genesis 一个进程
  只允许一次 `gs.init`。公开族的固定件不随 seed 变，只建一次场景；固定件随 seed 变化的族按组各建一次
  （mat 随 seed 移动的 F0 副本，两组各一例，在 GPU 上验证过）。
- **规则**是执行门的 `permitted` 再加一条：被操作的物体没被抓着时只能碰桌面、放置目标与机器人手指。门判的是静态位姿，
  认证判的是实际物理，没被抓的物体碰到墙说明它被推了。和见证的差别：搬运中物体擦到桌面也算违规。
- **certified**：正例要任务谓词成立且零违规；负例要第一步以 `Refused` 的标签（`search_exhausted` 等）停下且 reset 之后
  零物理步。其余结果分别记停下的标签（`search_exhausted`、`execution_failed`）、`violation`、`task_failed`、`init_failed`（静止检查不过）与 `error`（抛异常，重试一次，
  两次尝试都留在 `instances.jsonl`，取后一次）。Genesis 偶尔在某个物理步上挂住，进程内中断不了它，所以超时在进程外：
  入口经 `scripts/acceptance/batch.py` 的 `supervise` 在新进程组里起工作进程，工作进程每次尝试前写边车文件
  `instances.running`（seed、attempt、开始时刻）；一次尝试超过 600 s，`supervise` 杀掉整个进程组，在还没有该尝试的行时
  补一条 `error` 行（`why` 为 `killed after 600 s`），再以 `--resume` 重启，接着已写的行重试它并跑完其余 seed。
  每个 seed 至多两次尝试；一个 log 只能有一个 `supervise`。见证（`witness.py`）与 e3 的 GPU 入口用同一机制。
  Windows 上 `supervise` 读边车文件时，工作进程对它的替换与删除会抛 `PermissionError`；`run_batch` 对这两步每 10 ms 重试、
  至多 100 次（约 1 s），仍失败才抛出。Linux 上打开中的文件可以被替换与删除，不走到重试。
- 每实例一行：布局、静止后位置、难度参数、各阶段物理步与违规步、首个违规（阶段、span、接触对、物理步号）、
  正式帧数、交接记录、任务判定。

test seed 100–119 上的读数（commit `7707f92`，每族 20 个实例各跑一次，无重试；该 commit 把门前全部被拒记为
`infeasible`，对应现在的 `search_exhausted`）：

| 族 | certified | 其余 | 失败在哪 |
|---|---|---|---|
| F0 空桌面 | 19/20 | 1 `violation` | seed 108：任务谓词成立，但 pick 正式回放的第一个 transit 点读到 die–mat 接触（此时 die 在 y=0.16、垫在 y=−0.22），之后的 settle 另有 13 个违规步；同一 commit 重跑一次为 20/20，这一例不复现，成因未查 |
| F1 贴墙抓取 | 19/20 | 1 `infeasible` | 间隙最小的实例（7.7 mm）：8 个抓取候选在下探段全被门拒，接触对 die–wall；其余 19 个（8.4–40.9 mm）全部 certified |
| F2 越障搬运 | 0/20 | 20 `infeasible` | 都在 place：5 个搬运高度全部在 transit 段的同一个点被门拒，接触对 die–barrier |
| F4 罩住 | 20/20 | — | 都在 pick 判 `infeasible`，reset 之后零物理步 |

F2 的拒绝点不随搬运高度变：place 先用一段固定的 transit 在抬起高度（约 0.14 m）转基座对准垫子，搬运高度只作用于
transit 之后，20 cm 的墙在转基座时就挡住了 die。所以这一族是 place 的搬运路线缺陷，不是门或场景的问题。
单实例 45–50 s（F0、F1），F2 约 30 s，F4 约 11 s。

#### 逐片迁移的 dev 对照

模板链路逐片替换时，每片在 playground seed 0–9 上与这一行比较（同族同 seed，只差该片的改动），test seed 留到实现冻结后。
从拆出 `move_linear` 那一片起只比 F0 / F1 / F2：F4 在每一片上都是初始化后零步拒绝，不再测，负例零执行由 CPU 测试覆盖（human 2026-11-07）。
基线在 commit `b467daf` 上读取，含 e1 / e2 的门加固、链路代码未改，四族各 10 个实例各跑一次、无重试：

| 族 | certified | 其余 | 失败在哪 |
|---|---|---|---|
| F0 空桌面 | 10/10 | — | — |
| F1 贴墙抓取 | 10/10 | — | — |
| F2 越障搬运 | 0/10 | 10 `search_exhausted` | 都在 place：转基座的 transit 段被门拒，7 个接触对 die–barrier，seed 5、6、9 为机器人–barrier；5 个搬运高度都卡在这一段 |
| F4 罩住 | 10/10 | — | 都在 pick 零步 `search_exhausted` |

F2 的 episode 不是零物理步：pick 已经执行（试抓与正式回放），停在 place 时 place 自身零物理步。F0 / F1 在 seed 0–1 上
与 c1 dev 的结果相同，门加固没有让已通过的实例变坏。之后每片与上一片比较，各片结果见
[manipulation 的结果](manipulation.md#结果)；最近一片是 F0 10/10、F1 10/10、F2 10/10（plan 文件的 `actions` 直接执行，Franka 开满重力补偿）。

### 认证口径

六动作组合链路的 certified：正例为任务谓词成立且实际执行零违规接触；
负例为支持的动作域内候选与预算搜完后零步拒绝 `search_exhausted`。`unsupported` 与环境错误不计成功。试抓、回放、保持与
settle 都计真实物理步，恢复快照不清除计数；第一次试抓之后的失败记 `execution_failed`。上面 test seed 的表是模板链路的读数，
不当作现在链路的 certified。

公开 F0 / F1 / F2 是开发用的实验台，以 playground 0–9 的结果收口（[结果](manipulation.md#结果)），不跑 test seed；
F4 不再测，负例零执行由 CPU 测试覆盖。泛化读数只在新版实现与参数都提交、冻结之后读一次留出族，只报不判：
2026-11-10 在 `464edf4` 上读了 F5 的 test seed 100–109，10/10 certified。

## 不在这一版

- 侧插放置（格子、货架）：需要俯抓以外的抓取朝向，属于给链路加能力。
- 关节体（抽屉）、导入的房间、LLM 写步骤表、渲染。
