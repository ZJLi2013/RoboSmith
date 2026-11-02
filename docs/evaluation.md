# 评测：被测方跑任务，判定并归因失败

总体设计见 [system-design](system-design.md)。

## 要保证什么

被测方（agentic 系统或 VLA）经一个不绑框架的接口跑任务；判定读仿真状态、对照被测方看不到的成功树；每个失败说出哪个里程碑**从未达到**（never）
或**得而复失**（lost）。MVP 报告样例：

```text
RoboSmith eval  scene: <your desk>  evaluatee: <agentic baseline>
task                          certified  success  failures
put the cup on the tray       10/10      6/10     never: grasped ×3 · lost: on ×1
```

结论只在孪生里成立；没做过孪生与真机的对照，不作真机结论。

## 现在有什么

- 判定：谓词读仿真状态，任务与每一步共用，判定所用的状态整份保存、可离线复判；失败列出没成立的条件。
- 两种角色：证明可解（看得到成功树）与基线被测方（看不到）。

## 问题

- 没有被测方接口：只能跑 RoboSmith 自己的技能。
- 运行时绑着数据录制（每个控制步写数据集），输出状态还叫「数据生成成功 / 失败」。
- 只有「哪些条件没成立」，没有 never / lost 归因，没有报告。
- 渲染在 MI300X 上是 CPU 光栅：Genesis 物理在 GPU，渲染经光栅器落到 Mesa llvmpipe，约 28 s / episode
  （320 px 比 640 px 快 2–4×），没有光追，还不适合 VLA。Instinct（CDNA）没有光追硬件，
  [HIP RT](https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT) 在其上只能软件遍历；Genesis 光追器未在 MI300X 上试过。
  一个进程只能跑一个场景。

## 设计

- 录制改为可选的观察者，不在执行循环里。
- 接口第一版只接技能级：被测方给步骤，RoboSmith 经操作层的执行门执行；第一个被测方就是自带的基线 agent。之后再接动作级（VLA）。
- 归因：成功树的叶子按顺序当里程碑；按时间线判 never（从未成立）或 lost（成立过、终态丢失），先用合成状态序列钉住各种情形。
- 报告：按场景、任务、被测方汇总 certified / success / never / lost。

## 已定 / 不做

- 接口不绑任何框架（vla-eval、RPent 都只是适配目标）；记忆、感知、VLA 需要时接外部框架（如 RPent）。
- 不做标准 benchmark 套件、不训练策略、不做 sim-to-real 研究。

## 下一步

操作层可证明之后再开；提前探：MI300X 上 Genesis 能否 GPU 渲染。
