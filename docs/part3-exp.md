# Part 3 实验记录 — Eval Engine

> 基于 vla-eval-harness 的评估引擎：RoboSmithBenchmark 实现 + VLA model 评估验证。
> 设计详见 [design.md — Part 3](design.md#part-3标准化评估引擎)。

## Experiment Summary

| Exp | Hypothesis | Status | Key Result | Conclusion |
|-----|-----------|--------|------------|------------|
| P3.1 | vla-eval + RoboSmithBenchmark 端到端跑通 | 🔄 | | |
| P3.2 | SmolVLA 在 RoboSmith benchmark 复现 ~40% pick_cube | 📋 | | |
| P3.3 | 多任务 eval (pick/place/stack) | 📋 | | |

## 环境

| 项目 | 值 |
|------|-----|
| GPU | AMD Instinct MI300X |
| ROCm | 7.1.1 |
| Genesis | 0.4.5+ |
| vla-eval | `pip install vla-eval` (PyPI) |
| Python | 3.12 |
| Docker | `rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1` |
| Node | `banff-cyxtera-s71-4.ctr.dcgpu` |

---

## P3.1: vla-eval 集成 + RoboSmithBenchmark 实现

### Phase 0 确认

- 观测完备性: observation 包含 images (up + side, 640x480 RGB) + state (9D joint positions) + task_description (语言指令)。与 Part 2 collect_data.py 数据格式一致。
- 随机化变量: cube 位置每 episode 随机化，通过 images 感知（vision-only），与 Part 2 B0 eval 一致。

### 假设

vla-eval-harness 可在本地安装（纯 Python），RoboSmithBenchmark 实现 vla-eval `Benchmark` ABC 后，
SmolVLA model server + Genesis eval scene 可端到端跑通 2 episodes（smoke test）。

### 实验方案

**Step 1: vla-eval 安装**
```bash
pip install vla-eval
vla-eval --help
```

**Step 2: 实现 RoboSmithBenchmark**

vla-eval `Benchmark` ABC 接口（async）：

| 方法 | RoboSmith 实现 |
|------|---------------|
| `get_tasks()` | 返回 TASK_PRESETS 中指定的 task 列表 |
| `start_episode(task)` | 构建 Genesis scene, 随机 cube 位置, settle |
| `apply_action(action)` | smooth + `franka.control_dofs_position` + `scene.step()` |
| `get_observation()` | render cameras + get joint positions → vla-eval Observation dict |
| `is_done()` | composable predicate 判定 (`evaluate_predicate`) |
| `get_time()` | 返回 episode wall-clock 时间 |
| `get_result()` | 返回 `{"success": bool}` |

文件: `robotsmith/eval/benchmark.py`

**Step 3: SmolVLA model server**

文件: `scripts/part3/smolvla_server.py` — 实现 vla-eval `ModelServer` ABC

**Step 4: Smoke test (2 episodes)**
```bash
# Terminal 1
vla-eval serve scripts/part3/smolvla_server.py

# Terminal 2
vla-eval run --config configs/eval/robotsmith_pick_cube.yaml --no-docker
```

### 预期

- vla-eval CLI 安装成功
- 2 episodes 无 crash，eval report JSON 生成

### 结果（本地验证）

**Step 1: vla-eval 安装** ✅
```
Successfully installed vla-eval-0.1.0
python -c "import vla_eval; print(vla_eval.__version__)"  → 0.1.0
```

**Step 2: RoboSmithBenchmark 实现** ✅

文件结构:
```
robotsmith/eval/
├── __init__.py
└── benchmark.py          # RoboSmithBenchmark(Benchmark) — 7 abstract methods 全部实现

scripts/part3/
└── smolvla_server.py     # SmolVLAServer(ModelServer) — vla-eval model server

configs/eval/
├── robotsmith_pick_cube.yaml    # 10ep eval config
├── robotsmith_smoke_test.yaml   # 2ep smoke test config
└── smolvla_server.yaml          # model server config
```

本地验证:
```
# Benchmark ABC 导入 + 实例化
from robotsmith.eval.benchmark import RoboSmithBenchmark
b = RoboSmithBenchmark(tasks=['pick_cube'])
b.get_tasks()    → [{'name': 'pick_cube'}]
b.get_metadata() → {'benchmark': 'robotsmith', 'tasks': ['pick_cube'], ...}

# vla-eval import resolution
resolve_import_string('robotsmith.eval.benchmark:RoboSmithBenchmark')  → OK

# Abstract method conformance: 7/7 implemented
# get_tasks, start_episode, apply_action, get_observation, is_done, get_time, get_result
```

**Step 3-4: Smoke test** 📋 需要远端 GPU (Genesis 依赖)

### Next Step

将代码 push 到远端 MI300X 节点，在 Docker 容器内执行:
1. `pip install vla-eval`
2. 启动 SmolVLA model server
3. 运行 `vla-eval run --config configs/eval/robotsmith_smoke_test.yaml --no-docker`

---

## Debug Tracking

| Round | Issue | Fix | Result |
|-------|-------|-----|--------|
| | | | |
