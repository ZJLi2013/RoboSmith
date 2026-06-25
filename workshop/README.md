# Interaction-Ready Synthetic Data — Live Workshop (CDNA3 / gfx942)

**Real2Sim for Embodied Data Generation** — *Generating Interaction-Ready Data for
Long-Horizon Articulated Manipulation*.

现场跑的 Jupyter notebook，单一硬件路径 **CDNA3（MI300/MI325, gfx942）**。范围**只到
interaction-ready 数据生成**：以长程铰接体序列 `open→pick→place→close`（抽屉任务）为载体，
现场用 CAP 声明任务 → rocRobo 无碰撞规划 → 产出带交互语义的 LeRobot 数据集。**不做** train/eval。

---

## 镜像环境（两个 workshop 镜像，全部 self-contained）

| 角色 | 镜像 | Dockerfile |
|---|---|---|
| RoboSmith 运行时 + repo + assets + GraspGen 权重 + notebook | `robotsmith:workshop-gfx942` | `workshop/Dockerfile` |
| rocRobo 无碰撞规划 sidecar（rocRobo 源码烤进镜像） | `rocrobo:workshop-gfx942` | `workshop/Dockerfile.rocrobo` |

> 两个 workshop 镜像把 **repo / 资产 / GraspGen 源码 + 权重 / rocRobo 源码 / 对比视频 / 全部运行时
> 配置**都烤进镜像——运行时**不挂任何 code/data local path**（部署到 cloud 时不假定主机路径一致）。
> 唯一保留的是 GPU 设备和 `docker.sock`（固定众所周知路径，跨主机一致）。
>
> base 运行时镜像 `robotsmith:gfx942-rocm6.4.3-genesis0.4.5`（`docker/Dockerfile.gfx942`，
> 固定 ROCm 6.4.3 + Genesis 0.4.5 旧栈以避开 gfx942 的 Genesis 1.0 碰撞 kernel segfault）
> **保持不变**；workshop 镜像在它之上分层，不覆盖它。

---

## Teacher / Admin Setup

学生只跑 notebook。讲师/集群管理员在一台 gfx942 机器上**构建两个 self-contained 镜像**，
之后推到 registry / 部署到 cloud，运行时零挂载。

### 1. 构建 base 运行时镜像（若 registry 还没有）

```bash
# RoboSmith repo 根目录执行
docker build -f docker/Dockerfile.gfx942 \
  -t robotsmith:gfx942-rocm6.4.3-genesis0.4.5 .
```

### 2. 构建 RoboSmith workshop 镜像（repo + assets + GraspGen 权重全烤进去）

BuildKit 命名 build-context 把三个 sibling repo 拉进镜像，**没有任何运行时挂载**。
`GraspGenModels` 需已包含 spconv_rocm 转换后的 `*_rocm.pth`（`scripts/setup/_convert_spconv_ckpt.py`），
且 `graspgen_franka_panda.yml` 指向它们——与 runtime 契约一致。

```bash
# RoboSmith repo 根目录执行；按你的 checkout 调整三个路径
DOCKER_BUILDKIT=1 docker build -f workshop/Dockerfile \
  -t robotsmith:workshop-gfx942 \
  --build-context graspgen=/home/zhengjli/robot/GraspGen \
  --build-context graspgenmodels=/home/zhengjli/robot/GraspGenModels \
  --build-context rocrobo=/home/zhengjli/robot/rocRobo \
  .
```

### 3. 构建 rocRobo sidecar 镜像（源码烤进去）

```bash
# build context = rocRobo checkout；-f 指向 RoboSmith 里的 Dockerfile
DOCKER_BUILDKIT=1 docker build -f workshop/Dockerfile.rocrobo \
  -t rocrobo:workshop-gfx942 \
  /home/zhengjli/robot/rocRobo
```

### 4. 部署运行（两条 docker run，无 code/data 挂载）

```bash
# 4a. rocRobo sidecar（源码已在镜像里，无 -v）
docker rm -f rocrobo_dev 2>/dev/null || true
docker run -d --name rocrobo_dev \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -e AMD_COMGR_NAMESPACE=1 -e HIP_VISIBLE_DEVICES=0 \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  rocrobo:workshop-gfx942 sleep infinity

# warmup smoke（触发 jax 首次加载）
docker exec -i -w /rocrobo \
  -e PYTHONPATH=/rocrobo/rocRobo/core \
  -e ROCROBO_ASSETS=/rocrobo/pyroki/examples/assets \
  rocrobo_dev python -u -c "import rocrobo; print('rocrobo_ok')"

# 4b. RoboSmith + Jupyter（学生入口）。一切配置已烤进镜像，
#     只挂 docker.sock 让 notebook 能 docker exec 进 rocrobo_dev。
docker rm -f workshop_drawer 2>/dev/null || true
docker run -d --name workshop_drawer -p 8888:8888 \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc=host --shm-size=24g \
  -e HIP_VISIBLE_DEVICES=0 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  robotsmith:workshop-gfx942
```

> `ROCROBO_BACKEND=1` 等所有 env 已在 `workshop/Dockerfile` 里 `ENV` 写死；notebook 容器经
> `docker.sock` `docker exec` 进 `rocrobo_dev` 起 serve（`run_generated_scenario.py` 已封装握手）。
> 学生侧零配置，§4 的 `videos/rocrobo_compare.mp4` 也已烤进镜像。
>
> **云上注意**：sidecar 跨容器 `docker exec` 依赖 `docker.sock`；若部署环境禁止容器访问
> docker daemon，需由编排层负责把两个容器放到同主机并放通 socket（这是双栈 jax/torch 必须分容器
> 带来的约束，不是路径假设问题）。

---

## Student Quick Start

1. 打开 `workshop_cdna3.ipynb`
2. 按 cell 顺序执行——镜像已预装一切，产物落 `output/`

notebook 段落：环境探测 → Hook → Real2Sim 资产 → CAP → rocRobo → 现场生成 →
打开产出 → 改 CAP 重生成 → 小结。

---

## 目录组织

```
workshop/
├── README.md              ← 本文件（CDNA3 单路径）
├── Dockerfile             ← ★ RoboSmith workshop 镜像（self-contained，FROM base gfx942）
├── Dockerfile.rocrobo     ← ★ rocRobo sidecar workshop 镜像（源码烤进去）
├── workshop_cdna3.ipynb   ← ★ 现场跑的 notebook（gen-only）
├── images/                ← 预生成截图（notebook inspect 兜底素材）
└── videos/                ← rocRobo 对比 / episode 视频（构建时烤进镜像，repo 内只留 .gitkeep）
```

> base 运行时镜像 `docker/Dockerfile.gfx942` 保持通用、不被覆盖；workshop 镜像在它之上分层，
> 复用 `scripts/datagen/run_generated_scenario.py`、`scenarios/*.py`，不在 workshop/ 下重复造逻辑。
