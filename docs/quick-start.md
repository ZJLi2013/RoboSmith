# Quick start

From a fresh AMD GPU node to a recorded drawer episode (`open → pick → place → close`). For what each piece does,
see [design.md](design.md).

## 0. Without a GPU

The package and its unit tests run anywhere:

```shell
python -m pip install -e ".[dev]"
python -m pytest tests -q
```

Everything below needs an AMD GPU node with ROCm and Docker.

## 1. Layout on the host

`scripts/setup/run_scenario.sh` expects these directories under `$HOME` (override with `REPO`, `GRASPGEN_DIR`,
`GRASPGEN_MODELS`, `HF_CACHE`):

```text
$HOME/RoboSmith         this repository
$HOME/rocRobo           motion backend (sidecar)
$HOME/GraspGen          GraspGen source
$HOME/GraspGenModels    GraspGen checkpoints
$HOME/.cache/huggingface  LeRobot datasets are written here
```

```bash
cd $HOME
git clone https://github.com/ZJLi2013/RoboSmith.git
git clone --recurse-submodules https://github.com/ZJLi2013/rocRobo.git
git clone --depth 1 https://github.com/NVlabs/GraspGen.git
```

## 2. Build the RoboSmith image

Build from the repository root (`cd $HOME/RoboSmith`) with the image for your GPU architecture:

| GPU | Build | Tag |
|---|---|---|
| MI300 (gfx942) | `docker build -f docker/Dockerfile.gfx942 -t robotsmith:gfx942-rocm6.4.3-genesis0.4.5 .` | includes GraspGen deps |
| MI350 (gfx950) | `docker build -f docker/Dockerfile.gfx950 -t robotsmith:rocm-headless .`, then the GraspGen layer below | `robotsmith:rocm-headless-graspgen` |

gfx942 needs its own older stack (ROCm 6.4.3, Genesis 0.4.5) because newer Genesis crashes while compiling collision
kernels on that architecture. On gfx950, add the GraspGen layer:

```dockerfile
FROM robotsmith:rocm-headless
RUN python -m pip install --no-cache-dir git+https://github.com/ZJLi2013/spconv_rocm.git@rocm \
 && python -m pip install --no-cache-dir transformers==4.45.2 tensordict diffusers==0.11.1 \
      huggingface-hub==0.25.2 sharedarray
```

```bash
docker build -f Dockerfile.graspgen -t robotsmith:rocm-headless-graspgen .
```

Use `spconv_rocm`, not GraspGen's upstream `spconv-cu120`.

## 3. GraspGen weights

GraspGen is under an NVIDIA research / evaluation license and is not shipped with RoboSmith. Download the checkpoints
to the host and convert them to the `spconv_rocm` weight layout once:

```bash
IMAGE=robotsmith:gfx942-rocm6.4.3-genesis0.4.5   # or robotsmith:rocm-headless-graspgen
docker run --rm -v $HOME:/host $IMAGE bash -lc '
  python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id=\"adithyamurali/GraspGenModels\", local_dir=\"/host/GraspGenModels\")"
  cd /host/RoboSmith && python scripts/setup/_convert_spconv_ckpt.py \
    /host/GraspGenModels/checkpoints/graspgen_franka_panda_gen.pth \
    /host/GraspGenModels/checkpoints/graspgen_franka_panda_dis.pth'
```

This writes `*_rocm.pth` next to the originals; the grasp wrapper prefers them automatically. The run points
`GRASPGEN_CONFIG` at `GraspGenModels/checkpoints/graspgen_franka_panda.yml`.

## 4. Start the rocRobo sidecar

Once per machine. The image is not published; build it from the rocRobo checkout. The container must be named
`rocrobo_dev` (or set `ROCROBO_CONTAINER`) and mount rocRobo at `/rocrobo`:

```bash
cd $HOME/rocRobo && docker build -t rocrobo/pyroki:rocm7.2.4-jax0.8.2 .
docker run -d --name rocrobo_dev \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -e AMD_COMGR_NAMESPACE=1 -e HIP_VISIBLE_DEVICES=0 \
  -v $HOME/rocRobo:/rocrobo \
  rocrobo/pyroki:rocm7.2.4-jax0.8.2 sleep infinity
docker exec rocrobo_dev pip install -e /rocrobo/pyroki --no-deps
```

## 5. Run the drawer scenario

```bash
cd $HOME/RoboSmith
bash scripts/setup/run_scenario.sh                                     # MI300
IMAGE=robotsmith:rocm-headless-graspgen bash scripts/setup/run_scenario.sh   # MI350
docker logs -f f1_m1
```

The launcher starts a detached container (default name `f1_m1`; set `NAME`, `SCENARIO`, `SEED`, `EPISODES` to
change the run). It mounts the Docker socket so RoboSmith can `docker exec` into the sidecar.

Outputs:

- `output/<NAME>/scenario_run_summary.json`: per-episode success and diagnostics;
- `output/<NAME>/videos/*.mp4`;
- the LeRobot dataset under `$HOME/.cache/huggingface/lerobot/local/robotsmith-scenario-<task>/`.

To render a layout without rolling out (inside the same image, which provides Genesis):

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  -v $HOME/RoboSmith:/workspace/robotsmith robotsmith:gfx942-rocm6.4.3-genesis0.4.5 \
  bash -lc 'cd /workspace/robotsmith && python scripts/datagen/snapshot_scenario.py --scenario scenarios/pick_place_into_drawer.py'
```

## 6. Write your own scenario

Copy a file from [`scenarios/`](../scenarios/), change the scene, skill intents, and success tree, and pass it with
`SCENARIO=scenarios/<your_file>.py`. To have an agent draft one, give it
[`scenarios/AGENT_PROMPT.md`](../scenarios/AGENT_PROMPT.md) explicitly. To validate a single asset's picks, use
`scripts/datagen/validate_asset_picks.py --assets <name>`.

## Troubleshooting

- **Stuck after `starting warm serve` for ~10 minutes on the first run.** This is the JAX compile of the first
  `plan_motion`, not a hang. The compile is cached in `/rocrobo/.jax_cache` (override with `ROCROBO_JAX_CACHE`);
  later runs with the same scene geometry start in seconds. While waiting,
  `docker exec rocrobo_dev pgrep -f rocrobo.serve` should print one process. If it prints none and the run is still
  stuck, the serve died during startup; restart the run.
- **A GPU is held by a dead serve.** `docker exec rocrobo_dev pkill -9 -f python`, then rerun.
- **`collision-blind straight-line fallback` in the log.** `plan_motion` failed for that segment and it ran without
  collision checks. Treat it as a failed segment, not a success.
- **`Genesis already initialized`.** One process can build only one scene; run each asset or scenario in its own
  process.
- **Missing mesh files.** Some large meshes are not in git; copy them into `assets/objects/<name>/`
  (`visual.*`, `collision.obj`, `material.mtl`).
- **Changed rocRobo code has no effect.** The sidecar bind-mounts `/rocrobo`; restart the run so a new serve starts.
