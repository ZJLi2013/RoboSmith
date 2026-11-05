# Quick start

From a fresh AMD GPU node to a recorded supporter episode (`pick → place`, the left clip in the README). For what each
piece does, see [system-design.md](system-design.md).

## 0. Without a GPU

The package and its unit tests run anywhere:

```shell
python -m pip install -e ".[dev]"
python -m robotsmith.assets.fetch   # asset meshes are not in git; this downloads them by build hash
python -m pytest tests -q
```

Everything below needs an AMD GPU node with ROCm and Docker.

## 1. Layout on the host

`scripts/setup/run_scenario.sh` expects these directories under `$HOME` (override with `REPO`, `HF_CACHE`):

```text
$HOME/RoboSmith         this repository
$HOME/.cache/huggingface  LeRobot datasets are written here
```

```bash
cd $HOME
git clone https://github.com/ZJLi2013/RoboSmith.git
```

## 2. Build the RoboSmith image

Build from the repository root (`cd $HOME/RoboSmith`) with the image for your GPU architecture:

| GPU | Build |
|---|---|
| MI300 (gfx942) | `docker build -f docker/Dockerfile.gfx942 -t robotsmith:gfx942-rocm6.4.3-genesis0.4.5 .` |
| MI350 (gfx950) | `docker build -f docker/Dockerfile.gfx950 -t robotsmith:rocm-headless .` |

gfx942 needs its own older stack (ROCm 6.4.3, Genesis 0.4.5) because newer Genesis crashes while compiling collision
kernels on that architecture.

## 3. Run the supporter scenario

```bash
cd $HOME/RoboSmith
bash scripts/setup/run_scenario.sh                                     # MI300
IMAGE=robotsmith:rocm-headless bash scripts/setup/run_scenario.sh            # MI350
docker logs -f f1_m1
```

The launcher starts a detached container (default name `f1_m1`; set `NAME`, `SCENARIO`, `SEED`, `EPISODES` to
change the run).

Outputs:

- `output/<NAME>/scenario_run_summary.json`: per-episode success and diagnostics;
- `output/<NAME>/videos/*.mp4`;
- the LeRobot dataset under `$HOME/.cache/huggingface/lerobot/local/robotsmith-scenario-<task>/`.

To render a layout without rolling out (inside the same image, which provides Genesis):

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  -v $HOME/RoboSmith:/workspace/robotsmith robotsmith:gfx942-rocm6.4.3-genesis0.4.5 \
  bash -lc 'cd /workspace/robotsmith && python scripts/datagen/snapshot_scenario.py --scenario scenarios/pick_place_onto_supporter.yaml'
```

## 4. Write your own scenario

Copy a scenario YAML from [`scenarios/`](../scenarios/) and its teacher plan from `scenarios/teacher/` (same file
name), change the objects, relations, success tree, and steps, and pass it with
`SCENARIO=scenarios/<your_file>.yaml`. To have an agent draft one, give it
[`scenarios/AGENT_PROMPT.md`](../scenarios/AGENT_PROMPT.md) explicitly. To validate a single asset's picks, use
`scripts/datagen/validate_asset_picks.py --assets <name>`.

## Troubleshooting

- **`Genesis already initialized`.** One process can build only one scene; run each asset or scenario in its own
  process.
- **Missing mesh files.** Build files are not in git; run `python -m robotsmith.assets.fetch`.
