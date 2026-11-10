# Quick start

From a fresh AMD GPU node to one certified acceptance instance: the die on an open table picked and placed on the mat
through the gated six-action chain, judged at every physics step. For what each piece does, see [design.md](design.md).

## 0. Without a GPU

The package and its unit tests run anywhere:

```shell
python -m pip install -e ".[dev]"
python -m robotsmith.assets.fetch   # asset meshes are not in git; this downloads them by build hash
python -m pytest tests -q
```

Everything below needs an AMD GPU node with ROCm and Docker.

## 1. Get the repository

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
kernels on that architecture. The commands below use the gfx942 tag; on MI350 use `robotsmith:rocm-headless`.

## 3. Certify one acceptance instance

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  -v $HOME/RoboSmith:/workspace/robotsmith robotsmith:gfx942-rocm6.4.3-genesis0.4.5 \
  bash -lc 'cd /workspace/robotsmith && python -m robotsmith.assets.fetch && \
            python scripts/acceptance/certify.py --scenario scenarios/acceptance/f0_open.yaml --split playground \
              --limit 1 --out output/f0'
```

On a shared node, pick a free GPU with `-e HIP_VISIBLE_DEVICES=<n>`. The first run takes about 6 minutes on one
MI300X, kernel compilation included. It ends with `f0_open playground: 1/1 certified`, and writes to `output/f0/`:

- `instances.jsonl`: one row per instance, with `"certified": true` when the task holds and no physics step made an
  illegal contact; otherwise the label it stopped on and the first violation;
- `summary.json`, and copies of the family and plan files it ran.

Drop `--limit 1` for all ten playground seeds, or pass another family in `scenarios/acceptance/`: `f1_wall_gap`,
`f2_over_barrier`, `f4_caged` (the negative one: certified means refused with zero steps). What each family varies and
its readings: [acceptance.md](acceptance.md). Any scenario with a `split` and a plan file in its `plans/` directory
certifies the same way, e.g. `--scenario scenarios/pick_place_block_on_mat.yaml`.

## 4. Run your own scenario

`scripts/setup/run_scenario.sh` runs a scenario through the same chain in a detached container and records a LeRobot
dataset. The curated scenarios in [`scenarios/`](../scenarios/) load on this chain but have no readings on it.

```bash
cd $HOME/RoboSmith
bash scripts/setup/run_scenario.sh                                     # MI300
IMAGE=robotsmith:rocm-headless bash scripts/setup/run_scenario.sh      # MI350
docker logs -f f1_m1
```

It runs `scenarios/pick_place_onto_supporter.yaml` by default (container `f1_m1`); set `NAME`, `SCENARIO`, `SEED`,
`EPISODES` to change the run, and `HF_CACHE` if the Hugging Face cache is not `$HOME/.cache/huggingface`. It uses GPU 0.
Outputs:

- `output/<NAME>/scenario_run_summary.json`: per-episode success and diagnostics;
- `output/<NAME>/videos/*.mp4`;
- the LeRobot dataset under `$HOME/.cache/huggingface/lerobot/local/robotsmith-scenario-<task>/`.

To render a layout without rolling out:

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  -v $HOME/RoboSmith:/workspace/robotsmith robotsmith:gfx942-rocm6.4.3-genesis0.4.5 \
  bash -lc 'cd /workspace/robotsmith && python scripts/datagen/snapshot_scenario.py --scenario scenarios/pick_place_onto_supporter.yaml'
```

To write one, copy a scenario YAML from [`scenarios/`](../scenarios/) and its plan file from `scenarios/plans/` (same
file name), change the objects, relations, success tree, and steps, and pass it with
`SCENARIO=scenarios/<your_file>.yaml`. To have an agent draft one, give it
[`scenarios/AGENT_PROMPT.md`](../scenarios/AGENT_PROMPT.md) explicitly. To validate a single asset's picks, use
`scripts/datagen/validate_asset_picks.py --assets <name>`.

## Troubleshooting

- **`Genesis already initialized`.** One process can build only one scene; run each asset or scenario in its own
  process.
- **Missing mesh files.** Build files are not in git; run `python -m robotsmith.assets.fetch`.
