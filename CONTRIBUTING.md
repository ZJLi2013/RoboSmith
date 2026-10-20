# Contributing

Thanks for improving RoboSmith. Read the [README](README.md) first: RoboSmith turns sim-ready assets into
scenes, expert trajectories, and LeRobot datasets, and each layer below has one owner in the code.
Participation is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## All changes go through pull requests

`main` is protected: no direct pushes, no force pushes, and no branch deletion. Every change, including the
maintainer's, lands through a pull request whose checks pass. Branch from `main`, keep one topic per pull
request, and rebase on `main` before asking for review.

## Before coding

Open an Issue first when a change:

- changes a public API: the CAP scene / task / intent authoring API, skill names, or success predicates;
- changes an on-disk format: asset `metadata.json` / `provenance.json`, scenario files, or the LeRobot output layout;
- adds a runtime dependency, a new grasp model, or a new motion backend;
- removes or renames a scenario, script, or asset.

Name the layer that owns the change:

| Layer | Code |
|---|---|
| Assets | `robotsmith/assets/`, `assets/` |
| Scene and task authoring (CAP) | `robotsmith/cap/`, `robotsmith/scenes/`, `scenarios/` |
| Grasp | `robotsmith/grasp/` |
| Motion and execution | `robotsmith/motion/`, `robotsmith/execution/`, `robotsmith/skills/` |
| Rollout and data export | `robotsmith/rollout/`, `robotsmith/sim/`, `scripts/datagen/` |
| Runtime images | `docker/`, `scripts/setup/` |

Explain why the current owner cannot absorb the change before adding a new module.

## Development

Requirements: Python 3.10+ and git. The unit tests need no GPU, Genesis, or GraspGen:

```shell
python -m pip install -e ".[dev]"
python -m pytest tests -q
git diff --check
```

`tests/test_sim_env.py` is skipped unless `torch` and Genesis are installed (`pip install -e ".[sim,genesis]"`). Changes to motion,
grasp, or rollout behavior also need a GPU end-to-end run in the ROCm image (`docker/README.md`); report the
scenario, seed count, success rate, and image tag in the pull request.

## Assets

Every asset directory needs a `provenance.json` that names its source and license. Only add assets whose license
allows redistribution. Do not commit GraspGen code or checkpoints (NVIDIA research license), generated assets
under `assets/generated/`, or rollout output under `output/`.

## What never goes into the repository

Credentials, hostnames, personal paths, private prompts or transcripts, unpublished performance baselines, and
local agent state (`.agent-loop/`, `.harness/`, `workshop/`).

## Pull requests

A pull request contains:

- the problem and the resulting behavior;
- the layer it changes;
- evidence: test output, exit codes, or GPU e2e results with their conditions;
- compatibility notes for API, file-format, or dependency changes;
- the AI disclosure required by [`AI_POLICY.md`](AI_POLICY.md).

New behavior removes the path it replaces. Commit subjects follow the existing `type(scope): summary` style and
state the resulting behavior. Keep unrelated changes in separate commits. User-facing changes get a line in
[`CHANGELOG.md`](CHANGELOG.md).

## Reporting security issues

Do not open a public Issue for vulnerabilities. Follow [`SECURITY.md`](SECURITY.md).
