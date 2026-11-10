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

- changes a public API: the scenario YAML schema or relations, plan-file steps and actions, skill names, or success predicates;
- changes an on-disk format: asset `asset.json` / `checks.jsonl` / `provenance.json`, scenario files, or the LeRobot output layout;
- adds a runtime dependency, a new grasp model, or a new motion backend;
- removes or renames a scenario, script, or asset.

Name the layer that owns the change:

| Layer | Code |
|---|---|
| Assets | `robotsmith/assets/`, `assets/` |
| Scenario authoring | `robotsmith/scenario/`, `robotsmith/scenes/`, `scenarios/` |
| Grasp | `robotsmith/grasp/` |
| Motion and execution | `robotsmith/motion/`, `robotsmith/execution/`, `robotsmith/skills/` |
| Rollout and data export | `robotsmith/rollout/`, `robotsmith/sim/`, `scripts/datagen/` |
| Runtime images | `docker/`, `scripts/setup/` |

Explain why the current owner cannot absorb the change before adding a new module.

## Development

Requirements: Python 3.10+ and git. The unit tests need no GPU or Genesis:

```shell
python -m pip install -e ".[dev]"
python -m pytest tests -q
git diff --check
```

`tests/test_sim_env.py` is skipped unless `torch` and Genesis are installed (`pip install -e ".[sim,genesis]"`). Changes to motion,
grasp, or rollout behavior also need a GPU end-to-end run in the ROCm image (`docker/README.md`); report the
scenario, seed count, success rate, and image tag in the pull request.

### What the tests cover

- **Only code production reaches.** A module that loses its last production caller is deleted with its tests:
  `tests/test_catalog.py` went with `robotsmith/assets/catalog.py` once the image-to-3D producers that called it were removed.
- **Similar assertions stay when each tests its own owner.** Each meta action's `expand` has its own file
  (`test_gripper.py`, `test_move_linear.py`); zero-frame and zero-settle cases appear both at the gate
  (`test_execution_gate.py`) and in the runner (`test_place_completion.py`, `test_world_state.py`).
- **Span labels are the ones production emits** (`approach`, `close`, `lift`, `transport`, ...). The gate judges a span by
  `holding` and `allow` and uses the label only in its rejection message, so `test_execution_gate.py` may call a free
  span `transit`.
- **`to_legacy_*` in `robotsmith/scenario/adapters.py` is the current runtime path**, not a compatibility layer; its tests stay.
- **GPU tests** (`test_gate_gpu.py`, `test_sim_queries_gpu.py`, `test_certify_gpu.py`) drive the gate, `SimEnv`'s scene
  queries and the certify contact monitor on a real Genesis scene. They run when `ROBOSMITH_GPU_TESTS` is set; add `-s`
  for the readings.
- **Shared helpers live in `tests/conftest.py`**: `ClearEnv` (`test_asset_pick_validation.py`), the `gate` fixture,
  and `pick_skill` / `place_skill` / `plan_skills` (`test_execution_gate.py`, `test_scenario_runtime.py`,
  `test_skill_primitives.py`).
- **A test module's docstring lists what that file covers, and names no other test file.** `test_asset_pick_validation.py`
  covers the asset-pick rollout: `_make_scene` / `_make_task`, the object pinned at `object_xy` before settling, and the
  pick run through an `EpisodeRun`; its docstring no longer points to a `test_cap_authoring` file that does not exist.

## Assets

Every asset's `asset.json` names its license (`source.license`, SPDX). An asset without a redistributable license
or a passing `llm_review` stays a draft: scenarios refuse it and the package does not carry it. Do not commit asset build files
(they come from the package; see `robotsmith/assets/package.py`), or rollout output under `output/`.

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
