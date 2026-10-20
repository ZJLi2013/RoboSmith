## Problem and outcome

<!-- State the problem and the resulting behavior. -->

## Layer

- [ ] Assets
- [ ] Scene and task authoring (CAP)
- [ ] Grasp
- [ ] Motion and execution
- [ ] Rollout and data export
- [ ] Runtime images
- [ ] Documentation only

## Evidence

<!-- Commands and their output. For behavior changes, the GPU e2e: scenario, seeds, success rate, image tag. -->

```text
python -m pytest tests -q --ignore=tests/test_sim_env.py
```

## Compatibility

<!-- CAP API, asset metadata, scenario, LeRobot layout, or dependency changes; or "None". -->

## AI use

- Tool/model: None
- Extent: None
- Human verification: None

## Checklist

- [ ] The replaced path was removed.
- [ ] CHANGELOG is updated for user-facing changes.
- [ ] New assets have a `provenance.json` with a redistributable license.
- [ ] No credentials, hostnames, personal paths, GraspGen weights, generated assets, rollout output, or local agent state.
- [ ] AI use is disclosed according to `AI_POLICY.md`.
- [ ] I can explain and maintain this change without relying on an AI transcript.
