# RoboSmith Scenarios

`scenarios/` stores reviewed scenario definitions: one YAML file per scenario
(scene + task), and an optional teacher plan in `teacher/<same name>.yaml` for
scripted rollouts. The format and the agent prompt are in
[`AGENT_PROMPT.md`](AGENT_PROMPT.md); reference it explicitly when asking an
agent to write a scenario.

Lifecycle:

```text
natural language request
  -> agent writes output/<exp>/<name>.yaml with AGENT_PROMPT.md as context
  -> run_generated_scenario.py --materialize-only, then --smoke
  -> promote into scenarios/
```

Scenario files hold no positions beyond fixture anchors, no robot actions and no
code; the relation solver places objects per seed from asset geometry.
