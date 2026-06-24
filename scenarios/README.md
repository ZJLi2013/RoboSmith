# RoboSmith Scenarios

`scenarios/` stores stable, reviewable scenario definitions that can be generated
by an agent and then promoted after validation.

For public agent-based authoring, use `scenarios/AGENT_PROMPT.md` as the
explicit prompt contract. Do not rely on private chat history or hidden helper
knowledge when asking an agent to generate a scenario.

A scenario is higher level than CAP:

```text
scenario = scene + task + skill intents + metadata
```

Current scenarios may be authored with `robotsmith.cap` primitives, but this
directory is intentionally outside `robotsmith/cap/` so scenario organization is
not tied to one authoring language.

Recommended lifecycle:

```text
natural language request
  -> provide scenarios/AGENT_PROMPT.md as prompt context
  -> agent writes output/exp*/generated_scenario.py
  -> validate / lower / smoke run
  -> promote stable definition into scenarios/
```

New Cursor chats should reference the prompt contract explicitly, for example:

```text
Use @scenarios/AGENT_PROMPT.md to generate a RoboSmith scenario candidate.

Task request:
Move the die to the goal area while leaving the mug and apple as distractors.
```

Do not assume an agent will automatically discover `scenarios/AGENT_PROMPT.md`
from the task sentence alone. Reference the prompt contract explicitly when
asking an agent to generate or validate a scenario.

Do not put runtime code here. Scenario files should not call Genesis directly,
write waypoints, control grippers, or instantiate recorders. They should only
define reusable scenario artifacts that can be validated and lowered by the
normal RoboSmith adapters.
