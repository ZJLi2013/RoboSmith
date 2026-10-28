"""Re-trial the diagnosis agent's repair once from the failed run's start state; if it lifts, write it to the asset.

python -m robotsmith.grasp.diagnostics.writeback <context.json> <answer> [--by agent:<model>] [--assets DIR]

``answer`` is the agent's final message: one JSON object whose ``repair`` is the candidate to write. Answers without
a repair (reports) are left alone.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

from robotsmith.assets.package import DEFAULT_ROOT
from robotsmith.grasp import annotations
from robotsmith.grasp.diagnostics.tool import try_candidates

LIFT_M = 0.02


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("context", type=Path)
    ap.add_argument("answer", type=Path)
    ap.add_argument("--by", default="agent")
    ap.add_argument("--assets", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()
    context = json.loads(args.context.read_text(encoding="utf-8"))
    m = re.search(r"\{.*\}", args.answer.read_text(encoding="utf-8"), re.S)
    repair = json.loads(m.group(0)).get("repair") if m else None
    if not repair:
        print(json.dumps({"written": False, "reason": "answer has no repair"}))
        return
    with tempfile.TemporaryDirectory() as tmp:
        out = try_candidates(context, [repair], Path(tmp))
    trial, label = out["results"][0] if out["results"] else {}, out["label"]
    lift = trial.get("lift_m") or 0.0
    if not (label.get("success") and lift > LIFT_M):
        print(json.dumps({"written": False, "reason": "recheck did not lift", "recheck": trial}))
        return
    annotations.write_grasp(args.assets / context["asset"], repair, label["object_initial_pos"],
                            label["object_quat_wxyz"], lift_m=lift, by=args.by,
                            run={"object_xy": context["object_xy"], "seed": context["seed"]})
    print(json.dumps({"written": True, "recheck": trial}))


if __name__ == "__main__":
    main()
