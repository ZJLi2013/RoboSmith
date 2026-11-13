"""An external policy on a scenario's instances, judged from the physics.

  python scripts/eval/run_exam.py --scenario scenarios/acceptance/f0_open.yaml --split playground --limit 1 \
      --out output/eval/f0_agent

Each instance is one attempt of Inspect Robots' ``agent`` policy, as published, on ``RoboSmithEmbodiment``. The
LLM endpoint is ``ROBOSMITH_LLM_BASE_URL`` with ``ROBOSMITH_LLM_MODEL`` (the variables the agent planner reads);
the policy gets an httpx transport that adds the gateway's subscription header (``ROBOSMITH_LLM_KEY``) and
``api-version`` (``ROBOSMITH_LLM_API_VERSION``), and sends no bearer token. Writes ``run.json``, the policy's own
logs under ``trials/<seed>/attempt-<n>/inspect/``, the judge log beside them, ``instances.jsonl`` (one row per
attempt, its verdict judged from that attempt's log) and ``report.txt``.
"""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "acceptance"))
from batch import run_batch, supervise  # noqa: E402

ATTEMPT_TIMEOUT_S = 1800
NO_BEARER = "ROBOSMITH_EVAL_NO_BEARER"


def evaluatee_identity() -> dict:
    out = {"policy": "agent"}
    for dist in ("inspect-robots", "inspect-robots-agent"):
        meta = importlib.metadata.distribution(dist)
        out[dist] = meta.version
        direct = meta.read_text("direct_url.json")
        if direct and dist == "inspect-robots":
            out["commit"] = json.loads(direct).get("vcs_info", {}).get("commit_id")
    return out


def ending(record) -> str:
    if record.termination_reason in ("done", "give_up"):
        return record.termination_reason
    return "horizon" if record.truncated else "policy_error"


def gateway_transport():
    import httpx

    class Gateway(httpx.HTTPTransport):
        def handle_request(self, request):
            request.headers["Ocp-Apim-Subscription-Key"] = os.environ["ROBOSMITH_LLM_KEY"]
            request.url = request.url.copy_merge_params({"api-version": os.environ["ROBOSMITH_LLM_API_VERSION"]})
            return super().handle_request(request)

    return Gateway()


def run_attempt(args, seed: int, attempt: int) -> dict:
    from inspect_robots import eval as inspect_eval
    from inspect_robots.scene import Scene
    from inspect_robots.task import Task
    from inspect_robots_agent import LLMAgentPolicy

    from robotsmith.eval.embodiment import RoboSmithEmbodiment
    from robotsmith.eval.judge import judge

    trial = args.out / "trials" / str(seed) / f"attempt-{attempt}"
    log_path = trial / "judge.jsonl"
    embodiment = RoboSmithEmbodiment(args.scenario, seed=seed, attempt=attempt, log_path=log_path)
    policy = LLMAgentPolicy(model=os.environ["ROBOSMITH_LLM_MODEL"], base_url=os.environ["ROBOSMITH_LLM_BASE_URL"],
                            api_key_env=NO_BEARER, wire="responses", effort=args.effort,
                            max_llm_calls=args.max_llm_calls, transport=gateway_transport())
    task = Task(name=embodiment.scenario.task.name, scorer="episode_length", max_seconds=args.max_seconds,
                scenes=[Scene(id=f"seed{seed}", instruction=embodiment.scenario.task.instruction, init_seed=seed)])
    seen = {}
    try:
        inspect_eval(task, policy, embodiment, log_dir=str(trial / "inspect"), seed=seed,
                     before_scoring=lambda record, scene: seen.update(ended=ending(record)))
    finally:
        embodiment.close(seen.get("ended", "policy_error"))
    verdict = judge(log_path)
    llm_calls = sum(1 for f in (trial / "inspect").rglob("calls.jsonl") for _ in f.open(encoding="utf-8"))
    return {"outcome": "certified" if verdict.certified else "env_error" if verdict.env_error else "not_certified",
            "ended": verdict.ended, "success": verdict.success, "certified": verdict.certified,
            "violating_steps": verdict.violating_steps,
            "monitor_violating_steps": embodiment.monitor.violating if embodiment.monitor else None,
            "llm_calls": llm_calls, "verdict": verdict.to_dict()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=Path, required=True)
    ap.add_argument("--split", choices=["playground", "test"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-llm-calls", type=int, default=40)
    ap.add_argument("--max-seconds", type=float, default=240.0, help="simulated seconds per attempt")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--resume", action="store_true", help="continue the rows in --out")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    log = args.out / "instances.jsonl"
    if not args.worker:
        args.out.mkdir(parents=True, exist_ok=True)
        code = supervise([sys.executable, __file__, *sys.argv[1:], "--worker"], log, timeout=ATTEMPT_TIMEOUT_S,
                         failed={"certified": False})
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
        lines = [f"{args.scenario.stem} {args.split}: agent ({os.environ.get('ROBOSMITH_LLM_MODEL')}), "
                 "privileged observation object_pos.*"]
        lines += [f"seed {r['seed']} attempt {r['attempt']}: success={r.get('success')} "
                  f"violating_steps={r.get('violating_steps')} ended={r.get('ended')} certified={r.get('certified')} "
                  f"llm_calls={r.get('llm_calls')} outcome={r['outcome']}" for r in rows]
        (args.out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        sys.exit(code)

    lo, hi = yaml.safe_load(args.scenario.read_text(encoding="utf-8"))["split"][args.split]
    seeds = list(range(lo, hi + 1))[:args.limit]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    (args.out / "run.json").write_text(json.dumps({
        "scenario": args.scenario.as_posix(), "split": args.split, "seeds": seeds, "robosmith_commit": commit,
        "evaluatee": evaluatee_identity(), "model": os.environ.get("ROBOSMITH_LLM_MODEL"), "effort": args.effort,
        "max_llm_calls": args.max_llm_calls, "max_seconds": args.max_seconds,
        "privileged_observation": ["object_pos.*"]}, indent=1) + "\n", encoding="utf-8")
    prior = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if args.resume else []
    with log.open("a" if args.resume else "w", encoding="utf-8") as fh:
        run_batch(seeds, lambda seed, attempt: run_attempt(args, seed, attempt), fh, prior=prior,
                  failed={"certified": False}, show=("seed", "attempt", "outcome", "ended", "llm_calls", "seconds"),
                  running=log.with_suffix(".running"))


if __name__ == "__main__":
    main()
