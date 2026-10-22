"""Run the asset reviewer on the calibration cases and compare each verdict with the expected one.

Run after changing REVIEW.md, the review package or the readings. Needs Blender, a simulator runner
and a reviewer command; the reviewer runs in each package directory with the prompt on stdin.

Usage:
  python scripts/assets/review_calibration.py --reviewer "<cmd>" --sim-runner <run_sim.sh> --work <dir>
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robotsmith.assets.review import calibration  # noqa: E402
from robotsmith.assets.review.package import build_package, review  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reviewer", required=True, help="command run in each package directory")
    ap.add_argument("--sim-runner", type=Path, required=True, help="script the reviewer calls as ./run_sim.sh")
    ap.add_argument("--blender", default="bash run_blender.sh", help="command that runs a Blender script")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--cases", nargs="*", help="case ids (default: all)")
    ap.add_argument("--package-only", action="store_true")
    ap.add_argument("--out", type=Path, help="write the runs as JSON")
    args = ap.parse_args()

    cases = [c for c in calibration.load_cases() if not args.cases or c["id"] in args.cases]
    pkgs = {}
    for i, case in enumerate(cases, 1):
        # Neutral names so the reviewer cannot read the expected verdict off the package.
        asset_dir, renders = calibration.prepare(case, args.work / f"build_{i}", shlex.split(args.blender))
        request = (calibration.HERE / case["request"]).read_text(encoding="utf-8")
        pkgs[case["id"]] = build_package(asset_dir, request, renders, args.work / f"sample_{i}")
        print(f"[calibration] {case['id']}: {pkgs[case['id']]}", flush=True)
    if args.package_only:
        return

    jobs = []
    for case in cases:
        for r in range(1, args.repeat + 1):
            run_pkg = pkgs[case["id"]].with_name(f"{pkgs[case['id']].name}_run{r}")
            if run_pkg.exists():
                shutil.rmtree(run_pkg)
            shutil.copytree(pkgs[case["id"]], run_pkg)
            jobs.append((case, r, run_pkg))
    reviewer = shlex.split(args.reviewer)
    with ThreadPoolExecutor(args.jobs) as ex:
        verdicts = list(ex.map(lambda j: review(j[2], reviewer, args.sim_runner), jobs))

    runs, mismatches = [], 0
    for (case, r, run_pkg), rv in zip(jobs, verdicts):
        ok = rv["decision"] == case["expect"]
        mismatches += not ok
        runs.append({"case": case["id"], "run": r, "expect": case["expect"], "decision": rv["decision"],
                     "failed_claims": [c for c in rv["claims"] if c.get("verdict") == "fail"],
                     "package": str(run_pkg)})
        print(f"[calibration] {case['id']} run {r}: expect {case['expect']}, got {rv['decision']}"
              f"{'' if ok else '  MISMATCH'}", flush=True)
    if args.out:
        args.out.write_text(json.dumps({"runs": runs}, indent=2, ensure_ascii=False), encoding="utf-8")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
