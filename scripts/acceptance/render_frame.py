"""One frame of a scenario instance's overview camera, after the reset.

  python scripts/acceptance/render_frame.py --scenario scenarios/acceptance/f0_open.yaml --seed 0 --renderer hiprt --out f0.png

``genesis`` is Genesis' own camera; ``hiprt`` (direct lighting, one path sample) renders the same pose through
hipRT_render, which needs ``robosmith-hiprt.env`` sourced (``docker/README.md``); the default follows
``ROBOSMITH_RENDERER``. Writes the PNG and prints one JSON line: the renderer and the mean milliseconds of
``--frames`` frames after two warm-up frames (the first hipRT frame binds the scene and compiles its kernels, and is
reported apart as ``first_frame_s``).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from robotsmith.scenario.load import load_scenario  # noqa: E402
from robotsmith.scenario_runtime.runner import reset_scene  # noqa: E402
from robotsmith.sim.sim_env import RENDERERS, SimEnv  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--renderer", choices=RENDERERS, default=os.environ.get("ROBOSMITH_RENDERER") or "genesis")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()
    os.environ["ROBOSMITH_RENDERER"] = args.renderer

    import imageio

    package = load_scenario(args.scenario, seed=args.seed)
    env = SimEnv.build(package.scene, seed=args.seed)
    reset_scene(env, 30)

    started = time.perf_counter()
    env.render("up")
    reading = {"renderer": env.renderer, "seed": args.seed, "first_frame_s": round(time.perf_counter() - started, 2)}
    env.render("up")
    started = time.perf_counter()
    for _ in range(args.frames):
        frame = env.render("up")
    reading["ms"] = round((time.perf_counter() - started) * 1000 / args.frames, 2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(args.out, frame)
    reading["out"] = str(args.out)
    print(json.dumps(reading))


if __name__ == "__main__":
    main()
