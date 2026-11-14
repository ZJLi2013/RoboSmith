"""One frame of a scenario instance's overview camera, after the reset.

  python scripts/acceptance/render_frame.py --scenario scenarios/acceptance/f0_open.yaml --seed 0 --renderer hiprt --out f0.png

``genesis`` is Genesis' own camera; ``hip-raster`` and ``hiprt`` (direct lighting, one path sample) render the same
pose through hipRT_render, which needs ``robosmith-hiprt.env`` sourced (``docker/README.md``). Writes the PNG and prints
one JSON line: the renderer, the camera, and the mean milliseconds of ``--frames`` frames after two warm-up frames
(the first ``bind`` of a hipRT renderer, which compiles its kernels, is reported apart as ``bind_s``).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from robotsmith.scenario.load import load_scenario  # noqa: E402
from robotsmith.scenario_runtime.runner import reset_scene  # noqa: E402
from robotsmith.sim.sim_env import SimEnv  # noqa: E402

RESOLUTION, FOVY = (640, 480), 45.0


def _hiprt(renderer, eye, lookat):
    from instinct_render.genesis import HiprtRenderer, InstinctRenderer

    kwargs = dict(eye=eye, lookat=lookat, up=(0.0, 0.0, 1.0), fovy_deg=FOVY, resolution=RESOLUTION)
    if renderer == "hiprt":
        return HiprtRenderer(**kwargs, path_samples=1)
    return InstinctRenderer(**kwargs)


def _rgb(frame):
    image = frame.detach().float().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame, np.float32)
    image = image.reshape(RESOLUTION[1], RESOLUTION[0], -1)[..., :3]
    if image.max() <= 1.5:
        image = image * 255.0
    return image.clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--renderer", choices=("genesis", "hip-raster", "hiprt"), default="genesis")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()

    import imageio
    import torch

    package = load_scenario(args.scenario, seed=args.seed)
    env = SimEnv.build(package.scene, seed=args.seed)
    reset_scene(env, 30)
    eye, lookat = tuple(package.scene.camera_position), tuple(package.scene.camera_target)

    reading = {"renderer": args.renderer, "seed": args.seed, "eye": eye, "lookat": lookat,
               "resolution": RESOLUTION, "fovy_deg": FOVY}
    if args.renderer == "genesis":
        render = lambda: env.cam_up.render()[0]  # noqa: E731
    else:
        started = time.perf_counter()
        hiprt = _hiprt(args.renderer, eye, lookat).bind(env.scene)
        reading["bind_s"] = round(time.perf_counter() - started, 2)
        render = hiprt.render
    render()
    render()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(args.frames):
        frame = render()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    reading["ms"] = round((time.perf_counter() - started) * 1000 / args.frames, 2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(args.out, _rgb(frame))
    reading["out"] = str(args.out)
    print(json.dumps(reading))


if __name__ == "__main__":
    main()
