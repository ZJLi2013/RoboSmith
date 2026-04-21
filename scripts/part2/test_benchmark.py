"""Quick smoke test for RoboSmithBenchmark — verifies import + Genesis init + 1 episode."""

from __future__ import annotations

import asyncio
import sys
import time

import numpy as np


def test_import():
    """Test that benchmark can be imported and instantiated."""
    from robotsmith.eval.benchmark import RoboSmithBenchmark
    b = RoboSmithBenchmark(tasks=["pick_cube"], seed=42)
    print("[OK] RoboSmithBenchmark created")
    print("  tasks:", b.get_tasks())
    print("  metadata:", b.get_metadata())
    return b


async def test_episode(b):
    """Run 1 episode with random actions to test the full loop."""
    task = b.get_tasks()[0]
    print(f"\n[test] Starting episode: task={task}")

    t0 = time.monotonic()
    await b.start_episode(task)
    print(f"[test] Scene built + settled in {time.monotonic() - t0:.1f}s")

    obs = await b.get_observation()
    print(f"[test] Initial obs keys: {list(obs.keys())}")
    if "images" in obs:
        for k, v in obs["images"].items():
            print(f"  image.{k}: shape={np.array(v).shape}, dtype={np.array(v).dtype}")
    if "state" in obs:
        print(f"  state: shape={np.array(obs['state']).shape}")
    if "task_description" in obs:
        print(f"  task_description: '{obs['task_description']}'")

    n_steps = 10
    for step in range(n_steps):
        action_arr = np.zeros(7, dtype=np.float32)
        action_arr[:3] = np.random.randn(3).astype(np.float32) * 0.002  # small pos delta
        action_arr[3:6] = np.random.randn(3).astype(np.float32) * 0.01  # small rot delta
        action_arr[6] = 0.04  # gripper open

        await b.apply_action({"actions": action_arr})

        done = await b.is_done()
        if done:
            print(f"[test] Episode done at step {step + 1}")
            break

    elapsed = await b.get_time()
    result = await b.get_result()
    print(f"\n[test] Episode result: {result}")
    print(f"[test] Elapsed: {elapsed:.1f}s")
    print("[OK] Episode loop completed successfully")


def main():
    print("=" * 60)
    print("RoboSmithBenchmark Smoke Test")
    print("=" * 60)

    b = test_import()
    asyncio.run(test_episode(b))

    b.cleanup()
    print("\n[OK] All tests passed!")


if __name__ == "__main__":
    main()
