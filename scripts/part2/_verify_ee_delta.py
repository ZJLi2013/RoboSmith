"""Verify EE delta dataset shapes and values."""
import numpy as np

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("local/ee-delta-test")
print("num_episodes:", ds.meta.total_episodes)
print("num_frames:", ds.meta.total_frames)

f0 = ds[0]
print("\nFrame 0:")
print("  state shape:", f0["observation.state"].shape)
print("  action shape:", f0["action"].shape)
print("  state values:", f0["observation.state"].numpy())
print("  action values:", f0["action"].numpy())

f50 = ds[50]
print("\nFrame 50 (mid-trajectory):")
print("  state:", f50["observation.state"].numpy())
print("  action:", f50["action"].numpy())

action_norms = []
for i in range(min(135, len(ds))):
    a = ds[i]["action"].numpy()
    action_norms.append(np.linalg.norm(a[:3]))

print(f"\nAction pos-delta norms: min={min(action_norms):.6f}, max={max(action_norms):.6f}, mean={np.mean(action_norms):.6f}")
print("Sanity: EE deltas should be small (< 0.05m per step)")
