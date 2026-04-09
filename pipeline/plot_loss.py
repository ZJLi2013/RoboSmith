"""
RoboSmith pipeline — training loss visualization.

Plot SmolVLA training loss curve from train_metrics.json.

Usage:
  python plot_loss.py \
    --metrics /path/to/train_metrics.json \
    --summary /path/to/train_summary.json \
    --out /path/to/loss_curve.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(arr, kernel, mode="same")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training loss curve from train_metrics.json")
    parser.add_argument("--metrics", type=Path, required=True, help="Path to train_metrics.json")
    parser.add_argument("--summary", type=Path, default=None, help="Optional path to train_summary.json")
    parser.add_argument("--out", type=Path, required=True, help="Output png path")
    parser.add_argument("--smooth-window", type=int, default=51, help="Moving average window for smoothed curve")
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if not metrics:
        raise ValueError(f"Empty metrics file: {args.metrics}")

    steps = np.array([m["step"] for m in metrics], dtype=np.int32)
    loss = np.array([m["loss"] for m in metrics], dtype=np.float64)
    grad_norm = np.array([m.get("grad_norm", np.nan) for m in metrics], dtype=np.float64)
    smooth_loss = moving_average(loss, args.smooth_window)

    title = "SmolVLA Training Loss"
    if args.summary and args.summary.exists():
        s = json.loads(args.summary.read_text(encoding="utf-8"))
        title = (
            f"{s.get('dataset_id', 'dataset')} | "
            f"steps={s.get('n_steps', len(steps))} | "
            f"loss {s.get('loss_start', loss[0]):.3f} -> {s.get('loss_end', loss[-1]):.3f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.plot(steps, loss, linewidth=1.0, alpha=0.35, label="loss (raw)")
    ax1.plot(steps, smooth_loss, linewidth=2.0, label=f"loss (ma{args.smooth_window})")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.plot(steps, grad_norm, color="tab:orange", linewidth=1.0, alpha=0.35, label="grad_norm")
    ax2.set_ylabel("grad_norm")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"[ok] saved: {args.out}")


if __name__ == "__main__":
    main()
