"""
RoboSmith pipeline — SmolVLA post-training.

Fine-tunes lerobot/smolvla_base on a local LeRobot dataset.
Logs per-step metrics to JSON for analysis.

Usage:
  python scripts/part2/train_smolvla.py --dataset-id local/franka-pick-vision-100ep --n-steps 2000
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
import sys
import time
from pathlib import Path

import torch

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def make_delta_timestamps(delta_indices, fps: int):
    if delta_indices is None:
        return [0.0]
    return [i / fps for i in delta_indices]


def _sample_to_int(sample, key: str):
    value = sample.get(key)
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def build_trimmed_indices(dataset, trim_first_n_frames: int):
    """Return dataset indices after dropping first N frames of each episode."""
    if trim_first_n_frames <= 0:
        return list(range(len(dataset)))

    per_episode_counts = defaultdict(int)
    valid_indices = []
    n_missing_episode_id = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        ep = _sample_to_int(sample, "episode_index")
        frame = _sample_to_int(sample, "frame_index")
        if frame is None:
            frame = _sample_to_int(sample, "index_in_episode")

        if ep is None:
            # Fallback to a single implicit episode when metadata is unavailable.
            ep = -1
            n_missing_episode_id += 1

        if frame is None:
            frame = per_episode_counts[ep]
            per_episode_counts[ep] += 1
        else:
            per_episode_counts[ep] = max(per_episode_counts[ep], frame + 1)

        if frame >= trim_first_n_frames:
            valid_indices.append(idx)

    if n_missing_episode_id > 0:
        print(
            f"[train][warn] {n_missing_episode_id} samples missing episode_index; "
            "trim fallback treated them as a single episode."
        )
    return valid_indices


def main():
    ap = argparse.ArgumentParser(description="SmolVLA post-train on Genesis pick data")
    ap.add_argument("--dataset-id", default="local/so101-genesis-pick")
    ap.add_argument("--dataset-root", default=None,
                    help="Local root directory for the dataset (bypasses HF Hub download)")
    ap.add_argument("--pretrained", default="lerobot/smolvla_base")
    ap.add_argument("--n-steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None, help="Override learning rate")
    ap.add_argument("--save-dir", default="outputs/smolvla_pick_phase1")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Dataloader workers. Use 0 to avoid occasional video decoder worker crashes.",
    )
    ap.add_argument(
        "--episodes-json",
        default=None,
        help=(
            "Optional path to episode selection json. Accepts either a list [0,1,...] "
            "or dict with key --episodes-key."
        ),
    )
    ap.add_argument("--episodes-key", default="success_episode_ids")
    ap.add_argument(
        "--trim-first-n-frames",
        type=int,
        default=0,
        help="Drop first N frames of each episode before training (0 disables).",
    )
    ap.add_argument(
        "--video-backend",
        default=None,
        help="Video backend for LeRobot dataset (e.g. 'pyav' to avoid torchcodec CUDA dep).",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU[{i}]: {props.name}  VRAM: {props.total_memory / 1024**3:.1f} GB")

    try:
        from lerobot.configs.types import FeatureType
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.common.datasets.utils import dataset_to_policy_features
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
    except ImportError:
        from lerobot.configs.types import FeatureType
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.datasets.utils import dataset_to_policy_features
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors

    # ---- dataset ----
    ds_root = args.dataset_root
    print(f"\n[train] loading dataset: {args.dataset_id}" + (f" (root={ds_root})" if ds_root else ""))
    dataset_metadata = LeRobotDatasetMetadata(args.dataset_id, root=ds_root)
    print(f"  total_frames: {dataset_metadata.total_frames}")
    print(f"  episodes: {dataset_metadata.total_episodes}")
    print(f"  fps: {dataset_metadata.fps}")

    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}
    print(f"  input_features: {list(input_features.keys())}")
    print(f"  output_features: {list(output_features.keys())}")

    # ---- policy ----
    print(f"\n[train] loading SmolVLA from {args.pretrained}")
    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=50,
        n_action_steps=50,
        freeze_vision_encoder=True,
        train_expert_only=True,
        train_state_proj=True,
    )

    policy = SmolVLAPolicy.from_pretrained(args.pretrained, config=cfg, strict=False)
    policy.train()
    policy.to(device)

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  total params: {total_params:,} (~{total_params/1e6:.0f}M)")
    print(f"  trainable: {trainable_params:,} (~{trainable_params/1e6:.1f}M)")
    print(f"  frozen: {total_params - trainable_params:,}")

    preprocessor, postprocessor = make_pre_post_processors(
        cfg, dataset_stats=dataset_metadata.stats,
    )

    # ---- dataloader ----
    fps = dataset_metadata.fps
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, fps),
    }
    for img_key in cfg.image_features:
        delta_timestamps[img_key] = make_delta_timestamps(cfg.observation_delta_indices, fps)
    delta_timestamps["observation.state"] = make_delta_timestamps(cfg.observation_delta_indices, fps)

    selected_episodes = None
    if args.episodes_json:
        ep_obj = json.loads(Path(args.episodes_json).read_text(encoding="utf-8"))
        if isinstance(ep_obj, dict):
            selected_episodes = ep_obj.get(args.episodes_key, [])
            selected_episodes = [int(x) for x in selected_episodes]
        elif isinstance(ep_obj, list):
            if ep_obj and isinstance(ep_obj[0], dict):
                selected_episodes = [
                    int(x["episode_index"]) for x in ep_obj
                    if x.get("success", True)
                ]
            else:
                selected_episodes = [int(x) for x in ep_obj]
        else:
            raise ValueError("--episodes-json must be a list or dict")
        if not selected_episodes:
            raise ValueError(f"No episodes selected from {args.episodes_json}")
        print(f"  selected episodes ({len(selected_episodes)}): {selected_episodes[:10]}...")

    ds_kwargs = dict(
        episodes=selected_episodes,
        delta_timestamps=delta_timestamps,
    )
    if ds_root:
        ds_kwargs["root"] = ds_root
    if args.video_backend:
        ds_kwargs["video_backend"] = args.video_backend
    dataset = LeRobotDataset(args.dataset_id, **ds_kwargs)
    if args.trim_first_n_frames > 0:
        base_len = len(dataset)
        kept_indices = build_trimmed_indices(dataset, args.trim_first_n_frames)
        if not kept_indices:
            raise ValueError(
                f"No samples left after --trim-first-n-frames={args.trim_first_n_frames}. "
                "Lower trim value or check episode lengths."
            )
        dataset = torch.utils.data.Subset(dataset, kept_indices)
        print(
            f"[train] trim-first-n-frames={args.trim_first_n_frames}: "
            f"kept {len(dataset)}/{base_len} samples"
        )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    print(
        f"\n[train] dataloader: {len(dataset)} samples, "
        f"batch_size={args.batch_size}, num_workers={args.num_workers}"
    )

    # ---- optimizer ----
    trainable = [p for p in policy.parameters() if p.requires_grad]
    lr = args.lr if args.lr is not None else cfg.optimizer_lr
    optimizer = torch.optim.AdamW(
        trainable, lr=lr,
        betas=cfg.optimizer_betas, eps=cfg.optimizer_eps,
        weight_decay=cfg.optimizer_weight_decay,
    )
    print(f"[train] optimizer: AdamW lr={lr}")

    # ---- training loop ----
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics_log = []
    step = 0
    epoch = 0
    t_start = time.time()

    print(f"\n[train] starting {args.n_steps} steps...")
    while step < args.n_steps:
        epoch += 1
        for batch in dataloader:
            if step >= args.n_steps:
                break

            t0 = time.time()
            batch = preprocessor(batch)
            loss, info = policy.forward(batch)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.optimizer_grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
            step_time = time.time() - t0

            record = {
                "step": step,
                "epoch": epoch,
                "loss": float(loss.item()),
                "grad_norm": float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm),
                "lr": float(lr),
                "step_time_s": float(step_time),
            }
            metrics_log.append(record)

            if step % args.log_every == 0 or step == args.n_steps - 1:
                elapsed = time.time() - t_start
                print(
                    f"  step {step:5d}/{args.n_steps} | loss {record['loss']:.4f} | "
                    f"grad_norm {record['grad_norm']:.4f} | "
                    f"{step_time:.2f}s/step | elapsed {elapsed:.0f}s"
                )

            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                ckpt_dir = save_dir / f"checkpoint_{step:06d}"
                policy.save_pretrained(ckpt_dir)
                print(f"  [ckpt] saved to {ckpt_dir}")

            step += 1

    # ---- save final ----
    elapsed = time.time() - t_start
    print(f"\n[train] done: {step} steps, {epoch} epochs, {elapsed:.0f}s")

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  peak VRAM: {peak_mb:.0f} MB ({peak_mb/1024:.2f} GB)")

    final_dir = save_dir / "final"
    policy.save_pretrained(final_dir)
    preprocessor.save_pretrained(final_dir)
    postprocessor.save_pretrained(final_dir)
    print(f"  model saved: {final_dir}")

    # ---- save metrics ----
    metrics_summary = {
        "dataset_id": args.dataset_id,
        "episodes_json": args.episodes_json,
        "episodes_key": args.episodes_key,
        "selected_episodes": selected_episodes,
        "trim_first_n_frames": int(args.trim_first_n_frames),
        "pretrained": args.pretrained,
        "n_steps": step,
        "n_epochs": epoch,
        "batch_size": args.batch_size,
        "num_workers": int(args.num_workers),
        "lr": float(lr),
        "total_time_s": float(elapsed),
        "final_loss": float(metrics_log[-1]["loss"]) if metrics_log else None,
        "loss_start": float(metrics_log[0]["loss"]) if metrics_log else None,
        "loss_end": float(metrics_log[-1]["loss"]) if metrics_log else None,
        "peak_vram_mb": float(peak_mb) if torch.cuda.is_available() else None,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "device": str(device),
    }
    (save_dir / "train_summary.json").write_text(
        json.dumps(metrics_summary, indent=2), encoding="utf-8"
    )
    (save_dir / "train_metrics.json").write_text(
        json.dumps(metrics_log, indent=2), encoding="utf-8"
    )
    print(f"  metrics saved: {save_dir / 'train_summary.json'}")
    print(f"  per-step log: {save_dir / 'train_metrics.json'} ({len(metrics_log)} records)")


if __name__ == "__main__":
    main()
