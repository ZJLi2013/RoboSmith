"""Video export helpers for rollout datasets."""

from __future__ import annotations

import importlib
from pathlib import Path
import shutil
import subprocess


def extract_episode_clip(
    src: Path,
    target: Path,
    start_frame: int | None,
    frame_count: int | None,
    fps: int,
) -> None:
    if start_frame is None or frame_count is None:
        shutil.copy2(src, target)
        return
    start_s = start_frame / fps
    duration_s = frame_count / fps
    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_exe = "ffmpeg"
    subprocess.run(
        [
            ffmpeg_exe,
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_s:.6f}",
            "-t",
            f"{duration_s:.6f}",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(target),
        ],
        check=True,
    )


def copy_episode_videos(
    dataset_root: Path,
    episode_ids: list[int],
    dst: Path,
    asset_name: str,
    frames_per_episode: int | None,
    episode_frame_counts: dict[int, int] | None,
    fps: int,
) -> list[str]:
    dst.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for ep in episode_ids:
        for cam in ("up",):
            video_root = dataset_root / "videos" / f"observation.images.{cam}"
            episode_src = video_root / f"episode_{ep:06d}.mp4"
            chunk_src = video_root / "chunk-000" / "file-000.mp4"
            per_episode_chunk_src = video_root / "chunk-000" / f"file-{ep:03d}.mp4"
            if episode_src.exists():
                src = episode_src
                should_extract = False
            elif per_episode_chunk_src.exists() and ep > 0:
                src = per_episode_chunk_src
                should_extract = False
            elif chunk_src.exists():
                src = chunk_src
                should_extract = True
            else:
                continue
            target = dst / f"{asset_name}_ep{ep:03d}_{cam}.mp4"
            if should_extract:
                if episode_frame_counts:
                    frame_count = episode_frame_counts.get(ep)
                    start_frame = sum(
                        episode_frame_counts.get(prev_ep, 0)
                        for prev_ep in range(ep)
                    )
                else:
                    frame_count = frames_per_episode
                    start_frame = ep * frames_per_episode if frames_per_episode is not None else None
                extract_episode_clip(src, target, start_frame, frame_count, fps)
            else:
                shutil.copy2(src, target)
            copied.append(str(target))
    return copied
