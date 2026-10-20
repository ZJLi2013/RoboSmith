from pathlib import Path

from robotsmith.rollout import video_export


def _write_video(path: Path, content: bytes = b"video") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_copy_episode_videos_prefers_episode_file(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_video(
        dataset_root / "videos" / "observation.images.up" / "episode_000002.mp4",
        b"episode",
    )

    copied = video_export.copy_episode_videos(
        dataset_root,
        [2],
        tmp_path / "out",
        "asset",
        frames_per_episode=None,
        episode_frame_counts=None,
        fps=30,
    )

    assert copied == [str(tmp_path / "out" / "asset_ep002_up.mp4")]
    assert Path(copied[0]).read_bytes() == b"episode"


def test_copy_episode_videos_uses_per_episode_chunk_for_nonzero_episode(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_video(
        dataset_root / "videos" / "observation.images.up" / "chunk-000" / "file-003.mp4",
        b"chunk",
    )

    copied = video_export.copy_episode_videos(
        dataset_root,
        [3],
        tmp_path / "out",
        "asset",
        frames_per_episode=None,
        episode_frame_counts=None,
        fps=30,
    )

    assert copied == [str(tmp_path / "out" / "asset_ep003_up.mp4")]
    assert Path(copied[0]).read_bytes() == b"chunk"


def test_copy_episode_videos_extracts_from_shared_chunk(monkeypatch, tmp_path):
    dataset_root = tmp_path / "dataset"
    src = dataset_root / "videos" / "observation.images.up" / "chunk-000" / "file-000.mp4"
    _write_video(src, b"shared")
    calls = []

    def fake_extract(src_path, target, start_frame, frame_count, fps):
        calls.append((src_path, target, start_frame, frame_count, fps))
        target.write_bytes(b"clip")

    monkeypatch.setattr(video_export, "extract_episode_clip", fake_extract)

    copied = video_export.copy_episode_videos(
        dataset_root,
        [2],
        tmp_path / "out",
        "asset",
        frames_per_episode=None,
        episode_frame_counts={0: 10, 1: 20, 2: 30},
        fps=30,
    )

    target = tmp_path / "out" / "asset_ep002_up.mp4"
    assert copied == [str(target)]
    assert calls == [(src, target, 30, 30, 30)]
    assert target.read_bytes() == b"clip"
