"""scripts/setup/hiprt.sh: refuses a foreign HIPRT tree, reuses a build whose stamp matches, rebuilds when a patch
changes, leaves no install behind a failed build, and its env file makes instinct_render importable in a new shell."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup" / "hiprt.sh"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None or shutil.which("python3") is None,
    reason="needs bash and python3 on a POSIX host",
)

FAKE_BUILD = """#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo run >> "$ROOT/builds.log"
mkdir -p "$ROOT/third_party/hiprt_install/bin"
touch "$ROOT/third_party/hiprt_install/bin/libhiprt0300164.so"
[ -z "${FAKE_BUILD_FAIL:-}" ] || exit 1
"""


def _checkout(tmp_path):
    root = tmp_path / "hipRT_render"
    (root / "instinct_render" / "patches").mkdir(parents=True)
    (root / "instinct_render" / "genesis").mkdir()
    (root / "instinct_render" / "__init__.py").write_text("")
    (root / "instinct_render" / "genesis" / "__init__.py").write_text("")
    (root / "instinct_render" / "build_hiprt.sh").write_text(FAKE_BUILD, newline="\n")
    (root / "instinct_render" / "patches" / "hiprt-a.patch").write_text("a\n")
    (tmp_path / "rocm-version").write_text("7.2.4\n")
    return root


def _run(root, **env):
    full = {**os.environ, "ROCM_VERSION_FILE": str(root.parent / "rocm-version"), **env}
    return subprocess.run(["bash", str(SCRIPT), str(root)], capture_output=True, text=True, env=full)


def _builds(root):
    log = root / "builds.log"
    return len(log.read_text().splitlines()) if log.exists() else 0


def test_refuses_a_hiprt_tree_at_another_commit(tmp_path):
    root = _checkout(tmp_path)
    src = root / "third_party" / "HIPRT"
    src.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(src)], check=True)
    subprocess.run(["git", "-C", str(src), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
                    "--allow-empty", "-m", "other"], check=True)
    before = sorted(p.relative_to(root) for p in root.rglob("*") if ".git" not in p.parts)

    run = _run(root)

    assert run.returncode == 3 and "not HIPRT" in run.stderr
    assert sorted(p.relative_to(root) for p in root.rglob("*") if ".git" not in p.parts) == before
    assert _builds(root) == 0


def test_reuses_a_matching_build_and_rebuilds_when_a_patch_changes(tmp_path):
    root = _checkout(tmp_path)

    assert _run(root).returncode == 0
    assert _run(root).returncode == 0
    assert _builds(root) == 1

    (root / "instinct_render" / "patches" / "hiprt-a.patch").write_text("b\n")
    assert _run(root).returncode == 0
    assert _builds(root) == 2


def test_a_failed_build_leaves_no_install(tmp_path):
    root = _checkout(tmp_path)

    run = _run(root, FAKE_BUILD_FAIL="1")

    assert run.returncode == 1
    assert not (root / "third_party" / "hiprt_install").exists()
    assert not (root / "robosmith-hiprt.env").exists()


def test_env_file_makes_the_checkout_importable_in_a_new_shell(tmp_path):
    root = _checkout(tmp_path)
    assert _run(root).returncode == 0

    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "TORCH_EXTENSIONS_DIR")}
    probe = subprocess.run(
        ["bash", "-c", f"source '{root}/robosmith-hiprt.env' && cd / && "
                       "python3 -c 'import os, instinct_render; print(instinct_render.__file__); "
                       "print(os.environ[\"TORCH_EXTENSIONS_DIR\"])'"],
        capture_output=True, text=True, env=env,
    )

    assert probe.returncode == 0, probe.stderr
    module, extensions = probe.stdout.split()
    assert Path(module).parent == root / "instinct_render"
    assert Path(extensions) == root / ".torch_ext"
