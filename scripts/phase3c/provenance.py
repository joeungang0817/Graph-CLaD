"""Runtime provenance and tee logging helpers for Phase 3C runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import os
import platform
import subprocess
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TextIO


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repository: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def runtime_provenance(repository: Path | None = None) -> dict[str, Any]:
    """Capture the exact checkout and important runtime package versions."""

    repository = Path(repository or Path(__file__).resolve().parents[2]).resolve()
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    diff = _git(repository, "diff", "--binary", "HEAD")
    packages: dict[str, str | None] = {}
    for name in (
        "torch",
        "numpy",
        "h5py",
        "einops",
        "scikit-learn",
        "Pillow",
        "mujoco",
        "robosuite",
        "setuptools",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(repository),
        "git_commit": _git(repository, "rev-parse", "HEAD"),
        "git_branch": _git(repository, "rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "git_status_sha256": (
            hashlib.sha256(status.encode("utf-8")).hexdigest()
            if status is not None
            else None
        ),
        "git_diff_sha256": (
            hashlib.sha256(diff.encode("utf-8")).hexdigest()
            if diff is not None
            else None
        ),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


class _Tee(io.TextIOBase):
    def __init__(self, original: TextIO, log: TextIO):
        self.original = original
        self.log = log

    def write(self, value: str) -> int:
        written = self.original.write(value)
        self.log.write(value)
        self.log.flush()
        return written

    def flush(self) -> None:
        self.original.flush()
        self.log.flush()

    @property
    def encoding(self) -> str:
        return getattr(self.original, "encoding", "utf-8") or "utf-8"


@contextmanager
def tee_run_output(root: Path) -> Iterator[tuple[Path, Path]]:
    """Mirror Python stdout/stderr to per-run files without hiding the console."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stdout_path = root / "stdout.log"
    stderr_path = root / "stderr.log"
    with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr_handle:
        with redirect_stdout(_Tee(sys.stdout, stdout_handle)), redirect_stderr(
            _Tee(sys.stderr, stderr_handle)
        ):
            yield stdout_path, stderr_path


def attach_log_provenance(runtime: dict[str, Any], stdout_path: Path, stderr_path: Path) -> None:
    runtime.update(
        {
            "stdout_log": str(stdout_path),
            "stdout_log_sha256": sha256_file(stdout_path),
            "stderr_log": str(stderr_path),
            "stderr_log_sha256": sha256_file(stderr_path),
        }
    )
