"""Persist Phase 2D inputs and artifacts across Colab runtime resets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def backup(
    workspace: Path,
    persistent_root: Path,
    names: Iterable[str] = ("scripts", "configs", "data", "libero_spatial_hdf5"),
) -> dict[str, Any]:
    """Copy selected project paths to a persistent root and checksum all files."""

    persistent_root.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in names:
        source = workspace / name
        if not source.exists():
            continue
        _copy_path(source, persistent_root / name)
        copied.append(name)
    files = []
    for path in sorted(persistent_root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(persistent_root)),
                    "bytes": path.stat().st_size,
                    "sha256": _checksum(path),
                }
            )
    manifest = {
        "manifest_version": "phase2d-persistence.v1",
        "workspace": str(workspace),
        "persistent_root": str(persistent_root),
        "copied_roots": copied,
        "files": files,
    }
    (persistent_root / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def restore(persistent_root: Path, workspace: Path, verify: bool = True) -> dict[str, Any]:
    """Overlay a persistent bundle onto a fresh runtime without deleting local files."""

    manifest_path = persistent_root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    if verify:
        for record in manifest.get("files", []):
            source = persistent_root / record["path"]
            if not source.exists() or _checksum(source) != record["sha256"]:
                mismatches.append(record["path"])
        if mismatches:
            raise RuntimeError(f"persistent bundle checksum mismatch: {mismatches[:5]}")
    workspace.mkdir(parents=True, exist_ok=True)
    restored: list[str] = []
    for name in manifest.get("copied_roots", []):
        source = persistent_root / name
        if source.exists():
            _copy_path(source, workspace / name)
            restored.append(name)
    return {"status": "pass", "restored_roots": restored, "checksum_mismatches": mismatches}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    backup_parser = subparsers.add_parser("backup")
    backup_parser.add_argument("--workspace", type=Path, required=True)
    backup_parser.add_argument("--persistent-root", type=Path, required=True)
    backup_parser.add_argument("--names", nargs="+", default=["scripts", "configs", "data"])
    restore_parser = subparsers.add_parser("restore")
    restore_parser.add_argument("--persistent-root", type=Path, required=True)
    restore_parser.add_argument("--workspace", type=Path, required=True)
    restore_parser.add_argument("--no-verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "backup":
        result = backup(args.workspace, args.persistent_root, args.names)
    else:
        result = restore(args.persistent_root, args.workspace, verify=not args.no_verify)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

