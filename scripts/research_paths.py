"""Environment-neutral paths for local, Linux SSH, and Colab workflows.

Historical experiment configs intentionally keep their frozen absolute Drive
paths.  New notebooks and utilities should use this module for discovery and
override concrete paths through environment variables or command-line flags.
No directory is created unless :func:`ensure_local_output_layout` is called.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


PROJECT_ROOT_ENV = "GRAPH_CLAD_PROJECT_ROOT"
ARTIFACT_ROOT_ENV = "GRAPH_CLAD_ARTIFACT_ROOT"
LIBERO_ROOT_ENV = "GRAPH_CLAD_LIBERO_ROOT"
DEFAULT_COLAB_ARTIFACT_ROOT = Path(
    "/content/drive/MyDrive/Graph-CLaD/artifacts"
)


def running_in_colab(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether the current interpreter is a Google Colab runtime."""

    env = os.environ if environ is None else environ
    if env.get("COLAB_RELEASE_TAG") or env.get("COLAB_GPU"):
        return True
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        return False


def _resolved(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


@dataclass(frozen=True)
class ResearchPaths:
    project_root: Path
    artifact_root: Path
    data_root: Path
    config_root: Path
    notebook_root: Path
    output_root: Path
    libero_root: Path
    colab: bool

    def serializable(self) -> dict[str, str | bool]:
        payload = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in payload.items()
        }


def resolve_research_paths(
    *,
    project_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
    libero_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> ResearchPaths:
    """Resolve project, artifact, dataset, and simulator roots.

    Priority is explicit argument, environment variable, then an
    environment-specific default.  Local and Linux-SSH artifacts default to
    ``outputs/`` unless ``GRAPH_CLAD_ARTIFACT_ROOT`` is set; Colab artifacts
    default to persistent Google Drive.
    """

    env = os.environ if environ is None else environ
    in_colab = running_in_colab(env)
    project = _resolved(
        project_root
        or env.get(PROJECT_ROOT_ENV)
        or Path(__file__).resolve().parents[1]
    )
    artifact = _resolved(
        artifact_root
        or env.get(ARTIFACT_ROOT_ENV)
        or (DEFAULT_COLAB_ARTIFACT_ROOT if in_colab else project / "outputs")
    )
    libero = _resolved(
        libero_root
        or env.get(LIBERO_ROOT_ENV)
        or (Path("/content/LIBERO") if in_colab else project / "external" / "LIBERO")
    )
    return ResearchPaths(
        project_root=project,
        artifact_root=artifact,
        data_root=project / "data",
        config_root=project / "configs",
        notebook_root=project / "notebooks",
        output_root=project / "outputs",
        libero_root=libero,
        colab=in_colab,
    )


def add_project_to_sys_path(paths: ResearchPaths | None = None) -> Path:
    """Add the resolved repository root to ``sys.path`` and return it."""

    root = (paths or resolve_research_paths()).project_root
    if not (root / "scripts").is_dir():
        raise FileNotFoundError(f"Graph-CLaD scripts directory is missing: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def ensure_local_output_layout(paths: ResearchPaths | None = None) -> list[Path]:
    """Create the small local output layout when explicitly requested."""

    resolved = paths or resolve_research_paths()
    created: list[Path] = []
    for name in ("checkpoints", "logs", "figures", "metrics"):
        target = resolved.output_root / name
        target.mkdir(parents=True, exist_ok=True)
        created.append(target)
    return created


def mount_colab_drive(mount_point: str = "/content/drive") -> Path:
    """Mount Drive only in Colab; importing this module locally stays safe."""

    if not running_in_colab():
        raise RuntimeError("Google Drive mounting is available only in Colab")
    from google.colab import drive  # type: ignore[import-not-found]

    target = Path(mount_point)
    drive.mount(str(target), force_remount=False)
    return target


def preflight(paths: ResearchPaths | None = None) -> dict[str, object]:
    resolved = paths or resolve_research_paths()
    return {
        **resolved.serializable(),
        "project_exists": resolved.project_root.exists(),
        "scripts_exists": (resolved.project_root / "scripts").is_dir(),
        "artifact_root_exists": resolved.artifact_root.exists(),
        "libero_root_exists": resolved.libero_root.exists(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--libero-root", type=Path)
    parser.add_argument("--create-local-output-layout", action="store_true")
    args = parser.parse_args()
    paths = resolve_research_paths(
        project_root=args.project_root,
        artifact_root=args.artifact_root,
        libero_root=args.libero_root,
    )
    if args.create_local_output_layout:
        ensure_local_output_layout(paths)
    print(json.dumps(preflight(paths), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
