"""Inspect and apply the narrowly-scoped Colab robosuite/MuJoCo compatibility fix."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
from pathlib import Path
from typing import Any


OLD_CALL = "mujoco.mj_fullM(self.sim.model._model, mass_matrix, self.sim.data.qM)"
NEW_CALL = """try:
            mujoco.mj_fullM(self.sim.model._model, mass_matrix, self.sim.data.qM)
        except (AttributeError, TypeError):
            mujoco.mj_fullM(self.sim.model._model, self.sim.data._data, mass_matrix)"""


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_runtime(controller_path: Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "robosuite": package_version("robosuite"),
        "mujoco": package_version("mujoco"),
        "numpy": package_version("numpy"),
    }
    if controller_path is not None:
        text = controller_path.read_text(encoding="utf-8")
        result.update(
            controller_path=str(controller_path),
            old_call_present=OLD_CALL in text,
            compatibility_call_present="self.sim.data._data, mass_matrix" in text,
        )
    return result


def patch_mass_matrix_call(controller_path: Path, apply: bool = False) -> dict[str, Any]:
    """Patch only the known robosuite 1.4.1 call, keeping a recoverable backup."""

    before = controller_path.read_text(encoding="utf-8")
    if "self.sim.data._data, mass_matrix" in before:
        return {"status": "already_patched", **inspect_runtime(controller_path)}
    if OLD_CALL not in before:
        raise RuntimeError("expected robosuite mass-matrix call was not found; refusing to patch")
    after = before.replace(OLD_CALL, NEW_CALL, 1)
    compile(after, str(controller_path), "exec")
    if apply:
        backup = controller_path.with_suffix(controller_path.suffix + ".graph_clad.bak")
        if not backup.exists():
            shutil.copy2(controller_path, backup)
        controller_path.write_text(after, encoding="utf-8")
    return {
        "status": "patched" if apply else "patch_available",
        "apply_requested": apply,
        **inspect_runtime(controller_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("controller_path", type=Path)
    parser.add_argument("--apply", action="store_true", help="write the patch; default is dry-run")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(json.dumps(patch_mass_matrix_call(args.controller_path, args.apply), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

