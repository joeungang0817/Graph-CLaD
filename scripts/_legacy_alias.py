"""Compatibility helper for pre-reorganization module paths."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


def install_alias(wrapper_name: str, target: str) -> ModuleType:
    """Load ``target`` and expose it under an older module name."""

    implementation = importlib.import_module(target)
    if wrapper_name != "__main__":
        sys.modules[wrapper_name] = implementation
    return implementation

