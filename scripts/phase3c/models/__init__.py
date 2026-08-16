"""Phase 3C model components.

Imports are intentionally kept lazy at package level so the causal data
contract remains usable on a CPU environment without PyTorch.
"""

__all__ = ["ControlledCLaD"]


def __getattr__(name: str):
    if name == "ControlledCLaD":
        from .semantic_clad import ControlledCLaD

        return ControlledCLaD
    raise AttributeError(name)
