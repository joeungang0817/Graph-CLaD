"""Compatibility entry point; use :mod:`scripts.phase2r.collect_scaleup`."""

try:
    from scripts._legacy_alias import install_alias
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts._legacy_alias import install_alias

_implementation = install_alias(__name__, "scripts.phase2r.collect_scaleup")

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
