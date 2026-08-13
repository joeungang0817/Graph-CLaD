"""Compatibility entry point; use :mod:`scripts.phase3.offline_probe`."""

try:
    from scripts._legacy_alias import install_alias
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts._legacy_alias import install_alias

_implementation = install_alias(__name__, "scripts.phase3.offline_probe")

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
