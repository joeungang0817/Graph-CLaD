"""Compatibility alias for :mod:`scripts.phase2r.relation_handlers`."""

try:
    from scripts._legacy_alias import install_alias
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts._legacy_alias import install_alias

install_alias(__name__, "scripts.phase2r.relation_handlers")
