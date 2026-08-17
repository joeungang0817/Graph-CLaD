"""Attest that a legacy fixed joined manifest equals builder-v2 output.

Gzip container hashes can differ even when every JSON row is identical.  This
tool compares ordered canonical JSON rows, records both raw and semantic
SHA-256 values, and writes an immutable migration attestation.  It allows the
already-built semantic store to retain its exact source hash without silently
claiming that a one-off file was produced by the versioned builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .build_semantic_feature_store import _expand_path, sha256_file
from .contracts import canonical_json
from .io import iter_json_objects, write_json


def semantic_manifest_digest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows = 0
    first_sample_id: str | None = None
    last_sample_id: str | None = None
    for raw in iter_json_objects(path):
        sample_id = str(raw.get("sample_id", ""))
        if first_sample_id is None:
            first_sample_id = sample_id
        last_sample_id = sample_id
        digest.update(canonical_json(raw))
        digest.update(b"\n")
        rows += 1
    if rows == 0:
        raise ValueError(f"joined manifest is empty: {path}")
    return {
        "path": str(path),
        "raw_sha256": sha256_file(path),
        "canonical_rows_sha256": digest.hexdigest(),
        "rows": rows,
        "first_sample_id": first_sample_id,
        "last_sample_id": last_sample_id,
    }


def attest(existing: Path, candidate: Path, output: Path) -> dict[str, Any]:
    existing = Path(existing)
    candidate = Path(candidate)
    output = Path(output)
    if not existing.exists() or not candidate.exists():
        missing = [str(path) for path in (existing, candidate) if not path.exists()]
        raise FileNotFoundError(", ".join(missing))
    if existing.resolve() == candidate.resolve():
        raise ValueError("existing and candidate manifests must be different files")
    if output.exists():
        raise FileExistsError(
            f"joined-manifest attestation is immutable; use a new path: {output}"
        )
    legacy = semantic_manifest_digest(existing)
    built = semantic_manifest_digest(candidate)
    equivalent = (
        legacy["rows"] == built["rows"]
        and legacy["canonical_rows_sha256"] == built["canonical_rows_sha256"]
    )
    report = {
        "schema": "phase3c-joined-manifest-migration-attestation.v1",
        "status": "pass" if equivalent else "fail",
        "equivalent_ordered_canonical_rows": equivalent,
        "legacy_semantic_store_source": legacy,
        "builder_v2_candidate": built,
        "decision": (
            "legacy fixed manifest may remain the immutable semantic-store source"
            if equivalent
            else "do not reuse legacy semantic store; investigate and rebuild"
        ),
    }
    write_json(output, report)
    if not equivalent:
        raise ValueError("legacy and builder-v2 joined manifests are not equivalent")
    return report


def main() -> None:  # pragma: no cover - SSH CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = attest(
        _expand_path(args.existing),
        _expand_path(args.candidate),
        _expand_path(args.output),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
