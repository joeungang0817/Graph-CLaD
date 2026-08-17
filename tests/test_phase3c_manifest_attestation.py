from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scripts.phase3c.attest_joined_manifest import attest


def _write(path: Path, rows: list[dict], *, compact: bool) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            if compact:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            else:
                handle.write(json.dumps(row, sort_keys=True) + "\n")


class Phase3CManifestAttestationTest(unittest.TestCase):
    def test_equivalent_json_rows_pass_despite_container_difference(self):
        rows = [
            {"schema": "phase3c-joined-sample.v1", "sample_id": "a", "value": [1, 2]},
            {"schema": "phase3c-joined-sample.v1", "sample_id": "b", "value": [3, 4]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing.jsonl.gz"
            candidate = root / "candidate.jsonl.gz"
            output = root / "attestation.json"
            _write(existing, rows, compact=True)
            _write(candidate, rows, compact=False)
            report = attest(existing, candidate, output)
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["equivalent_ordered_canonical_rows"])

    def test_changed_row_fails_and_records_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing.jsonl.gz"
            candidate = root / "candidate.jsonl.gz"
            output = root / "attestation.json"
            _write(existing, [{"sample_id": "a", "value": 1}], compact=True)
            _write(candidate, [{"sample_id": "a", "value": 2}], compact=True)
            with self.assertRaises(ValueError):
                attest(existing, candidate, output)
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "fail")


if __name__ == "__main__":
    unittest.main(verbosity=2)
