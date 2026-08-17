"""Record the human orientation decision for a completed camera QA artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .io import write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attest(
    qa_path: Path,
    output: Path,
    *,
    reviewer: str,
    external_choice: str,
    wrist_choice: str,
    notes: str = "",
) -> dict[str, Any]:
    if not reviewer.strip():
        raise ValueError("reviewer must not be blank")
    choices = {"external": external_choice, "wrist": wrist_choice}
    if any(value not in {"configured", "flipped"} for value in choices.values()):
        raise ValueError("camera choices must be configured or flipped")
    with Path(qa_path).open("r", encoding="utf-8") as handle:
        qa = json.load(handle)
    if qa.get("schema") != "phase3c-camera-orientation-qa.v1" or qa.get("status") != "pass":
        raise ValueError("human attestation requires a passing camera-orientation QA v1")
    result = {
        "schema": "phase3c-camera-orientation-human-attestation.v1",
        "status": "pass",
        "source_qa": str(Path(qa_path)),
        "source_qa_sha256": _sha256(Path(qa_path)),
        "source_contact_sheet": qa.get("contact_sheet"),
        "reviewer": reviewer.strip(),
        "choices": choices,
        "accepted_existing_semantic_store": all(
            value == "configured" for value in choices.values()
        ),
        "notes": notes,
    }
    write_json(Path(output), result)
    return result


def main() -> None:  # pragma: no cover - human review CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--external-choice", choices=("configured", "flipped"), required=True)
    parser.add_argument("--wrist-choice", choices=("configured", "flipped"), required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    result = attest(
        args.qa,
        args.output,
        reviewer=args.reviewer,
        external_choice=args.external_choice,
        wrist_choice=args.wrist_choice,
        notes=args.notes,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
