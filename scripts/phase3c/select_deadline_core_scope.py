"""Select the deadline Core scope from a throughput-only benchmark.

The decision is intentionally independent of validation/test scores.  It uses
the completed wall time of a fixed 100-update, batch-64 technical run and a
conservative linear projection for the formal 200-update runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .io import write_json


BENCHMARK_PROTOCOL = "phase3c-deadline-postcache-throughput-v1"
BENCHMARK_UPDATES = 100
FORMAL_UPDATES = 200
THREE_MODEL_RUNS = 9
SIX_MODEL_RUNS = 18


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_scope(
    benchmark_runtime: Path,
    *,
    remaining_hours: float,
    reserve_hours: float = 1.0,
) -> dict[str, Any]:
    path = Path(benchmark_runtime)
    with path.open("r", encoding="utf-8") as handle:
        runtime = json.load(handle)
    if not isinstance(runtime, dict) or runtime.get("status") != "completed":
        raise ValueError("throughput benchmark runtime must be completed")
    required_identity = {
        "protocol": BENCHMARK_PROTOCOL,
        "claim_scope": "technical-throughput-only",
        "model_id": "C3-Sem-PastAct",
        "fold": "test_task0",
        "seed": 0,
        "updates": BENCHMARK_UPDATES,
        "batch_size": 64,
    }
    mismatches = {
        key: {"expected": expected, "observed": runtime.get(key)}
        for key, expected in required_identity.items()
        if runtime.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"throughput benchmark identity mismatch: {mismatches}")
    elapsed_seconds = float(runtime.get("total_elapsed_seconds", float("nan")))
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
        raise ValueError(
            "throughput benchmark total_elapsed_seconds must be finite and positive"
        )
    remaining_hours = float(remaining_hours)
    reserve_hours = float(reserve_hours)
    if (
        not math.isfinite(remaining_hours)
        or not math.isfinite(reserve_hours)
        or remaining_hours <= 0.0
        or reserve_hours < 0.0
    ):
        raise ValueError("remaining_hours must be positive and reserve_hours non-negative")

    # This deliberately multiplies the complete 100-update runtime—including
    # validation and evaluation—by two. Formal 200-update runs evaluate once,
    # not twice, so the estimate is conservative rather than optimistic.
    projected_seconds_per_formal_run = elapsed_seconds * (
        FORMAL_UPDATES / BENCHMARK_UPDATES
    )
    projections = {
        "three_model_hours": projected_seconds_per_formal_run
        * THREE_MODEL_RUNS
        / 3600.0,
        "six_model_hours": projected_seconds_per_formal_run
        * SIX_MODEL_RUNS
        / 3600.0,
    }
    available_core_hours = max(0.0, remaining_hours - reserve_hours)
    if projections["six_model_hours"] <= available_core_hours:
        selection = "six-model"
        config = "configs/phase3c_core_deadline_sixmodel_threefold_seed0_v1.json"
        analysis_config = (
            "configs/phase3c_analysis_deadline_sixmodel_threefold_seed0_v1.json"
        )
    elif projections["three_model_hours"] <= available_core_hours:
        selection = "three-model"
        config = "configs/phase3c_core_deadline_threefold_seed0_v1.json"
        analysis_config = "configs/phase3c_analysis_deadline_threefold_seed0_v1.json"
    else:
        selection = "insufficient-time"
        config = None
        analysis_config = None
    return {
        "schema": "phase3c-deadline-core-scope-selection.v1",
        "status": "pass",
        "decision_basis": "wall_clock_only_no_performance_metrics",
        "benchmark_runtime": str(path),
        "benchmark_runtime_sha256": _sha256(path),
        "benchmark_total_elapsed_seconds": elapsed_seconds,
        "formal_updates": FORMAL_UPDATES,
        "benchmark_updates": BENCHMARK_UPDATES,
        "projection_policy": "complete_100_update_runtime_times_two_per_formal_run",
        "projected": projections,
        "remaining_hours_at_decision": remaining_hours,
        "reserved_hours": reserve_hours,
        "available_core_hours": available_core_hours,
        "selection": selection,
        "core_config": config,
        "analysis_config": analysis_config,
        "performance_fields_consulted": [],
    }


def main() -> None:  # pragma: no cover - SSH orchestration CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-runtime", type=Path, required=True)
    parser.add_argument("--remaining-hours", type=float, required=True)
    parser.add_argument("--reserve-hours", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select_scope(
        args.benchmark_runtime,
        remaining_hours=args.remaining_hours,
        reserve_hours=args.reserve_hours,
    )
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
