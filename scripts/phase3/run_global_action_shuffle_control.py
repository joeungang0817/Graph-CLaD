"""Rerun G1 for three seeds with an episode-disjoint test action shuffle.

The historical ``shuffled_action`` control rolls actions by one position inside
each evaluation batch.  Fixed-manifest rows are commonly adjacent frames from
the same episode, so that control can leave action semantics nearly unchanged.
This runner adds ``global_shuffled_action``: a deterministic bijection that
preserves the split's exact action marginal while forcing every donor action to
come from a different episode.  It also persists each trained checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase3 import run_topology_action_followup as followup


def _event_f1(evaluation: Mapping[str, Any]) -> float:
    return float(evaluation["holding"]["change_event"]["f1"])


def _event_ap(evaluation: Mapping[str, Any]) -> float | None:
    value = evaluation["holding"]["change_event"]["pr_auc"]
    return None if value is None else float(value)


def _shuffle_strength(rows: Sequence[Mapping[str, Any]], probe) -> dict[str, Any]:
    actions = np.asarray([row["actions"] for row in rows], dtype=np.float32)
    episodes = np.asarray([str(row["episode_id"]) for row in rows])
    legacy_distances: list[float] = []
    legacy_same_episode: list[bool] = []
    for start in range(0, len(rows), 64):
        batch_actions = actions[start : start + 64]
        batch_episodes = episodes[start : start + 64]
        if len(batch_actions) > 1:
            legacy_distances.extend(
                np.linalg.norm(
                    batch_actions - np.roll(batch_actions, 1, axis=0), axis=1
                ).tolist()
            )
            legacy_same_episode.extend(
                (batch_episodes == np.roll(batch_episodes, 1)).tolist()
            )
    donors = probe._episode_disjoint_action_permutation(rows)
    global_distances = np.linalg.norm(actions - actions[donors], axis=1)
    return {
        "sample_count": len(rows),
        "legacy_batch_roll": {
            "action_l2_mean": float(np.mean(legacy_distances)),
            "same_episode_fraction": float(np.mean(legacy_same_episode)),
        },
        "global_episode_disjoint": {
            "action_l2_mean": float(global_distances.mean()),
            "same_episode_fraction": float(np.mean(episodes == episodes[donors])),
            "is_bijection": len(np.unique(donors)) == len(rows),
        },
    }


def _summarize(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    modes = (
        "no_action",
        "physical_zero_action",
        "shuffled_action",
        "global_shuffled_action",
        "reversed_action",
        "shuffled_gripper",
        "shuffled_arm",
        "shuffled_edge",
    )
    output: dict[str, Any] = {"correct": {}, "correct_minus_control": {}}
    for split in ("natural", "challenge"):
        correct_f1 = [_event_f1(item[split]["correct"]) for item in results]
        correct_ap = [
            value
            for item in results
            if (value := _event_ap(item[split]["correct"])) is not None
        ]
        output["correct"][split] = {
            "event_f1_mean": mean(correct_f1) if correct_f1 else None,
            "event_f1_std": pstdev(correct_f1) if len(correct_f1) > 1 else 0.0,
            "event_f1_values": correct_f1,
            "event_pr_auc_mean": mean(correct_ap) if correct_ap else None,
            "n": len(correct_f1),
        }
        output["correct_minus_control"][split] = {}
        for mode in modes:
            differences = [
                _event_f1(item[split]["correct"])
                - _event_f1(item[split][mode])
                for item in results
            ]
            output["correct_minus_control"][split][mode] = {
                "mean": mean(differences) if differences else None,
                "std": pstdev(differences) if len(differences) > 1 else 0.0,
                "values": differences,
                "positive_count": sum(value > 0 for value in differences),
                "n": len(differences),
            }
    return output


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def run_control(
    manifest_path: Path,
    output_path: Path,
    checkpoint_dir: Path,
    fold_name: str = "test_task0",
    seeds: Sequence[int] = (0, 1, 2),
    resume: bool = False,
) -> dict[str, Any]:
    import torch

    smoke = followup._load_smoke_module()
    probe = smoke._load_probe()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass":
        raise ValueError(f"manifest status is not pass: {manifest.get('status')}")
    relations = [str(value) for value in manifest["relations"]]
    selected = followup._selected_samples(manifest, fold_name)
    records = followup._topology_records(
        selected, "sparse", relations, smoke, probe
    )
    shape = followup._shape(records, len(relations), probe)
    normalization = probe._normalization(records["train"], "channel_v2")
    strength = {
        split: _shuffle_strength(records[split], probe)
        for split in ("validation", "natural_test", "challenge_test")
    }
    prior_results: list[dict[str, Any]] = []
    if resume and output_path.exists():
        prior = json.loads(output_path.read_text(encoding="utf-8"))
        if prior.get("protocol") != "phase3B-R1-global-action-shuffle-control-v1":
            raise ValueError("resume output uses a different protocol")
        if prior.get("fold") != fold_name or prior.get("seeds") != list(seeds):
            raise ValueError("resume output fold/seeds do not match")
        prior_results = list(prior.get("results", []))
    completed_seeds = {int(item["seed"]) for item in prior_results}
    results = prior_results
    modes = (
        "correct",
        "no_action",
        "physical_zero_action",
        "shuffled_action",
        "global_shuffled_action",
        "reversed_action",
        "shuffled_gripper",
        "shuffled_arm",
        "shuffled_edge",
    )

    def payload(status: str) -> dict[str, Any]:
        return {
            "protocol": "phase3B-R1-global-action-shuffle-control-v1",
            "status": status,
            "fold": fold_name,
            "seeds": list(seeds),
            "manifest": str(manifest_path),
            "model_id": "g1_sparse_holder_object_gnn",
            "hidden_dim": 48,
            "parameter_count": probe._parameter_count(
                probe.RelationalDynamicsProbe(
                    "g1_sparse_holder_object_gnn", shape, hidden_dim=48
                )
            ),
            "split_counts": {role: len(rows) for role, rows in records.items()},
            "shuffle_strength": strength,
            "evaluation_modes": list(modes),
            "checkpoint_dir": str(checkpoint_dir),
            "completed_runs": len(results),
            "expected_runs": len(seeds),
            "results": results,
            "summary": _summarize(results),
        }

    _write_json(output_path, payload("running"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for seed in seeds:
        if int(seed) in completed_seeds:
            continue
        probe._set_seed(int(seed))
        train_loader = probe._loader(records["train"], shape, normalization, 64, True)
        validation_loader = probe._loader(
            records["validation"], shape, normalization, 64, False
        )
        natural_loader = probe._loader(
            records["natural_test"], shape, normalization, 64, False
        )
        challenge_loader = probe._loader(
            records["challenge_test"], shape, normalization, 64, False
        )
        model, training = probe.train_one(
            "g1_sparse_holder_object_gnn",
            shape,
            records["train"],
            validation_loader,
            train_loader,
            device,
            48,
            10,
            3,
            0.001,
            0.25,
            relations=relations,
        )
        evaluations: dict[str, Any] = {}
        for split, loader in (
            ("natural", natural_loader),
            ("challenge", challenge_loader),
        ):
            evaluations[split] = {
                mode: probe.evaluate_model(
                    model,
                    loader,
                    device,
                    mode=mode,
                    relations=relations,
                    holding_threshold=training["holding_threshold"],
                )
                for mode in modes
            }
        checkpoint_path = checkpoint_dir / f"g1_test_task0_seed{seed}.pt"
        torch.save(
            {
                "protocol": "phase3B-R1-global-action-shuffle-control-v1",
                "model_id": "g1_sparse_holder_object_gnn",
                "seed": int(seed),
                "shape": shape.__dict__,
                "hidden_dim": 48,
                "relations": relations,
                "training": training,
                "normalization": {
                    key: np.asarray(value).tolist()
                    for key, value in normalization.items()
                },
                "state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
            },
            checkpoint_path,
        )
        result = {
            "seed": int(seed),
            "checkpoint": str(checkpoint_path),
            "training": training,
            **evaluations,
        }
        results.append(result)
        completed_seeds.add(int(seed))
        _write_json(output_path, payload("running"))
        print(
            json.dumps(
                {
                    "seed": seed,
                    "natural_correct": _event_f1(evaluations["natural"]["correct"]),
                    "natural_global_shuffle": _event_f1(
                        evaluations["natural"]["global_shuffled_action"]
                    ),
                    "challenge_correct": _event_f1(evaluations["challenge"]["correct"]),
                    "challenge_global_shuffle": _event_f1(
                        evaluations["challenge"]["global_shuffled_action"]
                    ),
                    "checkpoint": str(checkpoint_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    final = payload("completed")
    _write_json(output_path, final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--fold", default="test_task0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_control(
        args.manifest,
        args.output,
        args.checkpoint_dir,
        fold_name=args.fold,
        seeds=args.seeds,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "completed_runs": result["completed_runs"],
                "expected_runs": result["expected_runs"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
