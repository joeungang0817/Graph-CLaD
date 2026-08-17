# Phase 3C deadline-constrained pilot runbook

## Frozen claim and scope

Protocol ID: `phase3c-deadline-threefold-seed0-v1`.

This run is a deadline-constrained offline pilot, not the full preregistered
25K/10K screen and not evidence of policy or rollout improvement. The fixed
scope is:

- Base CLaD: 500 updates, batch 64, seed 0, three held-out-task folds;
- mandatory Core tier: 100 updates, batch 64, seed 0, the same three folds for
  `C3-Sem-PastAct`, `C3-RelMPNN-PastAct`, then `C3-RelPool-PastAct`;
- conditional expanded tier: all six Core models at the same budget only when
  the post-cache throughput rule selects it before any formal Core scores are
  observed;
- validation selects Base/Core checkpoints and relation thresholds separately
  inside each fold; held-out test data does not select either;
- final analysis reports per-relation and inverse-relation-family macros,
  paired task-to-episode bootstrap, no-change and train-prevalence baselines,
  and the `prev_step=0` sensitivity exclusion;
- additional seeds, Phase 4 training, Stage 2 policy training, and rollout
  evaluation are deferred. The original 10K six-model screen remains deferred
  even if the 100-update six-model deadline tier is selected.

The model order is deliberate. If wall-clock time is unexpectedly exhausted,
the semantic-to-RelMPNN primary contrast finishes before the RelPool fairness
control and before the three broader baselines. A partial run is recorded as
partial; its completed subset is not silently relabeled as a completed tier.

## 0. Preserve and verify the existing six-model technical smoke

Let the already-running Core v5 smoke finish before changing the SSH checkout.
It is technical executability evidence only. It is accepted only if the final
screen contains all six model IDs, one `test_task0`/seed-0 run each, 20 updates,
finite loss values, passing parameter matching, and existing checkpoint,
metric, and prediction artifacts. Do not use its metrics or checkpoints for
formal model selection.

## 1. Deploy and run the integrity gate

From a clean, updated SSH checkout:

```bash
cd /home/ubuntu/Graph-CLaD
source .venv/bin/activate
export GRAPH_CLAD_ARTIFACT_ROOT=/home/ubuntu/graphclad-artifacts

git status --porcelain

python -m unittest discover -s tests -p "test_phase3c_*.py" -v
python -m unittest discover -s tests -p "test_phase4_foresight_adapter.py" -v
```

The expected total is 63 Phase 3C tests plus 3 Phase 4 contract tests. On SSH,
none may be skipped for missing PyTorch. Stop before training on any failure or
unexpected skip.

This gate also covers the semantic-shard materialization optimization added
after the CPU-bound smoke: a compressed NPZ shard is inflated once per cache
residency, not once per sample. Because this changes the input pipeline's
runtime behavior, do not reuse a formal checkpoint produced before this gate.

## 2. Bind the already-reviewed camera sheet to the semantic store

The human review selected the configured orientation for both external and
wrist images. Recreate only the attestation after deploying the current code;
do not rebuild the already-accepted 150 semantic shards.

```bash
export ART_ROOT="$GRAPH_CLAD_ARTIFACT_ROOT/phase3c_oracle_graph_clad_v1"

python -m scripts.phase3c.attest_camera_orientation \
  --qa "$ART_ROOT/semantic_store/qa/determinism.json" \
  --output "$ART_ROOT/semantic_store/qa/orientation_human_attestation.json" \
  --reviewer phase3c-owner \
  --external-choice configured \
  --wrist-choice configured \
  --notes "Configured orientation accepted from reviewed contact sheet; deadline pilot v1."
```

The command must report `status=pass`, both choices `configured`, and
`accepted_existing_semantic_store=true`. Base startup then rechecks the contact
sheet, QA, attestation, manifest, and all declared shard hashes.

## 3. Run the reduced Base three-fold training

```bash
python -m scripts.phase3c.run_base_clad \
  --config configs/phase3c_base_deadline_threefold_seed0_v1.json
```

Expected root:
`$ART_ROOT/base_clad_deadline_threefold_seed0_v1`. The screen is complete only
with three `best.pt` validation-selected checkpoints and three `last.pt` resume
checkpoints. Rerunning the same command resumes an interrupted run and verifies
completed artifacts before skipping them.

## 4. Select and run the Core pilot scope

First measure the post-cache code with one fixed, technical-only 20-update run:

```bash
python -m scripts.phase3c.run_core \
  --config configs/phase3c_core_postcache_benchmark_v1.json
```

Record the actual hours remaining, then make the scope decision only from that
runtime. For example, if 7.5 hours remain:

```bash
python -m scripts.phase3c.select_deadline_core_scope \
  --benchmark-runtime "$ART_ROOT/core_postcache_benchmark_v1/C3-Sem-PastAct/test_task0/seed0/runtime_manifest.json" \
  --remaining-hours 7.5 \
  --reserve-hours 1.0 \
  --output "$ART_ROOT/analysis/deadline_core_scope_selection_v1.json"
```

The selector multiplies the entire 20-update wall time by five for every
formal run. This is conservative because it also multiplies validation and
evaluation overhead. It never reads a performance field. Use exactly the
config named in its `core_config` field:

```bash
# selection=three-model
python -m scripts.phase3c.run_core \
  --config configs/phase3c_core_deadline_threefold_seed0_v1.json

# selection=six-model
python -m scripts.phase3c.run_core \
  --config configs/phase3c_core_deadline_sixmodel_threefold_seed0_v1.json
```

Do not start both formal output roots. If the selector reports
`insufficient-time`, preserve the benchmark and Base artifacts and report that
the preregistered Core pilot could not be completed; do not shrink updates or
folds after seeing data.

### Three-model selected scope

Expected root:
`$ART_ROOT/core_deadline_threefold_seed0_v1`. This produces nine formal runs.
The runner completes all three folds for semantic first, all three folds for
RelMPNN second, and all three folds for RelPool last. Rerun the identical
command after interruption; never edit the config or output root in place.

## 5. Analyze only the complete three-fold artifacts

```bash
python -m scripts.phase3c.analyze_core \
  --config configs/phase3c_analysis_deadline_threefold_seed0_v1.json

# Use this instead when the selector chose six-model.
python -m scripts.phase3c.analyze_core \
  --config configs/phase3c_analysis_deadline_sixmodel_threefold_seed0_v1.json
```

Expected output:
`$ART_ROOT/analysis/core_deadline_threefold_seed0_v1.json`, or the corresponding
`core_deadline_sixmodel_threefold_seed0_v1.json`. The analyzer fails if a model
lacks a fold or seed. The primary estimate is
`C3-RelMPNN-PastAct - C3-Sem-PastAct`; the RelPool comparison isolates the
effect of message passing while keeping the relation-token input family.

## 6. Final evidence to preserve

Copy into the research record, without manually editing the JSON artifacts:

- git commit and clean status used by the runs;
- the 63+3 SSH test summaries and camera-attestation SHA;
- Core v5 smoke screen manifest and its narrow technical-only waiver;
- Base and Core screen manifests plus every runtime manifest path;
- the analysis artifact path and SHA-256;
- the throughput benchmark and immutable scope-selection artifact;
- wall time, GPU, update/batch budget, incomplete or resumed runs;
- the missing `on` positives, the one action-timing tolerance outlier, the
  0/90 deferred weak-label audit, single-seed uncertainty, reduced budget, and
  absence of Stage 2 rollout evidence.

The report may say the pilot tested whether relation-aware structure shows an
offline signal under a matched reduced protocol. It may not say Graph-CLaD
improves robot success, reproduces official CLaD, or completes Stage 2.
