# Phase 3C package

The first implementation milestone contains the causal data contract and
joined-manifest builder. It deliberately has no torch or LIBERO dependency.
Legacy Phase 2D rows with a blank `demo_key` are repaired deterministically
from `demo_id` or the `episode_id` suffix by the versioned builder; a one-off
postprocessing script is not part of the protocol.

```powershell
python -m scripts.phase3c.build_joined_manifest `
  --config configs/phase3c_joined_manifest_full_v2.json
```

The builder joins `(t-6 -> t)` and `(t -> t+6)` samples, retains only the
left/past action window, and writes relation any-change targets from the
current/future graph pair. It fails on graph hash mismatch, invalid action
shape, unsupported relations, missing split metadata, or a forbidden future
field in the model-input view.
The canonical output used by every Phase 3C config is
`data_contract/joined_manifest_full_demo_fixed.jsonl.gz`. Existing joined
artifacts are immutable; reruns must use a new candidate/versioned path.
For the current migrated artifact, build a v2 candidate at a different path
and run `scripts.phase3c.attest_joined_manifest`. Only a `status=pass`
attestation permits reuse of the semantic store whose raw source hash is tied
to the legacy fixed gzip file.

## Milestone 2: semantic feature store

`build_semantic_feature_store.py` renders only the three unique frame steps
needed by each joined sample (`t-6`, `t`, `t+6`) from the official HDF5 state,
selects exactly the two configured RGB observation keys, and caches frozen
DecisionNCE image/text embeddings in per-demo `.npz` shards. Camera key,
channel order, vertical orientation, preprocessing mode, checkpoint hash, and
simulator restore error are written to `manifest.json`; missing or ambiguous
configuration fails the build instead of falling back silently.

Fill the real SSH paths in
`configs/phase3c_semantic_store_example_v1.json`, then run:

```bash
python -m scripts.phase3c.build_semantic_feature_store \
  --config configs/phase3c_semantic_store_v1.json
```

Run a one-episode extraction first and inspect `qa/camera_inventory.json`.
The resulting immutable store is the input to the base CLaD and six-model
screen; this step does not train a model.

The semantic-store config must include the exact DecisionNCE repository commit,
checkpoint, and the action/state smoke's frozen state-restore tolerance. A
completed store also hashes every shard and refuses provenance-free extraction.

Before base training, render the configured and vertically flipped alternatives
for both cameras and verify repeat-render determinism:

```bash
python -m scripts.phase3c.qa_camera_orientation \
  --config "$GRAPH_CLAD_ARTIFACT_ROOT/phase3c_oracle_graph_clad_v1/semantic_store_full_config.json"
```

Inspect `semantic_store/qa/orientation_contact_sheet.png` and
`semantic_store/qa/determinism.json`. If the flipped alternative is correct,
create a new semantic-store version with both camera `vertical_flip` flags set
to `true`; do not overwrite the completed store.

## Milestone 3: controlled CLaD

`models/semantic_clad.py` wraps the original `baseline_code.LatentDynamics`
without editing it. `training_loss` receives the target state only for the
Stage 1 objective, while `encode_foresight` accepts only past action and
history tensors and returns the concatenated `[B, 2H]` foresight embedding.
The wrapper requires the explicit EMA call after each optimizer step and
rejects dimension or non-finite-value mismatches.

## Milestone 5: training and analysis commands

After the semantic store smoke passes, fill the paths in the three example
configs and run them in this order on SSH:

```bash
python -m scripts.phase3c.run_base_clad \
  --config configs/phase3c_base_screen_example_v1.json

python -m scripts.phase3c.train_core \
  --config configs/phase3c_core_smoke_example_v1.json

python -m scripts.phase3c.run_core \
  --config configs/phase3c_core_screen_example_v1.json
```

`run_core` derives `held_out_task_id` from fold names such as
`test_task0`, so training records exclude that task and evaluation records are
restricted to it. Core prediction artifacts contain only past-conditioned
inputs and evaluation labels; future action is never forwarded to a model.
Training uses a bounded-memory seeded shuffle rather than loading the joined
artifact into RAM. Relation eligibility and positive weights use train plus
validation support only, validation fixes F1 thresholds, and held-out test does
not optimize thresholds. The screen automatically parameter-matches every
adapter to the configured RelMPNN reference within the declared tolerance.
Base CLaD writes `best.pt` selected by minimum validation Stage-1 loss and a
separate `last.pt` for exact resume. Core training has a 10,000-update maximum,
validates every 500 updates, can stop after the frozen 3,000-update minimum,
and likewise separates validation-best and resume checkpoints. Completed runs
are skipped only after checkpoint, metric, and prediction hashes are verified.
Metrics retain per-relation macros and additionally report horizontal
(`left/right`), depth (`front/behind`), vertical (`above/below`), contact, and
support family macros so inverse pairs are not presented as independent
families of evidence.
