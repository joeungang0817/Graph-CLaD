# Phase 3C package

The first implementation milestone contains the causal data contract and
joined-manifest builder. It deliberately has no torch or LIBERO dependency.

```powershell
python -m scripts.phase3c.build_joined_manifest `
  --config configs/phase3c_contract_v1.json `
  --input <phase2d_task0.jsonl.gz> `
  --output <phase3c_joined.jsonl.gz> `
  --qa-output <phase3c_joined_qa.json>
```

The builder joins `(t-6 -> t)` and `(t -> t+6)` samples, retains only the
left/past action window, and writes relation any-change targets from the
current/future graph pair. It fails on graph hash mismatch, invalid action
shape, unsupported relations, missing split metadata, or a forbidden future
field in the model-input view.

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
