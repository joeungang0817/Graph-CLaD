# Phase 4 Stage-1 integration gate

This package deliberately stops at the model-independent residual foresight
interface until Phase 3C selects a structured architecture and its nearest
fairness control.

`ResidualGraphForesightAdapter` accepts the frozen semantic CLaD foresight and
any Phase 3C causal structured encoder. It adds separate residuals to the
proprio and scene branches, normalizes both, and retains the same `[B,2H]`
tensor shape as `SemanticForesightInterface`.

Before a Stage 2 controlled DDPM implementation is allowed, run:

```bash
python -m unittest discover -s tests -p 'test_phase4_foresight_adapter.py' -v
```

The gate requires exact semantic equivalence at zero alpha, an adapter-off
path that never invokes the structured encoder, identical output shape, and a
nonzero-residual response to causal structured input. Phase 3C winner
selection and a finite Phase 4 training smoke remain separate prerequisites.
