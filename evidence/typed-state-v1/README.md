# Typed state v1 evidence

The completed experiment did not pass the storage-pilot gate. The measured
ternary stress gain was 3.75 percentage points (paired 95% interval3.25–4.25),
and absolute ternary quality remained below the predeclared thresholds. See
[full measured results](../../docs/wiki/Typed-state-results.md).

The [frozen protocol](../../docs/wiki/Typed-state-experiment.md) and
[recipe](../../experiments/typed-state-v1.json) define this structured-event
experiment. All twelve conditions use source
`4144184b1a83986f44feee36bcef3bfc0a086c84`, with one fresh CUDA process per
condition. [Execution identity](execution.json) records the exact release binary,
recipe and outer runner. [Provenance](provenance.json) identifies source files,
input hashes and artifact collection.

[Analysis](analysis.json) is the predeclared aggregate result. The independently
recomputed [results audit](independent-results-audit.json) verifies predictions,
checkpoint identities and the paired whole-world bootstrap. [Summary](summary.json)
retains every condition's metrics, replay results, timings and memory observations;
[learning curves](learning-curves.csv) retain all epochs. The individual receipts
under `conditions/` contain full group/hop/outcome metrics, Brier scores,
unsupported predictions, renaming/order consistency and counterfactual responses.

The clean final [corpus audit](corpus/audit.json) and
[independent event audit](corpus/independent-python-audit.json) cover 47,104 queries
in 11,776 disjoint worlds. The independent checks include primitive coverage,
holdout constituents, identical nuisance metadata and equal Known/Conflict
observed degree. [Dataset inventory](corpus/dataset-inventory.json) records the
exact local corpus files. Regenerate them from the frozen revision with
`state-audit <new-dir>`; each dataset byte hash must match the inventory. Generation
and compilation are excluded from the per-condition 600-second execution budget.

All 60 prediction files are committed as lossless gzip with deterministic headers.
[Prediction inventory](prediction-inventory.json) records both compressed and
original byte hashes. To reconstruct the original receipt/prediction layout in a
new scratch directory:

```bash
python3 evidence/typed-state-v1/scripts/rehydrate-evidence.py evidence/typed-state-v1 /tmp/typed-state-reconstructed
cargo run --release --locked -p tergon-lab -- state-analyze /tmp/typed-state-reconstructed /tmp/typed-state-recomputed-analysis.json
```

The original per-world predictions/probabilities are restored byte-for-byte; no
training is required to recompute accuracy, Brier or the predeclared bootstrap.
Do not conflate recomputation of these metrics with retraining reproducibility on
other hardware or arithmetic.

Checkpoint payloads remain in the ignored local study directory. The committed
[checkpoint inventory](checkpoint-inventory.json) records their exact byte hashes,
shapes, optimizer counters and training cursors. To re-run the original independent
combined prediction/checkpoint verifier, provide that local checkpoint directory:

```bash
python3 evidence/typed-state-v1/scripts/rehydrate-evidence.py evidence/typed-state-v1 /tmp/typed-state-full-audit --checkpoints /home/kmosoti/Projects/tergon-lab/runs/typed-state-v1-formal
python3 evidence/typed-state-v1/scripts/tergon-independent-state-results.py /tmp/typed-state-full-audit /tmp/typed-state-full-audit/analysis.json
```

The unaltered outer runner, independent event auditor and independent results
verifier are in `scripts/`. Their hashes are covered by the
[bundle inventory](bundle-inventory.json). The repository-wide evidence inventory
also covers committed artifacts. Checkpoint bytes are intentionally not presented
as committed or remotely retrievable payloads.

Validation logs preserve the original ideal-f64 GPU failures and the separate
TF32 diagnosis. Pinned CUDA matmuls use TF32 operand rounding with f32 storage;
a precision-aware independent reference reduced direct stage residuals to about
3e-8. Existing layer/bridge/replay tolerances were unchanged. The failed ideal-f64
comparison remains evidence of arithmetic drift, not a discarded result.

The experiment is limited to synthetic structured worlds, a fixed 64-wide memory
reader and three model seeds. Process RSS includes corpus/runtime and briefly a
second replay model. GPU samples can miss transients. Host-read wait overlaps
computation/transfers. Packed-export bytes are hypothetical and nonresident;
latent matrices, biases, moments and sampled process memory are separate. No
energy, broad reasoning, generic ternary architecture, kernel-speed or total-memory
savings claim follows.
