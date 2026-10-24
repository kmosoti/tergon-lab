# Component development

Implement and assess one component at a time. Each row becomes an issue and PR
with a contract, independent reference, fixed budget, and saved limitations.

| Stage | Deliverable | Acceptance before advancing |
| --- | --- | --- |
| 1: quantization | Native packing and explicit CPU bridge | Rust f64 oracle, exact packing, invalid-input rejection, receipt; implemented in foundation |
| 2: ternary layer | Latent float weights, quantized forward, explicit surrogate | Shapes and scale axes specified; analytic surrogate tests; CPU and named GPU forward/backward comparison |
| 3: tiny model | Small model trained from scratch with AdamW | Learn a fixed small task, resume equivalence, held-out metric, three paired seeds, full peak memory accounting |
| 4: Muon | Matrix-only Muon plus AdamW for other parameters | Scalar/matrix reference, shape-aware update scaling, zero/rank-deficient cases, decay and ternary-transition ablations |
| 5: kernels | Packed inference kernel | Numerical agreement, tails/alignment, end-to-end speed and memory against dequantized baseline on named hardware |
| 6: placement | Explicit CPU/GPU/RAM/NVMe scheduling | Peak live memory, transfer bytes, pinned-memory costs, prefetch efficacy, failure behavior; faster or lower-memory under fixed quality |
| 7: distribution | Partitioned training/inference | Exact stated synchronization semantics, communication volume, recovery, single-device parity, measured network budget |

Muon is a challenger to AdamW, not the default winner. Compare decay on/off,
decay placement, learning-rate/update RMS, latent-weight norms, ternary sign/zero
transition rates, quantization scale drift, and quality. Apply Muon only to the
matrix parameter classes covered by its contract; keep embeddings, biases, and
other classes explicitly assigned. Attribute existing scaling and distributed
work rather than claiming these problems as new discoveries.

Optimizer state, floating master weights, gradients, activations, temporary
orthogonalization buffers, and transfer staging can dominate packed weights.
Do not infer training fit from an inference checkpoint size. Add CPU offload only
after measuring these components; budget PCIe and host bandwidth separately.

Use at least three paired seeds for initial comparisons and a held-out task split.
Predeclare compute budget, data order, success threshold, and stopping rule. New
defaults need a useful comparative result; failed hypotheses remain valid reports.
