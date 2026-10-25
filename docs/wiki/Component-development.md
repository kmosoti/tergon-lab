# Component development

Implement and assess one bounded component at a time, with a contract,
independent reference, fixed budget, and saved limitations. The research priority
was reconciled on 2026-10-25 after the layer and MNIST experiments. The earlier
optimizer, kernel, placement, and distribution sequence is no longer a mandatory
order. Those proposals remain useful questions, with evidence-based triggers.

## Completed foundation

Quantization and native packing, the trainable ternary layer, and the fixed MNIST
AdamW experiment are implemented and measured. The layer has independent CPU and
named GPU forward/backward checks; MNIST has three paired ternary/float seeds and
serialized training replay. See [Findings](Findings.md) for outcomes and limits.
These results establish the recorded component and learning behavior, not a
general memory/reasoning architecture or a kernel-performance advantage.

## Current experiment and conditional follow-up

| Priority | Issue | Acceptance |
| --- | --- | --- |
| Active | [#14: typed reads over explicit records](https://github.com/kmosoti/tergon-lab/issues/14) | Independent event/query reference; matched read-policy and ternary/float comparisons; frozen data and recipe; per-world reasoning/calibration metrics; explicit GPU evidence and replay |
| Gated | [#15: learned storage in typed slots](https://github.com/kmosoti/tergon-lab/issues/15) | Run only after every measured-advantage gate in #14 passes; separately measure bounded recurrent storage against float and exact symbolic references |

The [typed-state protocol](Typed-state-experiment.md) defines the experiment and
its fixed decision rule. A negative result is valid evidence; incomplete runs or
failed controls are inconclusive. Neither outcome silently relaxes the pilot
gate. HDC, learned type discovery, and ternary activations remain later research
candidates requiring a specific measured problem.

## Retained research

| Issue | Revisit when | Requirements retained |
| --- | --- | --- |
| [#4: Muon](https://github.com/kmosoti/tergon-lab/issues/4) | A stable reasoning workload exposes a specific optimizer question | Independent matrix reference, shape-aware scaling, zero/rank-deficient cases, decay and ternary-transition ablations |
| [#5: packed kernels](https://github.com/kmosoti/tergon-lab/issues/5) | A workload profile identifies matrix execution or materialization as a worthwhile target | Format/numerical agreement, tails/alignment, full inference cost, named hardware and a measured benefit |
| [#7: placement/offload](https://github.com/kmosoti/tergon-lab/issues/7) | Measured capacity or transfer pressure justifies another placement policy | Ownership and transfer contracts, peak live memory, stalls, bandwidth, failure behavior and fixed-quality comparison |
| [#6: distribution](https://github.com/kmosoti/tergon-lab/issues/6) | A measured single-device limitation and communication budget justify distribution | Partition/synchronization mathematics, network volume, stragglers, recovery and single-device parity |

Basic memory and transfer accounting is included in #14; that does not complete
the offload or placement experiment. Accepted AdamW, dequantized execution, and
explicit single-device placement are valid baselines. A new optimizer, custom
kernel, and offload scheduler are not prerequisites for considering distribution.
No retained issue is closed merely because its priority changed.

[#8: App identities and scoped-merge boundaries](https://github.com/kmosoti/tergon-lab/issues/8)
remains an independent administration task. Scoped merging stays disabled.

## Comparison discipline

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
