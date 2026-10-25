# Typed state v1 results

The storage-pilot gate is closed. All twelve runs completed, and every
metadata-only float control passed the ordinary 90% competence threshold.
Contract enforcement improved mean ternary stress accuracy by 3.75 percentage
points, below the predeclared 5-point requirement. Enforced ternary ordinary and
stress accuracy also remained well below their 90% and 85% thresholds. This is a
completed negative gate result for the fixed reader and recipe; it does not
establish a general limit of typed state or ternary architectures.

The [predeclared protocol](Typed-state-experiment.md) compares metadata-only and
contract-enforced reads with identical records, metadata, shapes, initial latent
parameters, examples, batch order and update count. All conditions completed 20
epochs and 5,120 updates within 600 seconds. Evaluation used final checkpoints;
there was no validation selection or post-outcome budget extension.

## Measured accuracy and calibration

These are mean exact-answer accuracies over seeds 17, 29, 43. Each test stratum has
512 worlds with four related queries per world. The five strata are equally
weighted where combined; outcome balance is 50% Known, 25% Unknown, 25% Conflict.

| Weights | Reader | Ordinary | Unseen composition | Replace/retract | Conflict/hypothesis | Distractors |
| --- | --- | --- | --- | --- | --- | --- |
| float | metadata-only | 95.48% | 89.39% | 94.84% | 94.87% | 84.91% |
| float | enforced | 99.30% | 96.83% | 99.32% | 99.48% | 97.61% |
| ternary | metadata-only | 53.84% | 37.89% | 54.02% | 55.63% | 45.85% |
| ternary | enforced | 58.97% | 39.62% | 58.09% | 60.24% | 50.44% |

The mean ternary stress gain was 3.7476 percentage points, with a paired 95%
whole-world interval of 3.2511–4.2521 points. All three gains were positive; their
sample standard deviation was 2.9861 points. The interval resamples worlds within
the four stress strata 10,000 times, averaging the three model seeds inside each
sampled world. It is conditional on those model seeds and does not substitute
for seed-to-seed uncertainty.

| Seed | Float control ordinary | Enforced ternary ordinary | Enforced ternary stress | Ternary stress gain | Five-group Brier change |
| --- | --- | --- | --- | --- | --- |
| 17 | 95.75% | 57.76% | 50.60% | +0.71 pp | +0.0030 |
| 29 | 94.87% | 61.04% | 54.52% | +6.68 pp | -0.0709 |
| 43 | 95.80% | 58.11% | 51.17% | +3.86 pp | -0.0479 |

The mean five-stratum multiclass Brier score improved from 0.61008 to 0.57147 for
ternary readers. The seed 17 worsening of 0.00300 remained within the 0.02 bound.
Every seed met the bounds on ordinary accuracy change and Brier worsening, but neither positive gains
nor acceptable calibration changes override the failed absolute-accuracy and
5-point mean-gain requirements.

Ordinary per-outcome accuracy shows a limitation hidden by the aggregate gain:
ternary Conflict accuracy fell from 13.67% to 11.46%, while Known and Unknown
accuracy improved. Aggregate gains therefore do not establish reliable conflict
handling. Renaming consistency improved, but remained imperfect. Record-order
consistency was 100% across all five groups in every condition.

| Weights | Reader | Known | Unknown | Conflict | Renaming consistency |
| --- | --- | --- | --- | --- | --- |
| float | metadata-only | 99.38% | 94.47% | 88.67% | 94.01% |
| float | enforced | 99.97% | 99.02% | 98.24% | 98.81% |
| ternary | metadata-only | 75.46% | 50.78% | 13.67% | 66.03% |
| ternary | enforced | 80.50% | 63.41% | 11.46% | 73.47% |

On ordinary queries, confidently unsupported entity predictions (probability
at least 0.9 and not the uniquely supported answer) averaged 11.33 per 2,048
examples for metadata-only ternary and 20.33 for enforced ternary. The corresponding
float averages were 8.33 and 0. Full group/hop/outcome, unsupported-confidence,
renaming/order and paired counterfactual metrics remain in every condition's
receipt. [Final metrics CSV](../../evidence/typed-state-v1/final-metrics.csv),
[all learning curves](../../evidence/typed-state-v1/learning-curves.csv),
[aggregate analysis](../../evidence/typed-state-v1/analysis.json), and
[complete summary](../../evidence/typed-state-v1/summary.json) retain the results.

## Correctness and provenance

All model runs used clean source
`4144184b1a83986f44feee36bcef3bfc0a086c84`, committed before formal execution.
[Provenance](../../evidence/typed-state-v1/provenance.json) records exact
source/config/binary/dataset identities. The clean corpus has 47,104 queries in
11,776 disjoint worlds. [Independent event replay](../../evidence/typed-state-v1/corpus/independent-python-audit.json)
verifies labels, primitive coverage, held-out constituents, equal branch nuisance
metadata and equal Known/Conflict active-observed degree. These checks establish
semantic/data correctness separately from learned-answer quality.

Every condition reloaded its actual step 200 on-disk checkpoint and independently
reconstructed the next 50 full 128 batches. Indices, labels, counters and cursors
matched; measured maximum loss, parameter/moment and logit errors were zero in
all twelve runs. The fixed GPU bound remained `1e-5 + 1e-4*abs(reference)`.
CPU serialized replay fixtures matched bit-for-bit. The
[independent results audit](../../evidence/typed-state-v1/independent-results-audit.json)
recomputed all predictions, checkpoint identities and the keyed whole-world
bootstrap, agreeing with the Rust analysis to 1e-12.

Pinned Burn/CubeK CUDA matmuls use TF32-rounded operands with f32 storage and
accumulation. An initial comparison to ideal-f64 arithmetic failed by about 2e-5.
Independent stage probes localized the discrepancy to operand conversion;
explicitly modeling NVIDIA nearest/ties-away TF32 arithmetic reduced direct
residuals to about 3e-8. Full forward/backward fixtures then passed the unchanged
declared thresholds. The [initial failure](../../evidence/typed-state-v1/validation/development/cuda-first-failure.txt)
and [precision probe](../../evidence/typed-state-v1/validation/development/cuda-stage-tf32-probe.txt)
remain preserved. These results do not claim agreement with ideal-f64 arithmetic.

All five repository checks passed on the final evidence branch; the
[verification receipt](../../evidence/typed-state-v1/validation/required/verification.json)
distinguishes that tree from the measured source. Earlier frozen-source and
explicit CUDA fixture logs remain in the evidence bundle. The evidence PR changes
documentation/artifacts only; measured code and recipe remain unchanged.

## Execution and memory boundaries

All conditions used the NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB, driver 615.78.08,
in sequential fresh processes. Experiment elapsed was 60.91–95.98 seconds,
including initialization, updates, validation/test, checkpoint writes, actual
50-update replay and sampling. Compilation and corpus audit/generation were
excluded. Outer process wall time is also retained. Earlier development probes
warmed compiler/runtime caches. These times show budget feasibility under these
conditions, without establishing a general throughput ranking.

Process VmHWM was 378,904–388,568 KiB. It includes corpus generation, data,
runtime and briefly a second replay model, even though corpus generation is
excluded from experiment elapsed. Sampled GPU process memory was 130 MiB in every
run and can miss transients. Host-read wait overlaps pending computation/transfers.
Latent/bias/moment/gradient payloads and hypothetical nonresident ternary export
size are separate quantities. [Checkpoint inventory](../../evidence/typed-state-v1/checkpoint-inventory.json)
records 24 local payload hashes; those large payloads remain uncommitted. All 60
per-world prediction files are committed as lossless gzip with raw/compressed
hashes and [exact rehydration commands](../../evidence/typed-state-v1/README.md).
No energy, custom-kernel or total-memory-savings conclusion follows.

## Decision

The evidence supports a positive average contract effect within this experiment,
while the fixed ternary reader remains below the predeclared quality targets and
handles conflicts poorly. The experiment does not identify the mechanism behind
that limitation. Issue 14 is complete as an experiment pending owner review;
[issue 15](https://github.com/kmosoti/tergon-lab/issues/15) remains gated and no
learned-storage pilot was implemented. No default changed.
