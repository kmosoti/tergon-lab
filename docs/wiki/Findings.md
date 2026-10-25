# Findings

## Current ternary component

The foundation tests numerical representation and CPU materialization. Recorded
receipts belong in evidence and identify the source/configuration. They do not
establish neural training quality, packed-kernel speed, GPU operation, or biological
efficiency. Muon has not yet been implemented or compared in this project.

The first verification passed ten tests and three deterministic CPU fixtures.
For each 1,025-weight fixture, 257 packed bytes and 20 scale bytes represented the
quantized values; the decoded and Burn-materialized bytes agreed exactly. This is
payload size and component correctness, not total runtime memory or model quality.
The optional CUDA build was blocked by the LLVM downloader's certificate rejection
in the cloud proxy environment; GPU execution is still unverified.

## Earlier symbolic-learning study

The preserved study ran 24 world blocks and 12,954 tasks. The EPIG/change learner
reduced unsupported confidence on random outcomes from 83.0% to 6.7%, while Brier
error improved from 0.460 to 0.301 and still missed the predeclared 0.270 target.
It recovered on 89/144 new relay regimes versus the baseline's 111/144, and its
stable-relay learning accuracy was 83.2% versus 92.4%.

Provisional typed concepts reduced target-only teaching answers on renamed pair
tasks from 23.2 to 16.7. Including source learning and validation reduced total
answers from 871.2 to 864.7, about 0.75%. None of 422 recorded concept instances
passed the predictive admission gate. These are bounded symbolic-world results,
not neural model or optimizer results.

The engineering lesson is to separate correctness, calibration, adaptation,
transfer, and total acquisition cost. A new mechanism can help one metric while
hurting another. Preserve the baseline and negative findings before adding more
components. Historical sources and exact files are linked in the repository's
[archive record](https://github.com/kmosoti/tergon-lab/tree/main/research/history).

## Trainable layer correctness (2026-10-25)

The bounded `[2,5]` layer fixture matches independent f64 scalar forward and
analytic identity-STE backward equations on Flex CPU and NVIDIA GeForce RTX 3060
Laptop GPU (driver 615.78.08). Zero rows, threshold ties, a packed-byte tail,
nonfinite values, invalid shapes, and overflow are covered. The new GPU fixture
uses its predeclared `1e-5 + 1e-5*abs(expected)` tolerance; existing tolerances
remain unchanged. All five required local checks passed.

The first CUDA readiness attempt lacked a system C++ compiler. Local user-space
compiler/headers and official NVIDIA NVRTC/NVCC 13.4.92 resolved provisioning,
without changing the driver. Earlier failure evidence remains unchanged.
[Receipt](../../evidence/ternary-layer-v1.json).

These are component correctness results. Training feasibility remains unmeasured
here. Host weight/input reads and decoded device buffers preclude a packed-kernel
or total-memory-savings conclusion.

## Fixed MNIST learning (2026-10-25)

All six predeclared runs completed ten epochs (4,300 updates each) on the NVIDIA
GeForce RTX 3060 Laptop GPU with the frozen AdamW recipe. Every ternary run exceeded
90% test accuracy and every paired float control exceeded 95%. The controls scored
higher in these runs; the experiment does not establish accuracy parity.

| Seed | Ternary test | Float test | Ternary wall seconds | Float wall seconds |
| --- | --- | --- | --- | --- |
| 17 | 96.44% | 97.91% | 26.41 | 16.64 |
| 29 | 96.86% | 97.71% | 24.07 | 16.71 |
| 43 | 96.80% | 97.67% | 26.33 | 18.65 |

Training loss fell from 0.492–0.508 to 0.080–0.083 for ternary runs, and from
0.368–0.383 to 0.032–0.035 for controls. Full loss and validation curves are in
[CSV](../../evidence/mnist-adamw-v1/learning-curves.csv) and the per-run receipts
listed by the [summary](../../evidence/mnist-adamw-v1/summary.json). Test evaluation
used the final fixed checkpoints, without retuning or selection on test results.

Every GPU run restored the step-200 JSON/checksum checkpoint, independently
reconstructed its next 50 batches/counters, and matched uninterrupted parameters
and optimizer moments with measured maximum absolute state error zero. Logits
passed the predeclared `1e-5 + 1e-4*abs(reference)` tolerance. Synthetic CPU JSON
restore tests match the next 50 updates after step 200 bit-for-bit for both modes.
A terminal checkpoint also passed the actual CLI resume path into a new directory,
with identical held-out metrics and no new updates. All five local checks passed.

Measured process VmHWM was 404,544–414,788 KiB; the sampled GPU process peak was
124 MiB in every run. VmHWM is cumulative across sequential runs, includes host
snapshots/data/runtime and briefly a second replay model, and is not isolated model
memory. GPU samples every 20 updates can miss transients. Analytic parameter,
gradient, moment and decoded-buffer counts are separate from this telemetry.
The 25,960-byte packed matrix export payload is not retained by training and is
not a claim about total memory savings.

Wall times include initialization, updates, evaluation, checkpoints, replay and
sampling; Rust compilation and initial IDX loading are excluded. Earlier GPU
smoke tests warmed some compiler/runtime caches. Host-read wait measurements
include pending device work and transfers and overlap update walltime. These
measurements establish the declared budget feasibility, not a general throughput
ranking. The task/model/seeds are small; no broad intelligence, inference-kernel,
energy, accuracy-parity or total-memory-saving conclusion follows.

Before measured learning, a CUDA finite reduction returned `Bool(U8)`, while a
strict native-bool host decode failed. Explicit supported storage conversion fixed
that error without bypassing finite checks; the development failure/probe and
clean GPU smoke are preserved. A separate startup attempt failed before learning
because the ignored output parent did not exist; creating it resolved startup.
No recipe or tolerance changed in response. Historical findings remain unchanged.
