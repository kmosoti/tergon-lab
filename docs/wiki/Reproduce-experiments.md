# Reproduce experiments

Use a checkout of the stated commit, the pinned toolchain, and Cargo.lock. Run
from the repository root:

```bash
cargo test --locked --workspace
cargo xtask check
mkdir -p runs
cargo run --locked -p tergon-lab -- smoke experiments/quantization-v1.json runs/local.json
```

The command rejects an existing output path. Three seeds generate 1,025 finite
weights each, with groups of 256. It packs them, decodes them, materializes a Flex
CPU tensor, and checks exact decoded agreement. The input/output and configuration
hashes, revision, dirty flag, Rust version, device, and payload sizes are saved.

This is a deterministic correctness fixture. It does not compare training quality,
latency, throughput, peak process memory, or energy. The larger numerical suite
checks quantization independently. A successful packing run cannot validate a
language model or demonstrate a benefit from Muon.

The evidence inventory verifies committed file hashes and lengths. It establishes
integrity, not the truth of a claim or completeness of a statistical design.
Hardware-specific experiments additionally need hardware/driver metadata, data
hashes, warmup protocol, synchronized timing, raw failures, and repeated runs.

For a new study, commit the experiment declaration before execution. Store outputs
under a new run identifier. Preserve failed/partial output paths and command logs;
never repair a result in place. Large artifacts belong outside Git with verified
checksums and an explicit retrieval location. Do not mark an upload complete until
the remote bytes have been verified.

## Fixed MNIST learning experiment

The [frozen recipe](../../experiments/mnist-adamw-v1.json) fixes three paired seeds
(17, 29, 43), ternary/float modes, 784→128 ReLU→10, ten epochs and a 30-minute
budget per run. It uses canonical MNIST IDX files with committed compressed and
raw hashes. Download/decompress the four official resources outside the tracked
tree; the CLI reads raw IDX only and checks raw SHA256 before allocating a run.
The first 55,000 training records train, the remaining 5,000 validate, and the
canonical 10,000-record test split is evaluated after the final epoch.

With CUDA tools and an NVIDIA GPU explicitly provisioned, from a clean committed
source tree:

```bash
cargo build --locked --release -p tergon-lab --features cuda
mkdir -p runs
./target/release/tergon-lab train experiments/mnist-adamw-v1.json scratch/mnist runs/mnist-v1
```

Compilation and initial dataset loading are excluded from per-run budgets.
Initialization, training, validation/test evaluation, checkpoints, the step-200
serialized restore comparison for the next 50 updates, and process-memory
sampling are included. Updates and final completion enforce the fixed budget;
failures are saved and remaining paired runs continue without retuning.

Resume only into a new output directory; the old run is never replaced:

```bash
./target/release/tergon-lab train experiments/mnist-adamw-v1.json scratch/mnist runs/mnist-resume --resume runs/mnist-v1/ternary-17/step-200.json
```

A checkpoint is versioned JSON plus its `.sha256` sidecar. It includes all latent
parameters, AdamW moments/times, source/config/data identities, epoch/batch,
shuffle state/permutation and partial metrics. Restore rejects incompatible
identities, malformed state, checksums, shapes, counters or permutations. Epoch
checkpoints use the next epoch's normalized cursor. A new output directory keeps
prior artifacts intact. The format is specific to this experiment.

Each result records loss/validation curves, final held-out quality, failure,
process VmHWM and sampled process-specific NVIDIA memory. Host-read wait timings
include pending device work, finite-reduction kernels, synchronization and
transfer; they overlap total update walltime. Static parameter/gradient/moment
and decoded-buffer sizes describe numerical payloads, not total training memory.
The resume comparison briefly keeps a second model alive. Process VmHWM carries
high-water marks across sequential runs; GPU samples may miss transient peaks.
No inference-throughput, energy or total-memory-saving conclusion follows.
