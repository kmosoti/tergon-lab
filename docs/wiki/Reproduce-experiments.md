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
