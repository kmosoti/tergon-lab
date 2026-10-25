# Tergon Lab

Rust experiments in trainable ternary models, measured learning quality, and
efficient execution on consumer hardware. Biological efficiency is a research
motivation; it is not a demonstrated result of this project.

The runtime includes a native two-bit ternary representation, explicit CPU/CUDA
placement, a trainable ternary linear layer with an identity surrogate, and a
fixed MNIST classifier trained with AdamW. The repository also preserves earlier
symbolic-learning observations. Muon, packed kernels, offload and distributed
execution remain later milestones.

Three fixed ternary MNIST runs reached 96.44–96.86% test accuracy, with paired f32
controls at 97.67–97.91%, on an NVIDIA RTX 3060 Laptop GPU. All predeclared quality
targets and checkpoint replay checks passed. This is a small-task learning result,
with host quantization and decoded buffers; it makes no throughput, energy,
accuracy-parity or total-memory-saving claim. See the
[measured summary](evidence/mnist-adamw-v1/summary.json) and
[learning curves](evidence/mnist-adamw-v1/learning-curves.csv).

```bash
cargo test --locked --workspace
cargo xtask check
mkdir -p runs
cargo run --locked -p tergon-lab -- smoke experiments/quantization-v1.json runs/quantization.json
```

Run from a Git checkout. The receipt records its revision, dirty state, config
hash, fixture hashes, toolchain, device, and limitations. Existing outputs are
never overwritten. This smoke run tests correctness, not model quality or speed.

- [Architecture](docs/wiki/Architecture.md) and [mathematical contract](docs/wiki/Mathematics.md)
- [Component roadmap](docs/wiki/Component-development.md)
- [Reproduction](docs/wiki/Reproduce-experiments.md) and [findings](docs/wiki/Findings.md)
- [Foundation verification](docs/VERIFICATION.md): ten tests and three CPU fixtures passed
- [Contributing](CONTRIBUTING.md), [agent agreement](AGENTS.md), and [GitHub App setup](docs/wiki/GitHub-App-setup.md)
- [Historical evidence and source](research/history/README.md)

Rust 1.99.1 and Burn 0.22.1 are pinned. CPU uses Burn Flex; CUDA is an explicit
optional feature; named-device evidence is recorded in the linked findings. New code
is Apache-2.0; attribution and historical exceptions are in [NOTICE](NOTICE).

The foundation was published in [PR #9](https://github.com/kmosoti/tergon-lab/pull/9).
Repository protection settings still require owner administration access; see
[publication status](docs/PUBLISHING.md).
