# Tergon Lab

Rust experiments in trainable ternary models, measured learning quality, and
efficient execution on consumer hardware. Biological efficiency is a research
motivation; it is not a demonstrated result of this project.

The first component defines and tests a native two-bit ternary representation,
an explicit Burn CPU tensor bridge, and reproducible component receipts. The
repository also contains contribution controls and preserved observations from
earlier symbolic-learning experiments. Neural training, Muon, packed kernels,
offload, and distributed execution are later milestones.

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
optional feature and has not been executed in this cloud environment. New code
is Apache-2.0; attribution and historical exceptions are in [NOTICE](NOTICE).

Publication is currently blocked by the connected GitHub integration's repository
creation permission. The source and history are local; see
[publication recovery](docs/PUBLISHING.md) for the prepared push, PR, and setup steps.
