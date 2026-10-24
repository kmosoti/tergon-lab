# ADR 0001: A Rust component laboratory

Status: proposed for owner review in the foundation PR.

The next research target is a small ternary neural model trained from scratch.
Earlier symbolic results do not establish a neural architecture or optimizer.
We need independently checkable components before a full model comparison.

Use three application crates and one developer tool. math defines numerics;
engine adapts those definitions to Burn; lab owns I/O and experiments; xtask
checks the repository. Pin Rust 1.99.1 and Burn 0.22.1. Start with Flex CPU and
add explicit CUDA execution only with target-device evidence. Use Rust f64
references rather than maintaining a second Python implementation.

The first deliverable is quantization, native packing, a reference tensor bridge,
and receipts. Do not add a training trait, an optimizer enum, or checkpoint format
until a component actually consumes it. OptimizerSpec and a versioned checkpoint
schema belong to their later milestones. PlacementPlan is single-device today.

Tradeoff: Burn adds substantial transitive dependencies, but supplies the tensor
and differentiation substrate. We avoid building another autodiff framework.
Pinned builds and explicit features make updates reviewable. A scalar f64 oracle
keeps the quantization contract independent of the tensor backend.

All changes initially need owner approval. Scoped agent merging remains disabled
until App permissions and trusted check provenance are verified separately.
