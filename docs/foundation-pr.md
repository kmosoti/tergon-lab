This establishes the first independently verifiable component for Tergon Lab:
absmean ternary quantization, a validated native two-bit representation, and an
explicit Burn CPU tensor bridge. A Rust-only workspace keeps numerical definitions,
execution, and experiment I/O separate.

The change also adds immutable run receipts, historical symbolic findings with
provenance, agent contribution rules, CI and evidence checks, issue/PR templates,
canonical wiki pages with drift-aware publishing, and disabled-by-default scoped
merge configuration. The baseline trusted owner workflow is already on main.

Validation: Rust format, Clippy with warnings denied, workspace tests, rustdoc,
repository checks, and three deterministic CPU smoke fixtures. Exact results and
source revisions are recorded in evidence. CUDA feature compilation is distinct
from GPU execution; see the verification report for the actual outcome.

This is an owner-required architecture/numerics/governance change. The larger
foundation is deliberate; future components are bounded issues and PRs. There is
no trained neural model, Muon implementation, packed GEMM, or offload scheduler yet.
The archived study's negative findings remain intact. App isolation and native
Wiki initialization require owner setup before their corresponding paths are used.
