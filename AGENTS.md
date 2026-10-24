# Agent working agreement

Read CONTRIBUTING.md, docs/wiki/Architecture.md, and the component's mathematical
contract before editing. The linked issue defines scope and acceptance criteria.

1. Keep dependencies lab → engine → math; lab may also call math. xtask is tooling.
2. Keep math free of dependencies and I/O, engine free of external I/O, and
   placement explicit. Never substitute CPU for a requested GPU.
3. Preserve independent numerical references and predeclared tolerances. A
   quantizer's true derivative and a training surrogate are different contracts.
4. Make one bounded change per branch. Explain complexity with a concrete benefit.
   Do not add traits, macros, services, or dependencies for hypothetical uses.
5. Run `cargo fmt --all -- --check`, `cargo clippy --locked --workspace
   --all-targets -- -D warnings`, `cargo test --locked --workspace`,
   `RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps`, and
   `cargo xtask check`. Record actual outcomes, including failures and missing GPU checks.
6. Describe measured results, hypotheses, and future work separately. Cite borrowed
   algorithms. Never rewrite old findings to make a new result appear stronger.
7. Open a PR; do not push to main, merge, publish releases, or edit repository
   protections. Owner-only changes include APIs, numerics, oracles, tolerances,
   dependencies, formats, unsafe code, CI, governance, and new defaults.
8. Scoped merging is disabled. An agent cannot approve itself by changing a label,
   policy file, workflow, or scope. The owner must approve the exact head/base.
9. Keep credentials out of files, logs, fixtures, and agent instructions. App keys
   belong in a separate broker; contributor tokens never carry checks:write.

Wiki source lives in docs/wiki. Historical research under research/history is
archival evidence and is not part of this Rust runtime or its acceptance suite.
