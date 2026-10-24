# Contributing

Start with an issue stating a problem, a bounded change, and observable acceptance
criteria. Use `feat/42-ternary-pack`, `fix/57-zero-scale`, or `exp/63-muon-decay`.
There is one main branch; no develop branch. Use Conventional Commits and squash
one reviewed change per PR. Set the optional local template with
`git config commit.template .gitmessage`.

An iteration is: agree on the contract, implement the smallest useful component,
compare against an independent reference, save the result, review, then decide
whether another component is justified. Negative findings are acceptable outputs.
Do not bundle an architectural rewrite into an experiment.

## Review boundaries

All PRs currently need a code-owner review and owner-review environment approval.
The initial owner-authored PR must be reopened by the contributor App before the
solo owner can approve it; GitHub forbids approving one's own PR. The planned scoped
agent path stays disabled until GitHub App identities and adversarial acceptance
tests pass. A future scope must already exist on main, name allowed files and
acceptance checks, and carry owner approval. More than 300 handwritten changed
lines removes scoped eligibility; that number is not a quality score.

Architecture, public APIs, numerical meaning, reference calculations, tolerances,
dependencies, unsafe code, persistent formats, backends, permissions, workflows,
releases, and new defaults always need owner review. Ambiguous changes do too.
CODEOWNERS routes review; labels and PR text never grant authority.

## Required local checks

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps
cargo xtask check
```

CI checks the crate graph, dependency allowlist, basic I/O boundary patterns,
evidence hashes, local documentation links, contribution configuration, and
numerical invariants. These checks are guardrails, not a proof of semantic
correctness. Review must still detect disguised I/O, altered assumptions, and
unjustified complexity. GPU changes need target-device evidence beyond CPU CI.

## Evidence and prose

Write the concrete behavior first. State shapes, units, ranges, error conditions,
and ownership where they affect a caller. Comments explain mathematical choices,
invariants, or tradeoffs rather than narrating syntax. Name a source and identify
adaptations when implementing a published algorithm.

Every experimental claim needs the run, baseline, code/config/data identifiers,
hardware, seeds, budget, tolerances, and limitations. Predeclare comparisons;
label post-hoc analyses. Report failures and peak memory as well as successes.
Do not infer joules from elapsed CPU time. Separate total model memory, packed
payload, latent weights, optimizer state, activations, allocator overhead, and I/O.

Prefer a function to a trait with one real implementation. New crates, macros,
dependencies, background workers, and abstraction layers need a measured or
testable benefit. Removing complexity is a valid contribution. Documentation
should have one canonical home; link instead of copying it into several pages.

Wiki changes go through docs/wiki and a PR. Publish the merged revision, with its
source SHA. Treat direct web edits as drift to reconcile, not a second authority.
