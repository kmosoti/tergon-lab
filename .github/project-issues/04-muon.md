# experiment: compare Muon and AdamW under ternary training

Attribute original Muon and scaling/distributed prior work. Specify matrix parameter
selection, momentum, orthogonalization iterations, update RMS, and decoupled decay.
Use an independent small-matrix Rust reference, including zero/rank-deficient and
rectangular cases. Compare decay on/off and scale-aware updates under a fixed
quality/compute budget with three paired seeds. Log latent norms, sign/zero changes,
scale drift, optimizer state, temporaries, and held-out quality. Keep AdamW for
uncovered parameter classes. No default change without positive comparative evidence.
