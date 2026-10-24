# experiment: train a tiny model from scratch with AdamW

Build on the accepted layer. Predeclare a small task, architecture, training and
held-out splits, data hashes, budget, and three paired seeds. Use AdamW as baseline.
Specify parameter groups and decay, save/resume state with equivalence checks,
and report learning curves, held-out quality, failures, and all live memory costs.
Train on one consumer NVIDIA GPU before any distribution work. Acceptance is
reproducible learning and checkpoint behavior, not a claim of general intelligence.
