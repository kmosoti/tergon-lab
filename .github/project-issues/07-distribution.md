# experiment: validate distributed execution after single-device baselines

Defer until the model, optimizer, kernel, and placement baselines are accepted.
Specify partitioning and synchronization mathematics; distinguish pipeline
inference from distributed optimizer semantics. Measure network latency/volume,
stragglers, recovery, and parity against one device under identical work. Do not
model separate GPUs as a transparent VRAM pool or assume LAN bandwidth is NVLink.
Any distributed Muon adaptation must cite prior work and isolate its own contribution.
