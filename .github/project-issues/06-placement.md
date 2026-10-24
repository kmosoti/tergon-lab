# experiment: measure explicit CPU GPU RAM and NVMe placement

Profile latent weights, packed weights, gradients, optimizer state, activations,
temporary buffers, and staging before designing a scheduler. Specify explicit
ownership and transfer boundaries. Compare a single-device baseline with one
offload policy; measure PCIe and DRAM bandwidth separately, prefetch stalls, bytes,
peak memory, failures, and quality. Use pinned buffers only when justified. Add
NVMe or asynchronous workers only after measurements identify a real need.
