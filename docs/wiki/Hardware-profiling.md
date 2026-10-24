# Hardware profiling

The current cloud validation has CPU access and no usable NVIDIA GPU. CPU results
do not validate the Razer or Dell GPUs, their drivers, memory bandwidth, or energy.
Treat previously supplied laptop specifications as claims to measure on each node.

Before a run, record CPU model/flags, cgroup CPU and RAM limits, RAM channels if
available, GPU name/VRAM/driver, OS/kernel, storage device, and network link. Use
`lscpu`, `/proc/meminfo`, cgroup files, and `nvidia-smi` where available. Capture
the commands and outputs as artifacts without credentials or unrelated host data.

Measure sustained memory bandwidth with a working set larger than cache. Nominal
DDR rate is not measured usable bandwidth. Measure PCIe transfers separately;
host RAM bandwidth is not CPU-to-GPU transfer bandwidth. Two GPUs on separate
machines do not form a transparent combined VRAM allocation.

Record synchronization and warmup for GPU timing. Report peak host RSS, live GPU
allocations, optimizer/activation/workspace budgets, and transfer staging. Avoid
shared CI timing as evidence for a performance improvement.

Use a real energy sensor with a stated sampling/integration method and idle
baseline. If unavailable, record energy as missing; elapsed time is not joules.
Do not change drivers, firmware, RAM configuration, or network exposure as part
of a numerical component test.
