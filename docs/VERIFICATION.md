# Foundation verification: 2026-10-24

CPU component source: `90493482f08089b565cc9dadbc6915fed4f12d87`.
The smoke run used a clean checkout of that revision and the predeclared
[fixture configuration](../experiments/quantization-v1.json).

| Check | Observed result |
| --- | --- |
| Format, Clippy with warnings denied, rustdoc with warnings denied | Passed |
| Workspace tests | 10 passed: five quantization, three engine, one output-preservation, one wiki-drift test |
| Independent quantization oracle | 1,158 generated length/group combinations agreed |
| Packed representations | All 9,841 words of lengths zero through eight decoded correctly |
| CPU smoke fixtures | Three seeds, 1,025 weights each; decoded and materialized hashes identical |
| Architecture, evidence, docs, contribution checks | Passed |
| Workflow syntax | actionlint 1.7.7 passed; shellcheck was unavailable |
| Publication script syntax | bash -n passed; remote administrative actions remain untested |
| Optional CUDA feature build | Blocked by upstream LLVM downloader certificate rejection; exit 101 |
| GPU execution | Unavailable in this environment |
| GitHub repository creation | Both API routes rejected the integration with HTTP 403 |

Each smoke fixture used 257 packed bytes plus 20 scale bytes, compared with 4,100
bytes for the original f32 values. This is payload accounting only. The reference
bridge also allocates decoded and backend tensors; these measurements do not
establish reduced process memory or a faster kernel.

Quantization error against the original generated floats is nonzero and recorded
separately. Zero bridge error means the tensor preserved already-quantized values;
it does not mean ternary quantization is lossless. The fixtures are numerical
tests, not trained-model weights or an accuracy benchmark.

The host reports an Intel Xeon Platinum 8573C, a four-CPU cgroup quota, and a
16 GiB memory limit. No NVIDIA device node or nvidia-smi was available. No energy
measurement was taken. Do not extrapolate these checks to the two laptop GPUs.

Evidence: [machine-readable verification](../evidence/verification.json),
[CPU run receipt](../evidence/quantization-cpu.json),
[CUDA build failure](../evidence/cuda-compile-failure.log), and
[hash inventory](../evidence/inventory.json). The CUDA attempt used the same engine
and dependency code before the lab receipt's bit-comparison correction.

Neural training, AdamW/Muon comparison, packed GEMM, offload, and distribution are
future milestones. Scoped merges remain disabled. The App and remote
branch protections remain pending. Source, issues, and PR #9 are now published; see
[publication recovery](PUBLISHING.md).
