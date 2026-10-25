# Mathematical conventions

This is numerical contract v1 for the first component. Changes require owner
review and a new comparison against the preserved reference.

Input is a contiguous slice of finite f32 weights. A positive group size partitions
it into consecutive groups; the last may be partial. Empty input is valid.
For a nonempty group of n weights, accumulate
`mean = sum(abs(f64(w_i))) / n` in index order, then store `scale = f32(mean)`.

If every weight is zero, the scale and codes are positive zero. If a nonzero mean
underflows to zero when stored as f32, return an error. NaN/infinite inputs are
errors. Accumulation and the stored-scale rounding are part of the contract.

For positive scale, `q_i = clamp(round_away(w_i / scale), -1, +1)` where division
and classification use f64 and `round_away` means nearest, with ties away from
zero. The implementation uses equivalent comparisons with `scale / 2` to avoid
division overflow. Reconstruction is `f32(scale * q_i)` with canonical +0.

Four trits occupy a byte, lowest two bits first: 00 is zero, 01 is +1, 10 is -1.
Code 11 is invalid. Unused high bits must be zero. Each group has one f32 scale;
scales must be finite and nonnegative, excluding -0. Zero-scale groups must have
only zero codes. Buffer lengths must agree with the declared weight count.

The packed payload is `ceil(N/4) + 4*ceil(N/group_size)` bytes. This excludes
Vec metadata, allocator capacity, tensor buffers, and runtime memory. It is a
native in-memory representation, not a stabilized file format or GGUF block type.

Quantization discards information. Agreement between two executions of these
quantized weights does not imply agreement with the original floating-point
model. Group scaling also differs from tensor-wide scaling in some BitNet recipes.

Tests use an independent f64 division/rounding oracle, exhaustive small packed
words, malformed encodings, threshold ties, zero/subnormal/extreme inputs, and
shape overflow. The CPU matrix reference uses absolute plus relative tolerance
`1e-6 + 1e-6*abs(expected)`; exact decoded values must match bit-for-bit.

## Trainable ternary linear contract

`TernaryLinear` owns latent finite f32 weights `[out, in]` and finite f32 bias
`[out]`, with positive dimensions and one scale per output row (`group_size=in`).
Finite f32 inputs have shape `[batch, in]`, with positive batch size. The forward
is `Y = X Q(W)^T + b`; activations and bias are floating point. Packed exports
are immutable snapshots, separate from the trainable latent parameters.

For upstream `G = dL/dY`, the identity straight-through training surrogate is
`dW = G^T X`, `dX = G Q(W)`, and `db = sum_batch G`. No gradient flows through
scales or ternary codes; no clipping is applied. This chosen surrogate is not
the true derivative of discrete quantization. Independent analytic equations,
not finite differences of the discontinuous quantizer, validate backward.

The implementation quantizes a synchronized host copy using the existing f64
contract, uploads the decoded values to the selected device, and computes
`Qdet + (W - stop_gradient(W))`. The subtraction happens before addition to
preserve finite forward values without rounding `Q+W-W`. Matrix products and
autodiff run on the explicitly selected device. This introduces host transfers
and decoded buffers every forward, plus a synchronized host input read for
nonfinite validation; it is not a packed training kernel.

The bounded layer fixture fixes CPU tolerance at `1e-6 + 1e-6*abs(expected)`
and NVIDIA GPU tolerance at `1e-5 + 1e-5*abs(expected)` before execution.
These new backward/forward fixtures do not relax existing bridge tolerances.


## Fixed MNIST classifier and optimizer

The concrete classifier has 784 inputs, 128 ReLU hidden units, and 10 logits.
Both matrices use the ternary contract above or both use f32. Biases and all
activations stay f32. Mean softmax cross entropy reduces the class sum then the
batch mean. The final partial batch participates in every epoch.

Burn 0.22.1 AdamW implements decoupled weight decay (Loshchilov and Hutter,
[2019](https://arxiv.org/abs/1711.05101)): learning rate .001, betas .9/.999,
epsilon 1e-8, weight decay .01 on matrices and zero on biases. AMSGrad, cautious
decay, clipping and scheduling are disabled. All four gradient and optimizer
proposals must be finite before any parameter/state is committed. Host snapshots
carry every parameter, first/second moment and update counter; restore validates
shape, finite state, nonnegative second moments and common counters.

The fixed experiment uses host He-uniform initialization, zero biases and
SplitMix64 (Sebastiano Vigna's public-domain reference, adapted to explicit saved
state), with rejection-sampled Fisher-Yates permutations. The paired modes use
identical initial parameters and batches. See the
[frozen recipe](../../experiments/mnist-adamw-v1.json) for the complete bounds,
data identities, quality targets and resume tolerances. CPU serialized state
replay after step 200 is required to match the next 50 updates bit-for-bit. GPU
state and logits use `1e-5 + 1e-4*abs(reference)` over those same updates.
