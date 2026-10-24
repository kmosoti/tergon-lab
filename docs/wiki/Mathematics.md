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

No gradient is implemented yet. The true derivative of discrete quantization is
not the straight-through surrogate used in training. The next component must
state the surrogate explicitly and test it independently; a finite-difference
check of the discontinuous quantizer cannot validate a chosen surrogate.
