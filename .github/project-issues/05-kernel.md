# component: benchmark a packed ternary inference kernel

Preserve the approved format and numerical contract. Implement one kernel for
one named target; compare with scalar and dequantized Burn references. Test tails,
alignment, scale boundaries, shape overflow, and malformed input. Measure complete
inference including packing/materialization where applicable, peak live memory,
and synchronized timing after warmup. Accept a new default only if the declared
quality tolerance and useful end-to-end improvement both hold. No shared-CI speed claims.
