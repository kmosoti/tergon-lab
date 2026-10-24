# component: implement the ternary trainable layer and surrogate contract

Specify latent f32 parameter shape [out,in], scale axes, quantized forward, and
the exact surrogate backward equation. Keep packed inference weights immutable.
Compare forward against scalar f64 and backward against the stated analytic
surrogate; do not use finite differences of discrete quantization as its oracle.
Validate zero/tail/nonfinite cases and CPU plus named NVIDIA GPU behavior. Fix
tolerances before running. No full model, optimizer research, or offload in this PR.
