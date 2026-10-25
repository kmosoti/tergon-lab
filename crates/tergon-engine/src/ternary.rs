use burn::tensor::{Tensor, TensorData};
use std::time::Instant;
use tergon_math::{PackedTernary, QuantSpec};

use crate::{EngineError, PlacementPlan};

/// Trainable latent `[out, in]` weights and floating-point `[out]` bias.
///
/// Forward uses one canonical absmean scale per output row. Quantization runs
/// on host f64, including a synchronized device-to-host weight read each call;
/// decoded f32 values are then uploaded. Inputs are also read on host to reject
/// nonfinite activations. This is a correctness path, not a
/// packed kernel. See the mathematical contract for its identity STE.
#[derive(Clone, Debug)]
pub struct TernaryLinear {
    weights: Tensor<2>,
    bias: Tensor<1>,
}

impl TernaryLinear {
    pub(crate) fn from_parameters(weights: Tensor<2>, bias: Tensor<1>) -> Self {
        Self { weights, bias }
    }

    /// Validate finite host parameters, then allocate trainable device leaves.
    pub fn new(
        weights: &[f32],
        bias: &[f32],
        inputs: usize,
        placement: PlacementPlan,
    ) -> Result<Self, EngineError> {
        let outputs = bias.len();
        if inputs == 0 || outputs == 0 || outputs.checked_mul(inputs) != Some(weights.len()) {
            return Err(EngineError::Shape);
        }
        PackedTernary::quantize(weights, QuantSpec::new(inputs)?)?;
        if bias.iter().any(|v| !v.is_finite()) {
            return Err(EngineError::NonFinite);
        }
        let device = placement.device()?.autodiff();
        Ok(Self {
            weights: Tensor::from_data(
                TensorData::new(weights.to_vec(), [outputs, inputs]),
                &device,
            )
            .require_grad(),
            bias: Tensor::from_data(TensorData::new(bias.to_vec(), [outputs]), &device)
                .require_grad(),
        })
    }

    /// Latent trainable weights; cloning retains the autodiff leaf identity.
    pub fn weights(&self) -> Tensor<2> {
        self.weights.clone()
    }

    /// Floating-point trainable bias.
    pub fn bias(&self) -> Tensor<1> {
        self.bias.clone()
    }

    /// Immutable inference export, independent of the latent tensor lifecycle.
    pub fn packed(&self) -> Result<PackedTernary, EngineError> {
        self.packed_profiled(&mut 0.)
    }

    fn packed_profiled(&self, host_read_wait: &mut f64) -> Result<PackedTernary, EngineError> {
        let started = Instant::now();
        let values = self
            .weights
            .clone()
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|_| EngineError::Data)?;
        *host_read_wait += started.elapsed().as_secs_f64();
        Ok(PackedTernary::quantize(
            &values,
            QuantSpec::new(self.weights.dims()[1])?,
        )?)
    }

    /// Compute `[batch, in] Q(W)^T + b` on the parameter device.
    ///
    /// For upstream `G`, the chosen surrogate is `dW=G^T X`, `dX=G Q(W)`,
    /// `db=sum_batch G`. Scales and codes have no gradient. This is not the
    /// derivative of the discontinuous quantizer. Inputs must be finite.
    pub fn forward(&self, input: Tensor<2>) -> Result<Tensor<2>, EngineError> {
        self.forward_profiled(input, &mut 0.)
    }

    pub(crate) fn forward_profiled(
        &self,
        input: Tensor<2>,
        host_read_wait: &mut f64,
    ) -> Result<Tensor<2>, EngineError> {
        let [batch, inputs] = input.dims();
        if batch == 0 || inputs != self.weights.dims()[1] {
            return Err(EngineError::Shape);
        }
        if input.device() != self.weights.device() {
            return Err(EngineError::Device);
        }
        let started = Instant::now();
        let values = input
            .clone()
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|_| EngineError::Data)?;
        *host_read_wait += started.elapsed().as_secs_f64();
        if values.iter().any(|v| !v.is_finite()) {
            return Err(EngineError::NonFinite);
        }
        let packed = self.packed_profiled(host_read_wait)?;
        let quantized = Tensor::<2>::from_data(
            TensorData::new(packed.dequantize().collect::<Vec<_>>(), self.weights.dims()),
            &self.weights.device(),
        );
        // Subtract first: finite W-W is exactly zero, avoiding rounding Q+W-W.
        // detach preserves retention in Burn; explicitly disable it on the stop branch.
        let stop = self.weights.clone().detach().set_require_grad(false);
        let surrogate = quantized + (self.weights.clone() - stop);
        Ok(input.matmul(surrogate.transpose()) + self.bias.clone().unsqueeze::<2>())
    }
}
