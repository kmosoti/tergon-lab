//! Fixed MNIST classifier and AdamW state, with no file or dataset I/O.
use burn::optim::{AdamWConfig, AdamWState, AdaptiveMomentumState, Optimizer};
use burn::tensor::{Tensor, TensorData, activation, loss};
use crate::{EngineError, PlacementPlan, TernaryLinear};

/// Host snapshot of a parameter and its AdamW moments. Shape is checked on restore.
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterSnapshot {
    /// Row-major f32 parameter values.
    pub values: Vec<f32>,
    /// First moment, or empty before the first update.
    pub moment_1: Vec<f32>,
    /// Second moment, or empty before the first update.
    pub moment_2: Vec<f32>,
    /// Number of completed updates for this parameter.
    pub time: usize,
}

/// Fixed order: first weights, first bias, second weights, second bias.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassifierSnapshot {
    /// All four parameter groups and optimizer state.
    pub parameters: [ParameterSnapshot; 4],
}

struct Parameter<const D: usize> {
    value: Tensor<D>,
    state: Option<AdamWState<D>>,
}
fn floats<const D: usize>(value: Tensor<D>) -> Result<Vec<f32>, EngineError> {
    value.into_data().try_to_vec().map_err(|_| EngineError::Data)
}
fn finite<const D: usize>(value: Tensor<D>) -> Result<bool, EngineError> {
    value.without_autodiff().is_finite().all().into_data().try_to_vec::<bool>()
        .map(|v| v[0]).map_err(|_| EngineError::Data)
}
impl<const D: usize> Parameter<D> {
    fn restore(saved: &ParameterSnapshot, shape: [usize; D], plan: PlacementPlan) -> Result<Self, EngineError> {
        let len = shape.iter().product::<usize>();
        if saved.values.len() != len || saved.values.iter().any(|v| !v.is_finite()) {
            return Err(EngineError::Shape);
        }
        let device = plan.device()?.autodiff();
        let value = Tensor::from_data(TensorData::new(saved.values.clone(), shape), &device).require_grad();
        let state = if saved.time == 0 {
            if !saved.moment_1.is_empty() || !saved.moment_2.is_empty() { return Err(EngineError::Shape); }
            None
        } else {
            if saved.time > i32::MAX as usize || saved.moment_1.len() != len || saved.moment_2.len() != len
                || saved.moment_1.iter().any(|v| !v.is_finite())
                || saved.moment_2.iter().any(|v| !v.is_finite() || *v < 0.) { return Err(EngineError::Shape); }
            let device = device.without_autodiff();
            Some(AdamWState { momentum: AdaptiveMomentumState {
                time: saved.time,
                moment_1: Tensor::from_data(TensorData::new(saved.moment_1.clone(), shape), &device),
                moment_2: Tensor::from_data(TensorData::new(saved.moment_2.clone(), shape), &device),
                max_moment_2: None,
            }})
        };
        Ok(Self { value, state })
    }
    fn snapshot(&self) -> Result<ParameterSnapshot, EngineError> {
        Ok(ParameterSnapshot {
            values: floats(self.value.clone())?,
            moment_1: self.state.as_ref().map(|s| floats(s.momentum.moment_1.clone())).transpose()?.unwrap_or_default(),
            moment_2: self.state.as_ref().map(|s| floats(s.momentum.moment_2.clone())).transpose()?.unwrap_or_default(),
            time: self.state.as_ref().map_or(0, |s| s.momentum.time),
        })
    }
    fn proposed(&self, gradients: &mut burn::tensor::Gradients, decay: f32) -> Result<Self, EngineError> {
        let grad = self.value.grad_remove(gradients).ok_or(EngineError::Data)?;
        if !finite(grad.clone())? { return Err(EngineError::NonFinite); }
        let optimizer = AdamWConfig::new().with_beta_1(0.9).with_beta_2(0.999)
            .with_epsilon(1e-8).with_weight_decay(decay).with_amsgrad(false)
            .with_cautious_weight_decay(false).build();
        let (value, state) = optimizer.step(0.001, self.value.clone().without_autodiff(), grad, self.state.clone());
        if !finite(value.clone())? || state.as_ref().is_some_and(|s| s.momentum.time > i32::MAX as usize) { return Err(EngineError::NonFinite); }
        if let Some(s) = &state {
            if !finite(s.momentum.moment_1.clone())? || !finite(s.momentum.moment_2.clone())? { return Err(EngineError::NonFinite); }
        }
        Ok(Self { value: value.autodiff().require_grad(), state })
    }
}

/// Concrete 784 → 128 ReLU → 10 classifier; either both matrices ternary or both f32.
/// AdamW uses lr .001, betas .9/.999, epsilon 1e-8, matrix decay .01, bias decay 0.
pub struct TinyClassifier {
    w1: Parameter<2>, b1: Parameter<1>, w2: Parameter<2>, b2: Parameter<1>,
    ternary: bool,
}
impl TinyClassifier {
    /// Restore complete state after validation; no optimizer state is silently discarded.
    pub fn restore(saved: &ClassifierSnapshot, ternary: bool, plan: PlacementPlan) -> Result<Self, EngineError> {
        if saved.parameters.iter().any(|p| p.time != saved.parameters[0].time) { return Err(EngineError::Shape); }
        Ok(Self {
            w1: Parameter::restore(&saved.parameters[0], [128,784], plan)?,
            b1: Parameter::restore(&saved.parameters[1], [128], plan)?,
            w2: Parameter::restore(&saved.parameters[2], [10,128], plan)?,
            b2: Parameter::restore(&saved.parameters[3], [10], plan)?, ternary,
        })
    }
    /// Copy latent parameters and all optimizer moments to a plain host snapshot.
    pub fn snapshot(&self) -> Result<ClassifierSnapshot, EngineError> {
        Ok(ClassifierSnapshot { parameters: [self.w1.snapshot()?, self.b1.snapshot()?, self.w2.snapshot()?, self.b2.snapshot()?] })
    }
    fn forward(&self, inputs: &[f32]) -> Result<Tensor<2>, EngineError> {
        if inputs.is_empty() || !inputs.len().is_multiple_of(784) { return Err(EngineError::Shape); }
        if inputs.iter().any(|v| !v.is_finite()) { return Err(EngineError::NonFinite); }
        let x = Tensor::from_data(TensorData::new(inputs.to_vec(), [inputs.len()/784, 784]), &self.w1.value.device());
        let linear = |x: Tensor<2>, w: Tensor<2>, b: Tensor<1>| {
            if self.ternary { TernaryLinear { weights: w, bias: b }.forward(x) }
            else { Ok(x.matmul(w.transpose()) + b.unsqueeze::<2>()) }
        };
        linear(activation::relu(linear(x, self.w1.value.clone(), self.b1.value.clone())?), self.w2.value.clone(), self.b2.value.clone())
    }
    /// Evaluate logits. The caller owns batches, held-out splits, and reporting.
    pub fn logits(&self, inputs: &[f32]) -> Result<Vec<f32>, EngineError> { floats(self.forward(inputs)?) }
    /// One mean-cross-entropy update, retaining the last partial batch.
    pub fn step(&mut self, inputs: &[f32], labels: &[usize]) -> Result<f32, EngineError> {
        if labels.len().checked_mul(784) != Some(inputs.len()) || labels.iter().any(|y| *y >= 10) { return Err(EngineError::Shape); }
        let output = self.forward(inputs)?;
        let mut targets = vec![0.; labels.len()*10];
        for (n,y) in labels.iter().enumerate() { targets[n*10+y] = 1.; }
        let targets = Tensor::from_data(TensorData::new(targets, [labels.len(),10]), &output.device());
        let loss = loss::cross_entropy_with_logits(output, targets);
        let value = floats(loss.clone())?[0];
        if !value.is_finite() { return Err(EngineError::NonFinite); }
        let mut grads = loss.backward();
        // Construct all proposals before committing any parameter or moment state.
        let w1 = self.w1.proposed(&mut grads, 0.01)?;
        let b1 = self.b1.proposed(&mut grads, 0.)?;
        let w2 = self.w2.proposed(&mut grads, 0.01)?;
        let b2 = self.b2.proposed(&mut grads, 0.)?;
        self.w1 = w1; self.b1 = b1; self.w2 = w2; self.b2 = b2;
        Ok(value)
    }
}
