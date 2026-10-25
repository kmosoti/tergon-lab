//! Fixed tied-hop typed-state memory reader. No generation, oracle labels or I/O.
use crate::state::{Evidence, Query, ReadPolicy, State};
use crate::{
    EngineError, ParameterSnapshot, PlacementPlan, TernaryLinear,
    training::{Parameter, floats},
};
use burn::tensor::{Bool, Tensor, TensorData, activation, loss};
use std::time::Instant;
/// Record feature width, fixed by the frozen protocol.
pub const RECORD_FEATURES: usize = 85;
/// Fixed latent width, with floating activations in both weight modes.
pub const WIDTH: usize = 64;
/// Six matrices in key/value/query/relation/update/decoder order.
pub const SHAPES: [(usize, usize); 6] = [(64, 85), (64, 85), (64, 20), (64, 5), (64, 64), (18, 64)];
/// Plain host model/optimizer state; serialization belongs to lab.
#[derive(Clone, Debug, PartialEq)]
pub struct ReasonerSnapshot {
    /// Alternating matrix/bias groups, in SHAPES order.
    pub parameters: Vec<ParameterSnapshot>,
}
struct Linear {
    matrix: Parameter<2>,
    bias: Parameter<1>,
}
impl Linear {
    fn forward(
        &self,
        x: Tensor<2>,
        training: bool,
        ternary: bool,
        wait: &mut f64,
    ) -> Result<Tensor<2>, EngineError> {
        let w = if training {
            self.matrix.value.clone()
        } else {
            self.matrix.value.clone().without_autodiff()
        };
        let b = if training {
            self.bias.value.clone()
        } else {
            self.bias.value.clone().without_autodiff()
        };
        if ternary {
            TernaryLinear::from_parameters(w, b).forward_profiled(x, wait)
        } else {
            Ok(x.matmul(w.transpose()) + b.unsqueeze::<2>())
        }
    }
}
/// Fixed six-map memory network, one tied hop per relation, unrestricted decoder.
pub struct StateReasoner {
    layers: Vec<Linear>,
    ternary: bool,
    policy: ReadPolicy,
    last_host_read_wait_seconds: f64,
}
/// Host-side feature buffers contain only public record/query metadata.
struct Features {
    records: Vec<f32>,
    query: Vec<f32>,
    operations: [Vec<f32>; 3],
    eligible: [Vec<bool>; 3],
    running: [Vec<f32>; 3],
}
fn features(batch: &[(State, Query)], policy: ReadPolicy) -> Result<Features, EngineError> {
    if batch.is_empty() || batch.len() > 1024 {
        return Err(EngineError::Shape);
    }
    let mut f = Features {
        records: vec![0.; batch.len() * 33 * 85],
        query: vec![0.; batch.len() * 20],
        operations: std::array::from_fn(|_| vec![0.; batch.len() * 5]),
        eligible: std::array::from_fn(|_| vec![false; batch.len() * 33]),
        running: std::array::from_fn(|_| vec![0.; batch.len()]),
    };
    for (n, (state, query)) in batch.iter().enumerate() {
        state.validate().map_err(|_| EngineError::Shape)?;
        query.validate().map_err(|_| EngineError::Shape)?;
        f.query[n * 20 + query.start] = 1.;
        f.query[n * 20 + 16 + state.types[query.start].index()] = 1.;
        for (slot, record) in state.records.iter().enumerate() {
            let a = &record.assertion;
            if a.id > 32 || a.time > 96 {
                return Err(EngineError::Shape);
            }
            let x = &mut f.records[(n * 33 + slot) * 85..(n * 33 + slot + 1) * 85];
            x[a.subject] = 1.;
            x[16 + a.object] = 1.;
            x[32 + a.relation.index()] = 1.;
            x[37 + state.types[a.subject].index()] = 1.;
            x[41 + state.types[a.object].index()] = 1.;
            x[45 + a.source] = 1.;
            x[49 + a.id as usize - 1] = 1.;
            x[81] = a.time as f32 / 96.;
            x[82] = f32::from(record.active);
            x[83] = f32::from(a.evidence == Evidence::Observed);
        }
        f.records[(n * 33 + 32) * 85 + 84] = 1.;
        for hop in 0..3 {
            f.eligible[hop][n * 33 + 32] = true;
            if let Some(relation) = query.relations.get(hop) {
                f.running[hop][n] = 1.;
                f.operations[hop][n * 5 + relation.index()] = 1.;
                for slot in 0..32 {
                    f.eligible[hop][n * 33 + slot] = state.eligible(slot, *relation, policy);
                }
            }
        }
    }
    Ok(f)
}
fn attention_read(
    keys: Tensor<3>,
    values: Tensor<3>,
    query: Tensor<2>,
    eligible: Tensor<2, Bool>,
) -> Tensor<2> {
    let [b, slots, width] = keys.dims();
    let attention = activation::softmax(
        (keys
            .matmul(query.reshape([b, width, 1]))
            .reshape([b, slots])
            / (width as f32).sqrt())
        .mask_fill(eligible.bool_not(), f32::NEG_INFINITY),
        1,
    );
    attention
        .reshape([b, 1, slots])
        .matmul(values)
        .reshape([b, width])
}

impl StateReasoner {
    /// Restore every parameter and AdamW state; fail before partial state is committed.
    pub fn restore(
        saved: &ReasonerSnapshot,
        ternary: bool,
        policy: ReadPolicy,
        plan: PlacementPlan,
    ) -> Result<Self, EngineError> {
        if saved.parameters.len() != 12
            || saved
                .parameters
                .iter()
                .any(|p| p.time != saved.parameters[0].time)
        {
            return Err(EngineError::Shape);
        }
        let layers = SHAPES
            .iter()
            .enumerate()
            .map(|(i, (out, input))| {
                Ok(Linear {
                    matrix: Parameter::restore(&saved.parameters[2 * i], [*out, *input], plan)?,
                    bias: Parameter::restore(&saved.parameters[2 * i + 1], [*out], plan)?,
                })
            })
            .collect::<Result<Vec<_>, EngineError>>()?;
        Ok(Self {
            layers,
            ternary,
            policy,
            last_host_read_wait_seconds: 0.,
        })
    }
    /// Synchronized host snapshot, retaining all moments and counters.
    pub fn snapshot(&self) -> Result<ReasonerSnapshot, EngineError> {
        let mut parameters = Vec::with_capacity(12);
        for l in &self.layers {
            parameters.push(l.matrix.snapshot()?);
            parameters.push(l.bias.snapshot()?);
        }
        Ok(ReasonerSnapshot { parameters })
    }
    fn forward(
        &self,
        batch: &[(State, Query)],
        training: bool,
        wait: &mut f64,
    ) -> Result<Tensor<2>, EngineError> {
        let f = features(batch, self.policy)?;
        let b = batch.len();
        let device = if training {
            self.layers[0].matrix.value.device()
        } else {
            self.layers[0].matrix.value.device().without_autodiff()
        };
        let records = Tensor::<2>::from_data(TensorData::new(f.records, [b * 33, 85]), &device);
        let keys = self.layers[0]
            .forward(records.clone(), training, self.ternary, wait)?
            .reshape([b, 33, 64]);
        let values = self.layers[1]
            .forward(records, training, self.ternary, wait)?
            .reshape([b, 33, 64]);
        let mut h = self.layers[2].forward(
            Tensor::from_data(TensorData::new(f.query, [b, 20]), &device),
            training,
            self.ternary,
            wait,
        )?;
        for hop in 0..3 {
            let op = Tensor::from_data(TensorData::new(f.operations[hop].clone(), [b, 5]), &device);
            let z = h.clone() + self.layers[3].forward(op, training, self.ternary, wait)?;
            let eligible = Tensor::<2, Bool>::from_data(
                TensorData::new(f.eligible[hop].clone(), [b, 33]),
                &device,
            );
            let read = attention_read(keys.clone(), values.clone(), z.clone(), eligible);
            let candidate =
                activation::relu(self.layers[4].forward(z, training, self.ternary, wait)? + read);
            let running =
                Tensor::<2>::from_data(TensorData::new(f.running[hop].clone(), [b, 1]), &device);
            h = candidate * running.clone() + h * (running.ones_like() - running);
        }
        self.layers[5].forward(h, training, self.ternary, wait)
    }
    /// Evaluate all 18 logits without oracle-derived output restrictions.
    pub fn logits(&self, batch: &[(State, Query)]) -> Result<Vec<f32>, EngineError> {
        floats(self.forward(batch, false, &mut 0.)?)
    }
    /// One final-answer mean-cross-entropy AdamW update. Labels are used only here.
    pub fn step(&mut self, batch: &[(State, Query)], labels: &[usize]) -> Result<f32, EngineError> {
        if labels.len() != batch.len() || labels.iter().any(|y| *y >= 18) {
            return Err(EngineError::Shape);
        }
        let mut wait = 0.;
        let logits = self.forward(batch, true, &mut wait)?;
        let device = logits.device();
        let mut targets = vec![0.; batch.len() * 18];
        for (n, y) in labels.iter().enumerate() {
            targets[n * 18 + y] = 1.;
        }
        let loss = loss::cross_entropy_with_logits(
            logits,
            Tensor::from_data(TensorData::new(targets, [batch.len(), 18]), &device),
        );
        let started = Instant::now();
        let value = floats(loss.clone())?[0];
        wait += started.elapsed().as_secs_f64();
        if !value.is_finite() {
            return Err(EngineError::NonFinite);
        }
        let mut gradients = loss.backward();
        let proposals = self
            .layers
            .iter()
            .map(|l| {
                Ok(Linear {
                    matrix: l.matrix.proposed(&mut gradients, 0.01, &mut wait)?,
                    bias: l.bias.proposed(&mut gradients, 0., &mut wait)?,
                })
            })
            .collect::<Result<Vec<_>, EngineError>>()?;
        self.layers = proposals;
        self.last_host_read_wait_seconds = wait;
        Ok(value)
    }
    /// Last successful update's synchronized host-read wait, overlapping wall time.
    pub fn last_host_read_wait_seconds(&self) -> f64 {
        self.last_host_read_wait_seconds
    }
}

#[cfg(test)]
#[path = "state_reasoner_tests.rs"]
mod tests;
