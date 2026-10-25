use super::*;
use crate::{ComputeDevice, state::*};
fn initial() -> ReasonerSnapshot {
    let mut index = 0usize;
    ReasonerSnapshot {
        parameters: SHAPES
            .iter()
            .flat_map(|(out, input)| [(*out * *input, *input), (*out, 0)])
            .map(|(n, fan)| ParameterSnapshot {
                values: (0..n)
                    .map(|_| {
                        index += 1;
                        if fan == 0 {
                            0.09
                        } else {
                            ((index * 17 % 103) as f32 - 51.) / 1000.
                        }
                    })
                    .collect(),
                moment_1: vec![],
                moment_2: vec![],
                time: 0,
            })
            .collect(),
    }
}
fn context() -> (State, Query) {
    let types = std::array::from_fn(|i| {
        [
            EntityType::Person,
            EntityType::Item,
            EntityType::Container,
            EntityType::Place,
        ][i / 4]
    });
    let records = vec![
        Record {
            assertion: Assertion {
                id: 1,
                subject: 4,
                relation: Relation::Owner,
                object: 0,
                source: 2,
                time: 1,
                evidence: Evidence::Observed,
            },
            active: true,
        },
        Record {
            assertion: Assertion {
                id: 2,
                subject: 0,
                relation: Relation::At,
                object: 12,
                source: 0,
                time: 2,
                evidence: Evidence::Observed,
            },
            active: true,
        },
        Record {
            assertion: Assertion {
                id: 3,
                subject: 12,
                relation: Relation::Link,
                object: 13,
                source: 1,
                time: 3,
                evidence: Evidence::Hypothesis,
            },
            active: true,
        },
        Record {
            assertion: Assertion {
                id: 4,
                subject: 12,
                relation: Relation::Link,
                object: 14,
                source: 3,
                time: 4,
                evidence: Evidence::Observed,
            },
            active: false,
        },
    ];
    (
        State { types, records },
        Query {
            start: 4,
            relations: vec![Relation::Owner, Relation::At, Relation::Link],
        },
    )
}
fn near(actual: &[f32], expected: &[f64], gpu: bool) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        let tolerance = if gpu {
            1e-5 + 1e-5 * b.abs()
        } else {
            1e-6 + 1e-6 * b.abs()
        };
        assert!(
            (*a as f64 - *b).abs() <= tolerance,
            "{i}: {a} vs {b} tol{tolerance}"
        );
    }
}
// NVIDIA cvt.rna.tf32.f32: discard13fraction bits, nearest with ties away.
fn tf32(value: f64) -> f64 {
    let bits = (value as f32).to_bits();
    let magnitude = bits & 0x7fff_ffff;
    let rounded = magnitude.wrapping_add(0x1000) & !0x1fff;
    f32::from_bits((bits & 0x8000_0000) | rounded) as f64
}
fn stored(value: f64, gpu: bool) -> f64 {
    if gpu { value as f32 as f64 } else { value }
}
fn operand(value: f64, gpu: bool) -> f64 {
    if gpu { tf32(value) } else { value }
}
fn precision_linear(x: &[f64], w: &[f64], bias: &[f64], gpu: bool) -> Vec<f64> {
    if !gpu {
        return linear(x, w, bias);
    }
    let n = x.len();
    bias.iter()
        .enumerate()
        .map(|(i, b)| {
            stored(
                stored((0..n).map(|j| tf32(x[j]) * tf32(w[i * n + j])).sum(), true) + b,
                true,
            )
        })
        .collect()
}
#[test]
fn tf32_halfway_rounds_away_from_zero() {
    let half = f32::from_bits(1.0_f32.to_bits() + 0x1000);
    let next = f32::from_bits(1.0_f32.to_bits() + 0x2000);
    assert_eq!(tf32(half as f64), next as f64);
    assert_eq!(tf32(-half as f64), -next as f64);
    assert_eq!(tf32(f32::from_bits(half.to_bits() - 1) as f64), 1.);
    assert_eq!(tf32(-f32::from_bits(half.to_bits() - 1) as f64), -1.);
    assert_eq!((tf32(-0.) as f32).to_bits(), (-0.0_f32).to_bits());
}
fn linear(x: &[f64], w: &[f64], bias: &[f64]) -> Vec<f64> {
    let n = x.len();
    bias.iter()
        .enumerate()
        .map(|(row, b)| b + (0..n).map(|j| w[row * n + j] * x[j]).sum::<f64>())
        .collect()
}
fn backward_linear(x: &[f64], w: &[f64], dy: &[f64], dw: &mut [f64], db: &mut [f64]) -> Vec<f64> {
    let n = x.len();
    let mut dx = vec![0.; n];
    for i in 0..dy.len() {
        db[i] += dy[i];
        for j in 0..n {
            dw[i * n + j] += dy[i] * x[j];
            dx[j] += dy[i] * w[i * n + j];
        }
    }
    dx
}
fn precision_backward_linear(
    x: &[f64],
    w: &[f64],
    dy: &[f64],
    dw: &mut [f64],
    db: &mut [f64],
    gpu: bool,
) -> Vec<f64> {
    if !gpu {
        return backward_linear(x, w, dy, dw, db);
    }
    let n = x.len();
    let mut dx = vec![0.; n];
    for j in 0..n {
        dx[j] = stored(
            (0..dy.len())
                .map(|i| tf32(dy[i]) * tf32(w[i * n + j]))
                .sum(),
            true,
        );
    }
    for i in 0..dy.len() {
        db[i] = stored(db[i] + dy[i], true);
        for j in 0..n {
            dw[i * n + j] = stored(dw[i * n + j] + stored(tf32(dy[i]) * tf32(x[j]), true), true);
        }
    }
    dx
}
// Independent division/rounding quantizer (no PackedTernary or engine quantizer).
fn weights(p: &ParameterSnapshot, inputs: usize, ternary: bool) -> Vec<f64> {
    if !ternary {
        return p.values.iter().map(|x| *x as f64).collect();
    }
    p.values
        .chunks(inputs)
        .flat_map(|row| {
            let scale = (row.iter().map(|x| (*x as f64).abs()).sum::<f64>() / inputs as f64) as f32;
            row.iter().map(move |x| {
                if scale == 0. {
                    0.
                } else {
                    (scale * (((*x as f64 / scale as f64).round().clamp(-1., 1.)) as f32)) as f64
                }
            })
        })
        .collect()
}
fn scalar(
    snapshot: &ReasonerSnapshot,
    state: &State,
    query: &Query,
    policy: ReadPolicy,
    ternary: bool,
    gpu: bool,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let w: Vec<_> = SHAPES
        .iter()
        .enumerate()
        .map(|(i, (_, input))| weights(&snapshot.parameters[2 * i], *input, ternary))
        .collect();
    let bias: Vec<Vec<_>> = (0..6)
        .map(|i| {
            snapshot.parameters[2 * i + 1]
                .values
                .iter()
                .map(|x| *x as f64)
                .collect()
        })
        .collect();
    let mut x = vec![vec![0.; 85]; 33];
    // Independent metadata encoding, intentionally not calling features().
    for (slot, r) in state.records.iter().enumerate() {
        let a = &r.assertion;
        x[slot][a.subject] = 1.;
        x[slot][16 + a.object] = 1.;
        x[slot][32 + a.relation.index()] = 1.;
        x[slot][37 + state.types[a.subject].index()] = 1.;
        x[slot][41 + state.types[a.object].index()] = 1.;
        x[slot][45 + a.source] = 1.;
        x[slot][49 + a.id as usize - 1] = 1.;
        x[slot][81] = (a.time as f32 / 96.) as f64;
        x[slot][82] = f64::from(r.active);
        x[slot][83] = f64::from(a.evidence == Evidence::Observed);
    }
    x[32][84] = 1.;
    let k: Vec<_> = x
        .iter()
        .map(|x| precision_linear(x, &w[0], &bias[0], gpu))
        .collect();
    let v: Vec<_> = x
        .iter()
        .map(|x| precision_linear(x, &w[1], &bias[1], gpu))
        .collect();
    let mut start = vec![0.; 20];
    start[query.start] = 1.;
    start[16 + state.types[query.start].index()] = 1.;
    let mut h = precision_linear(&start, &w[2], &bias[2], gpu);
    let mut caches = Vec::new();
    for relation in &query.relations {
        let mut op = vec![0.; 5];
        op[relation.index()] = 1.;
        let projected = precision_linear(&op, &w[3], &bias[3], gpu);
        let z: Vec<_> = h
            .iter()
            .zip(projected)
            .map(|(h, r)| stored(h + r, gpu))
            .collect();
        let eligible: Vec<_> = (0..33)
            .map(|slot| {
                slot == 32
                    || (slot < state.records.len()
                        && (policy == ReadPolicy::MetadataOnly || {
                            let r = &state.records[slot];
                            r.active
                                && r.assertion.evidence == Evidence::Observed
                                && r.assertion.relation == *relation
                        }))
            })
            .collect();
        let scores: Vec<_> = (0..33)
            .map(|i| {
                if eligible[i] {
                    stored(
                        (0..64)
                            .map(|j| operand(k[i][j], gpu) * operand(z[j], gpu))
                            .sum::<f64>(),
                        gpu,
                    ) / 8.
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum = scores.iter().map(|s| (s - max).exp()).sum::<f64>();
        let p: Vec<_> = scores
            .iter()
            .map(|s| stored((s - max).exp() / sum, gpu))
            .collect();
        let mut pre = precision_linear(&z, &w[4], &bias[4], gpu);
        for j in 0..64 {
            pre[j] = stored(
                pre[j]
                    + stored(
                        (0..33)
                            .map(|i| operand(p[i], gpu) * operand(v[i][j], gpu))
                            .sum::<f64>(),
                        gpu,
                    ),
                gpu,
            );
        }
        h = pre.iter().map(|a| a.max(0.)).collect();
        caches.push((z, op, p, pre));
    }
    let logits = precision_linear(&h, &w[5], &bias[5], gpu);
    let mut grads: Vec<_> = snapshot
        .parameters
        .iter()
        .map(|p| vec![0.; p.values.len()])
        .collect();
    let mut apply = |layer: usize, input: &[f64], dy: &[f64]| {
        let split = 2 * layer + 1;
        let (a, b) = grads.split_at_mut(split);
        precision_backward_linear(input, &w[layer], dy, &mut a[split - 1], &mut b[0], gpu)
    };
    let upstream: Vec<_> = (0..18)
        .map(|i| stored((i as f64 - 8.) / 10., gpu))
        .collect();
    let mut dh = apply(5, &h, &upstream);
    let mut dk = vec![vec![0.; 64]; 33];
    let mut dv = dk.clone();
    for (z, op, p, pre) in caches.into_iter().rev() {
        let dt: Vec<_> = dh
            .iter()
            .zip(pre)
            .map(|(d, y)| if y > 0. { *d } else { 0. })
            .collect();
        let mut dz = apply(4, &z, &dt);
        let dp: Vec<_> = (0..33)
            .map(|i| {
                stored(
                    (0..64)
                        .map(|j| operand(dt[j], gpu) * operand(v[i][j], gpu))
                        .sum::<f64>(),
                    gpu,
                )
            })
            .collect();
        let pdp = stored((0..33).map(|i| stored(p[i] * dp[i], gpu)).sum::<f64>(), gpu);
        let ds: Vec<_> = (0..33)
            .map(|i| stored(stored(p[i] * stored(dp[i] - pdp, gpu), gpu) / 8., gpu))
            .collect();
        for i in 0..33 {
            for j in 0..64 {
                dv[i][j] = stored(
                    dv[i][j] + stored(operand(p[i], gpu) * operand(dt[j], gpu), gpu),
                    gpu,
                );
                dk[i][j] = stored(
                    dk[i][j] + stored(operand(ds[i], gpu) * operand(z[j], gpu), gpu),
                    gpu,
                );
            }
        }
        for j in 0..64 {
            dz[j] = stored(
                dz[j]
                    + stored(
                        (0..33)
                            .map(|i| operand(ds[i], gpu) * operand(k[i][j], gpu))
                            .sum::<f64>(),
                        gpu,
                    ),
                gpu,
            );
        }
        apply(3, &op, &dz);
        dh = dz;
    }
    apply(2, &start, &dh);
    for i in 0..33 {
        apply(0, &x[i], &dk[i]);
        apply(1, &x[i], &dv[i]);
    }
    (logits, grads)
}
fn scalar_fixture(compute: ComputeDevice) {
    let plan = PlacementPlan { compute };
    let gpu = matches!(compute, ComputeDevice::Cuda(_));
    let snapshot = initial();
    for ternary in [false, true] {
        for policy in [ReadPolicy::MetadataOnly, ReadPolicy::Enforced] {
            let model = StateReasoner::restore(&snapshot, ternary, policy, plan).unwrap();
            let (state, query) = context();
            let (expected, gradients) = scalar(&snapshot, &state, &query, policy, ternary, gpu);
            let public_logits = model.logits(&[(state.clone(), query.clone())]).unwrap();
            let output = model.forward(&[(state, query)], true, &mut 0.).unwrap();
            near(&public_logits, &expected, gpu);
            near(&floats(output.clone()).unwrap(), &expected, gpu);
            let upstream: Vec<_> = (0..18).map(|i| (i as f32 - 8.) / 10.).collect();
            let objective = (output
                * Tensor::from_data(
                    TensorData::new(upstream, [1, 18]),
                    &plan.device().unwrap().autodiff(),
                ))
            .sum();
            let mut grads = objective.backward();
            for (i, l) in model.layers.iter().enumerate() {
                near(
                    &floats(l.matrix.value.grad_remove(&mut grads).unwrap()).unwrap(),
                    &gradients[2 * i],
                    gpu,
                );
                near(
                    &floats(l.bias.value.grad_remove(&mut grads).unwrap()).unwrap(),
                    &gradients[2 * i + 1],
                    gpu,
                );
            }
        }
    }
}
#[test]
fn scalar_forward_backward_cpu() {
    scalar_fixture(ComputeDevice::Cpu);
}
#[cfg(feature = "cuda")]
#[test]
fn scalar_forward_backward_cuda() {
    scalar_fixture(ComputeDevice::Cuda(0));
}
fn attention_fixture(compute: ComputeDevice) {
    let device = PlacementPlan { compute }.device().unwrap().autodiff();
    let keys = Tensor::<3>::from_data(
        TensorData::new(vec![0.2, -0.3, 0.5, 0.8, -0.1, 0.4], [1, 3, 2]),
        &device,
    )
    .require_grad();
    let values = Tensor::<3>::from_data(
        TensorData::new(vec![2., 3., 100., 200., 5., 7.], [1, 3, 2]),
        &device,
    )
    .require_grad();
    let query =
        Tensor::<2>::from_data(TensorData::new(vec![0.6, -0.2], [1, 2]), &device).require_grad();
    let mask =
        Tensor::<2, Bool>::from_data(TensorData::new(vec![false, false, true], [1, 3]), &device);
    let output = attention_read(keys.clone(), values.clone(), query.clone(), mask);
    assert_eq!(floats(output.clone()).unwrap(), vec![5., 7.]);
    let mut grads = output.sum().backward();
    assert!(
        floats(keys.grad_remove(&mut grads).unwrap())
            .unwrap()
            .iter()
            .all(|v| *v == 0.)
    );
    assert_eq!(
        floats(values.grad_remove(&mut grads).unwrap()).unwrap(),
        vec![0., 0., 0., 0., 1., 1.]
    );
    assert_eq!(
        floats(query.grad_remove(&mut grads).unwrap()).unwrap(),
        vec![0., 0.]
    );
}
#[test]
fn null_only_and_masked_gradients_cpu() {
    attention_fixture(ComputeDevice::Cpu);
}
#[cfg(feature = "cuda")]
#[test]
fn null_only_and_masked_gradients_cuda() {
    attention_fixture(ComputeDevice::Cuda(0));
}
#[test]
fn encoder_rejects_unsupported_id_and_time() {
    let (mut state, q) = context();
    state.records[0].assertion.id = 33;
    assert!(features(&[(state.clone(), q.clone())], ReadPolicy::Enforced).is_err());
    state.records[0].assertion.id = 1;
    state.records[0].assertion.time = 97;
    assert!(features(&[(state, q)], ReadPolicy::Enforced).is_err());
}
fn mixed_batch_fixture(compute: ComputeDevice) {
    let gpu = matches!(compute, ComputeDevice::Cuda(_));
    let plan = PlacementPlan { compute };
    let snapshot = initial();
    let (state, query) = context();
    let batch: Vec<_> = [2, 1, 3]
        .map(|hops| {
            (
                state.clone(),
                Query {
                    start: query.start,
                    relations: query.relations[..hops].to_vec(),
                },
            )
        })
        .into();
    for ternary in [false, true] {
        for policy in [ReadPolicy::MetadataOnly, ReadPolicy::Enforced] {
            let mut model = StateReasoner::restore(&snapshot, ternary, policy, plan).unwrap();
            let expected: Vec<_> = batch
                .iter()
                .flat_map(|(s, q)| scalar(&snapshot, s, q, policy, ternary, gpu).0)
                .collect();
            near(&model.logits(&batch).unwrap(), &expected, gpu);
            let labels = [3, 7, 16];
            let mut expected_loss = 0.;
            for (i, label) in labels.iter().enumerate() {
                let row = &expected[i * 18..(i + 1) * 18];
                let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                expected_loss +=
                    (row.iter().map(|x| (x - max).exp()).sum::<f64>()).ln() + max - row[*label];
            }
            expected_loss /= 3.;
            near(
                &[model.step(&batch, &labels).unwrap()],
                &[expected_loss],
                gpu,
            );
        }
    }
}
#[test]
fn mixed_batch_and_mean_loss_cpu() {
    mixed_batch_fixture(ComputeDevice::Cpu);
}
#[cfg(feature = "cuda")]
#[test]
fn mixed_batch_and_mean_loss_cuda() {
    mixed_batch_fixture(ComputeDevice::Cuda(0));
}
fn partial_mask_fixture(compute: ComputeDevice) {
    let gpu = matches!(compute, ComputeDevice::Cuda(_));
    let device = PlacementPlan { compute }.device().unwrap().autodiff();
    let kh = [0.2_f32, -0.3, 0.5, 0.8, -0.1, 0.4];
    let vh = [2.0_f32, 3., 100., 200., 5., 7.];
    let qh = [0.6_f32, -0.2];
    let keys =
        Tensor::<3>::from_data(TensorData::new(kh.to_vec(), [1, 3, 2]), &device).require_grad();
    let values =
        Tensor::<3>::from_data(TensorData::new(vh.to_vec(), [1, 3, 2]), &device).require_grad();
    let query =
        Tensor::<2>::from_data(TensorData::new(qh.to_vec(), [1, 2]), &device).require_grad();
    let mask =
        Tensor::<2, Bool>::from_data(TensorData::new(vec![true, false, true], [1, 3]), &device);
    let output = attention_read(keys.clone(), values.clone(), query.clone(), mask);
    let scale = stored(2_f64.sqrt(), gpu);
    let score = |i: usize| {
        stored(
            stored(
                (0..2)
                    .map(|j| operand(kh[2 * i + j] as f64, gpu) * operand(qh[j] as f64, gpu))
                    .sum::<f64>(),
                gpu,
            ) / scale,
            gpu,
        )
    };
    let p0 = 1. / (1. + (score(2) - score(0)).exp());
    let p = [stored(p0, gpu), 0., stored(1. - p0, gpu)];
    let expected: Vec<_> = (0..2)
        .map(|j| {
            stored(
                (0..3)
                    .map(|i| operand(p[i], gpu) * operand(vh[2 * i + j] as f64, gpu))
                    .sum::<f64>(),
                gpu,
            )
        })
        .collect();
    near(&floats(output.clone()).unwrap(), &expected, gpu);
    let upstream = [0.7_f32, -0.2];
    let mut grads = (output
        * Tensor::from_data(TensorData::new(upstream.to_vec(), [1, 2]), &device))
    .sum()
    .backward();
    let dp: Vec<_> = (0..3)
        .map(|i| {
            stored(
                (0..2)
                    .map(|j| operand(upstream[j] as f64, gpu) * operand(vh[2 * i + j] as f64, gpu))
                    .sum::<f64>(),
                gpu,
            )
        })
        .collect();
    let mean = stored((0..3).map(|i| stored(p[i] * dp[i], gpu)).sum::<f64>(), gpu);
    let ds: Vec<_> = (0..3)
        .map(|i| stored(stored(p[i] * stored(dp[i] - mean, gpu), gpu) / scale, gpu))
        .collect();
    let dk: Vec<_> = (0..6)
        .map(|n| {
            stored(
                operand(ds[n / 2], gpu) * operand(qh[n % 2] as f64, gpu),
                gpu,
            )
        })
        .collect();
    let dv: Vec<_> = (0..6)
        .map(|n| {
            stored(
                operand(p[n / 2], gpu) * operand(upstream[n % 2] as f64, gpu),
                gpu,
            )
        })
        .collect();
    let dq: Vec<_> = (0..2)
        .map(|j| {
            stored(
                (0..3)
                    .map(|i| operand(ds[i], gpu) * operand(kh[2 * i + j] as f64, gpu))
                    .sum::<f64>(),
                gpu,
            )
        })
        .collect();
    near(
        &floats(keys.grad_remove(&mut grads).unwrap()).unwrap(),
        &dk,
        gpu,
    );
    near(
        &floats(values.grad_remove(&mut grads).unwrap()).unwrap(),
        &dv,
        gpu,
    );
    near(
        &floats(query.grad_remove(&mut grads).unwrap()).unwrap(),
        &dq,
        gpu,
    );
}
#[test]
fn partial_mask_gradients_cpu() {
    partial_mask_fixture(ComputeDevice::Cpu);
}
#[cfg(feature = "cuda")]
#[test]
fn partial_mask_gradients_cuda() {
    partial_mask_fixture(ComputeDevice::Cuda(0));
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "development precision probe; run explicitly before numerical freeze"]
fn cuda_stage_precision_probe() {
    let snapshot = initial();
    let plan = PlacementPlan {
        compute: ComputeDevice::Cuda(0),
    };
    let device = plan.device().unwrap().without_autodiff();
    for ternary in [false, true] {
        for policy in [ReadPolicy::MetadataOnly, ReadPolicy::Enforced] {
            let model = StateReasoner::restore(&snapshot, ternary, policy, plan).unwrap();
            let (state, query) = context();
            let f = features(&[(state, query)], policy).unwrap();
            let stage = |index: usize, input: Tensor<2>| {
                let dims = input.dims();
                let host = floats(input.clone()).unwrap();
                let w = weights(&snapshot.parameters[2 * index], SHAPES[index].1, ternary);
                let b: Vec<_> = snapshot.parameters[2 * index + 1]
                    .values
                    .iter()
                    .map(|x| *x as f64)
                    .collect();
                let expected: Vec<_> = host
                    .chunks(dims[1])
                    .flat_map(|row| {
                        precision_linear(
                            &row.iter().map(|x| *x as f64).collect::<Vec<_>>(),
                            &w,
                            &b,
                            true,
                        )
                    })
                    .collect();
                let output = model.layers[index]
                    .forward(input, false, ternary, &mut 0.)
                    .unwrap();
                let actual = floats(output.clone()).unwrap();
                let error = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (*a as f64 - b).abs())
                    .fold(0., f64::max);
                println!(
                    "mode={ternary} policy={policy:?} linear{index} dtype={:?} direct_max_abs={error:.10e}",
                    output.dtype()
                );
                output
            };
            let records = Tensor::from_data(TensorData::new(f.records, [33, 85]), &device);
            let keys = stage(0, records.clone()).reshape([1, 33, 64]);
            let values = stage(1, records).reshape([1, 33, 64]);
            let mut h = stage(
                2,
                Tensor::from_data(TensorData::new(f.query, [1, 20]), &device),
            );
            for hop in 0..3 {
                let relation = stage(
                    3,
                    Tensor::from_data(TensorData::new(f.operations[hop].clone(), [1, 5]), &device),
                );
                let z = h + relation;
                let k = floats(keys.clone()).unwrap();
                let v = floats(values.clone()).unwrap();
                let q = floats(z.clone()).unwrap();
                let scores: Vec<_> = (0..33)
                    .map(|i| {
                        if f.eligible[hop][i] {
                            (0..64)
                                .map(|j| tf32(k[64 * i + j] as f64) * tf32(q[j] as f64))
                                .sum::<f64>()
                                / 8.
                        } else {
                            f64::NEG_INFINITY
                        }
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let denom = scores.iter().map(|x| (x - max).exp()).sum::<f64>();
                let p: Vec<_> = scores.iter().map(|x| (x - max).exp() / denom).collect();
                let expected: Vec<_> = (0..64)
                    .map(|j| {
                        (0..33)
                            .map(|i| tf32(p[i]) * tf32(v[64 * i + j] as f64))
                            .sum::<f64>()
                    })
                    .collect();
                let read = attention_read(
                    keys.clone(),
                    values.clone(),
                    z.clone(),
                    Tensor::from_data(TensorData::new(f.eligible[hop].clone(), [1, 33]), &device),
                );
                let error = floats(read.clone())
                    .unwrap()
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (*a as f64 - b).abs())
                    .fold(0., f64::max);
                println!(
                    "mode={ternary} policy={policy:?} attention hop{hop} direct_max_abs={error:.10e}"
                );
                h = activation::relu(stage(4, z) + read);
            }
            let output = stage(5, h);
            let expected = scalar(
                &snapshot,
                &context().0,
                &context().1,
                policy,
                ternary,
                false,
            )
            .0;
            let error = floats(output)
                .unwrap()
                .iter()
                .zip(&expected)
                .map(|(a, b)| (*a as f64 - b).abs())
                .fold(0., f64::max);
            println!("mode={ternary} policy={policy:?} composed_max_abs={error:.10e}");
        }
    }
}
