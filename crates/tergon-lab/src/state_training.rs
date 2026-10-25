//! Frozen typed-state training, checkpoint serialization and per-world evidence.
use crate::{
    state_data::{self, Example, Rng},
    training,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{error::Error, fs, io::Write, path::Path, time::Instant};
use tergon_engine::{
    ComputeDevice, ParameterSnapshot, PlacementPlan,
    state::ReadPolicy,
    state_reasoner::{ReasonerSnapshot, SHAPES, StateReasoner},
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
const RECIPE: &str = include_str!("../../../experiments/typed-state-v1.json");
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedParameter {
    values: Vec<f32>,
    moment_1: Vec<f32>,
    moment_2: Vec<f32>,
    time: usize,
}
fn saved(s: ReasonerSnapshot) -> Vec<SavedParameter> {
    s.parameters
        .into_iter()
        .map(|p| SavedParameter {
            values: p.values,
            moment_1: p.moment_1,
            moment_2: p.moment_2,
            time: p.time,
        })
        .collect()
}
fn snapshot(p: &[SavedParameter]) -> ReasonerSnapshot {
    ReasonerSnapshot {
        parameters: p
            .iter()
            .map(|p| ParameterSnapshot {
                values: p.values.clone(),
                moment_1: p.moment_1.clone(),
                moment_2: p.moment_2.clone(),
                time: p.time,
            })
            .collect(),
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    schema: u32,
    source_revision: String,
    config_sha256: String,
    dataset_sha256: Vec<String>,
    seed: u64,
    selection: Option<String>,
    mode: String,
    policy: String,
    epoch: usize,
    next_batch: usize,
    step: usize,
    shuffle_state: u64,
    order: Vec<usize>,
    parameters: Vec<SavedParameter>,
    elapsed_seconds: f64,
    epoch_loss: f64,
    epoch_examples: usize,
    curves: Vec<Value>,
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}
fn write_json(path: &Path, value: &impl Serialize) -> Result<String> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    let hash = crate::hash(&bytes);
    write_new(path, &bytes)?;
    write_new(
        &path.with_extension("sha256"),
        format!("{hash}\n").as_bytes(),
    )?;
    Ok(hash)
}
fn load(path: &Path) -> Result<Checkpoint> {
    if fs::metadata(path)?.len() > 8 * 1024 * 1024 {
        return Err("state checkpoint exceeds bounded format".into());
    }
    let bytes = fs::read(path)?;
    if fs::read_to_string(path.with_extension("sha256"))?.trim() != crate::hash(&bytes) {
        return Err("checkpoint checksum mismatch".into());
    }
    Ok(serde_json::from_slice(&bytes)?)
}
fn initial(seed: u64) -> ReasonerSnapshot {
    let mut rng = Rng(seed);
    ReasonerSnapshot {
        parameters: SHAPES
            .iter()
            .flat_map(|(out, input)| [(*out * *input, *input), (*out, 0)])
            .map(|(n, fan)| ParameterSnapshot {
                values: (0..n)
                    .map(|_| {
                        if fan == 0 {
                            0.
                        } else {
                            let u = (rng.next() >> 11) as f64 / (1_u64 << 53) as f64;
                            ((2. * u - 1.) * (6. / fan as f64).sqrt()) as f32
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
fn validate(cp: &Checkpoint, source: &str, config: &str, data: &[String]) -> Result<()> {
    if cp
        .selection
        .as_ref()
        .is_some_and(|s| *s != format!("{}:{}:{}", cp.seed, cp.policy, cp.mode))
        || cp.schema != 1
        || cp.source_revision != source
        || cp.config_sha256 != config
        || cp.dataset_sha256 != data
        || ![17, 29, 43].contains(&cp.seed)
        || !["float", "ternary"].contains(&cp.mode.as_str())
        || !["metadata-only", "enforced"].contains(&cp.policy.as_str())
        || cp.epoch > 20
        || cp.next_batch > 32768
        || !cp.next_batch.is_multiple_of(128)
        || cp.step != cp.epoch * 256 + cp.next_batch / 128
        || cp.elapsed_seconds < 0.
        || !cp.elapsed_seconds.is_finite()
        || !cp.epoch_loss.is_finite()
        || cp.epoch_examples != cp.next_batch
        || cp.parameters.len() != 12
        || cp.parameters.iter().any(|p| p.time != cp.step)
    {
        return Err("checkpoint identity, position or counter mismatch".into());
    }
    let mut rng = Rng(cp.seed ^ 0x53485546464c45);
    let mut expected: Vec<_> = (0..32768).collect();
    for _ in 0..=cp.epoch {
        rng.shuffle(&mut expected);
    }
    if cp.order != expected || cp.shuffle_state != rng.0 {
        return Err("checkpoint shuffle state/order mismatch".into());
    }
    Ok(())
}
fn policy(name: &str) -> ReadPolicy {
    if name == "enforced" {
        ReadPolicy::Enforced
    } else {
        ReadPolicy::MetadataOnly
    }
}
fn probabilities(logits: &[f32]) -> Result<Vec<f64>> {
    if logits.len() != 18 || logits.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite or malformed logits".into());
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let p: Vec<_> = logits.iter().map(|v| (*v as f64 - max).exp()).collect();
    let sum = p.iter().sum::<f64>();
    Ok(p.iter().map(|v| v / sum).collect())
}
fn prediction(p: &[f64]) -> usize {
    p.iter()
        .enumerate()
        .max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a)))
        .unwrap()
        .0
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Prediction {
    pub group: String,
    pub world: u64,
    pub branch: usize,
    pub hops: usize,
    pub label: usize,
    pub predicted: usize,
    pub probabilities: Vec<f64>,
    pub brier: f64,
    pub confidently_unsupported: bool,
    pub renamed_consistent: bool,
    pub reordered_consistent: bool,
}
fn evaluate(
    model: &StateReasoner,
    examples: &[Example],
    metamorphic: bool,
) -> Result<(Value, Vec<Prediction>)> {
    let mut rows = Vec::new();
    let mut rng = Rng(0x52454e414d45);
    for chunk in examples.chunks(128) {
        let contexts: Vec<_> = chunk.iter().map(Example::materialize).collect();
        let logits = model.logits(&contexts)?;
        let mut perm = std::array::from_fn(|i| i);
        rng.shuffle(&mut perm);
        let renamed = if metamorphic {
            Some(
                model.logits(
                    &chunk
                        .iter()
                        .map(|e| e.renamed(&perm).materialize())
                        .collect::<Vec<_>>(),
                )?,
            )
        } else {
            None
        };
        let reordered = if metamorphic {
            let mut reordered = contexts.clone();
            for (s, _) in &mut reordered {
                rng.shuffle(&mut s.records);
            }
            Some(model.logits(&reordered)?)
        } else {
            None
        };
        for (i, e) in chunk.iter().enumerate() {
            let p = probabilities(&logits[i * 18..(i + 1) * 18])?;
            let predicted = prediction(&p);
            let brier = p
                .iter()
                .enumerate()
                .map(|(j, p)| (p - f64::from(j == e.label)).powi(2))
                .sum();
            let renamed_consistent = if let Some(logits) = &renamed {
                prediction(&probabilities(&logits[i * 18..(i + 1) * 18])?)
                    == if predicted < 16 {
                        perm[predicted]
                    } else {
                        predicted
                    }
            } else {
                true
            };
            let reordered_consistent = if let Some(logits) = &reordered {
                prediction(&probabilities(&logits[i * 18..(i + 1) * 18])?) == predicted
            } else {
                true
            };
            rows.push(Prediction {
                group: e.group.clone(),
                world: e.world,
                branch: e.branch,
                hops: e.relations.len(),
                label: e.label,
                predicted,
                brier,
                confidently_unsupported: predicted < 16
                    && predicted != e.label
                    && p[predicted] >= 0.9,
                probabilities: p,
                renamed_consistent,
                reordered_consistent,
            });
        }
    }
    Ok((metrics(&rows), rows))
}
fn metrics(rows: &[Prediction]) -> Value {
    let summarize = |subset: Vec<&Prediction>| {
        let n = subset.len();
        json!({"examples":n,"accuracy":subset.iter().filter(|p|p.label==p.predicted).count()as f64/n as f64,"brier":subset.iter().map(|p|p.brier).sum::<f64>()/n as f64,"confidently_unsupported":subset.iter().filter(|p|p.confidently_unsupported).count(),"renaming_consistency":subset.iter().filter(|p|p.renamed_consistent).count()as f64/n as f64,"order_consistency":subset.iter().filter(|p|p.reordered_consistent).count()as f64/n as f64})
    };
    let mut by_hop = serde_json::Map::new();
    for hop in 1..=3 {
        let subset = rows.iter().filter(|p| p.hops == hop).collect::<Vec<_>>();
        if !subset.is_empty() {
            by_hop.insert(hop.to_string(), summarize(subset));
        }
    }
    let mut by_outcome = serde_json::Map::new();
    for (name, class) in [("known", 0), ("unknown", 16), ("conflict", 17)] {
        let subset = rows
            .iter()
            .filter(|p| {
                if class == 0 {
                    p.label < 16
                } else {
                    p.label == class
                }
            })
            .collect::<Vec<_>>();
        if !subset.is_empty() {
            by_outcome.insert(name.into(), summarize(subset));
        }
    }
    let mut counterfactual_pairs = 0;
    let mut changed = 0;
    let mut correct = 0;
    for world in rows.as_chunks::<4>().0 {
        let a = world.iter().find(|p| p.branch == 0).unwrap();
        let b = world.iter().find(|p| p.branch == 1).unwrap();
        counterfactual_pairs += 1;
        changed += usize::from(a.predicted != b.predicted);
        correct += usize::from(a.predicted == a.label && b.predicted == b.label);
    }
    json!({"all":summarize(rows.iter().collect()),"by_hop":by_hop,"by_outcome":by_outcome,"known_counterfactual_pairs":counterfactual_pairs,"prediction_changed_pairs":changed,"both_changed_answers_correct":correct})
}
/// Run frozen paired conditions; an optional condition selection never changes the recipe.
pub(crate) fn train(config_path: &Path, directory: &Path, resume: Option<&Path>) -> Result<()> {
    let bytes = fs::read(config_path)?;
    let config: Value = serde_json::from_slice(&bytes)?;
    if config != serde_json::from_str::<Value>(RECIPE)? {
        return Err("typed-state config differs from frozen recipe".into());
    }
    let source = crate::command("git", &["rev-parse", "HEAD"])?;
    let dirty = !crate::command("git", &["status", "--porcelain"])?.is_empty();
    if dirty {
        return Err("formal state training requires a committed clean source tree".into());
    }
    let plan = PlacementPlan {
        compute: ComputeDevice::Cuda(0),
    };
    plan.device()?;
    let config_hash = crate::hash(&bytes);
    let selected = std::env::var("TERGON_STATE_RUN").ok();
    let resumed = resume.map(load).transpose()?;
    if resumed
        .as_ref()
        .is_some_and(|cp| cp.elapsed_seconds >= 600.)
    {
        return Err("checkpoint has exhausted its original600-second budget".into());
    }
    if let Some(cp) = &resumed
        && selected != cp.selection
    {
        return Err("resume selection conflicts with checkpoint".into());
    }
    let conditions: Vec<_> = [17_u64, 29, 43]
        .into_iter()
        .flat_map(|seed| {
            ["metadata-only", "enforced"]
                .into_iter()
                .flat_map(move |policy| {
                    ["float", "ternary"]
                        .into_iter()
                        .map(move |mode| (seed, policy, mode))
                })
        })
        .filter(|(seed, policy, mode)| {
            selected
                .as_ref()
                .is_none_or(|s| *s == format!("{seed}:{policy}:{mode}"))
                && resumed
                    .as_ref()
                    .is_none_or(|cp| cp.seed == *seed && cp.policy == *policy && cp.mode == *mode)
        })
        .collect();
    if conditions.is_empty() {
        return Err("invalid condition selection/resume combination".into());
    }
    fs::create_dir(directory)?;
    write_json(&directory.join("config.json"), &config)?;
    state_data::audit(&directory.join("data"))?;
    let datasets: Vec<Vec<Example>> = (0..7)
        .map(|group| {
            state_data::dataset(
                group,
                if group == 0 {
                    8192
                } else if group == 1 {
                    1024
                } else {
                    512
                },
            )
        })
        .collect();
    let hashes: Vec<_> = datasets
        .iter()
        .map(|data| serde_json::to_vec(data).map(|b| crate::hash(&b)))
        .collect::<std::result::Result<_, _>>()?;
    let data_contexts: Vec<_> = datasets[0].iter().map(Example::materialize).collect();
    let mut receipt = json!({"schema":1,"experiment":"typed-state-v1","source_revision":source,"source_dirty":dirty,"config_sha256":config_hash,"dataset_sha256":hashes,"selection":selected,"selection_parsed":conditions.iter().map(|(seed,policy,mode)|json!({"seed":seed,"policy":policy,"mode":mode})).collect::<Vec<_>>(),"device":"cuda:0","hardware":crate::command("nvidia-smi",&["--query-gpu=name,uuid,driver_version,memory.total","--format=csv,noheader"] )?,"rustc":crate::command("rustc",&["--version"])?,"runs":[],"limitations":["host canonical quantization and decoded GPU uploads; no packed training kernel","sampled GPU process memory is not guaranteed allocator peak","no energy sensor; no joule claim"]});
    for (seed, policy_name, mode) in conditions {
        let run_dir = directory.join(format!("{seed}-{policy_name}-{mode}"));
        fs::create_dir(&run_dir)?;
        let started = Instant::now();
        let mut cp = if let Some(cp) = &resumed {
            validate(cp, &source, &config_hash, &hashes)?;
            cp.clone()
        } else {
            let mut rng = Rng(seed ^ 0x53485546464c45);
            let mut order = (0..32768).collect::<Vec<_>>();
            rng.shuffle(&mut order);
            Checkpoint {
                schema: 1,
                source_revision: source.clone(),
                config_sha256: config_hash.clone(),
                dataset_sha256: hashes.clone(),
                seed,
                selection: selected.clone(),
                mode: mode.into(),
                policy: policy_name.into(),
                epoch: 0,
                next_batch: 0,
                step: 0,
                shuffle_state: rng.0,
                order,
                parameters: saved(initial(seed)),
                elapsed_seconds: 0.,
                epoch_loss: 0.,
                epoch_examples: 0,
                curves: vec![],
            }
        };
        let elapsed_before = cp.elapsed_seconds;
        let mut model = StateReasoner::restore(
            &snapshot(&cp.parameters),
            mode == "ternary",
            policy(policy_name),
            plan,
        )?;
        let mut host_wait = 0.;
        let mut gpu_peak = None;
        let mut failure = None;
        let execution = (|| -> Result<()> {
            while cp.epoch < 20 {
                while cp.next_batch < 32768 {
                    if elapsed_before + started.elapsed().as_secs_f64() >= 600. {
                        return Err("600-second budget exhausted".into());
                    }
                    let indices = &cp.order[cp.next_batch..cp.next_batch + 128];
                    let batch: Vec<_> = indices.iter().map(|i| data_contexts[*i].clone()).collect();
                    let labels: Vec<_> = indices.iter().map(|i| datasets[0][*i].label).collect();
                    let loss = model.step(&batch, &labels)?;
                    cp.epoch_loss += loss as f64 * 128.;
                    cp.epoch_examples += 128;
                    cp.next_batch += 128;
                    cp.step += 1;
                    host_wait += model.last_host_read_wait_seconds();
                    if cp.step.is_multiple_of(20)
                        && let Some(memory) = training::gpu_memory_mib()
                    {
                        gpu_peak = Some(gpu_peak.map_or(memory, |p: u64| p.max(memory)));
                    }
                    if cp.step == 200 {
                        cp.parameters = saved(model.snapshot()?);
                        cp.elapsed_seconds = elapsed_before + started.elapsed().as_secs_f64();
                        write_json(&run_dir.join("step-200.json"), &cp)?;
                    }
                }
                let (validation, _) = evaluate(&model, &datasets[1], false)?;
                cp.curves.push(json!({"epoch":cp.epoch+1,"step":cp.step,"train_cross_entropy":cp.epoch_loss/cp.epoch_examples as f64,"validation":validation,"elapsed_seconds":elapsed_before+started.elapsed().as_secs_f64()}));
                cp.epoch += 1;
                cp.next_batch = 0;
                cp.epoch_loss = 0.;
                cp.epoch_examples = 0;
                let mut rng = Rng(cp.shuffle_state);
                rng.shuffle(&mut cp.order);
                cp.shuffle_state = rng.0;
            }
            Ok(())
        })();
        if let Err(e) = execution {
            failure = Some(e.to_string());
        }
        let mut groups = serde_json::Map::new();
        if failure.is_none() {
            for (group, data) in datasets.iter().enumerate().skip(2) {
                if elapsed_before + started.elapsed().as_secs_f64() >= 600. {
                    failure = Some(
                        "600-second budget exhausted before final evaluation completed".into(),
                    );
                    break;
                }
                match evaluate(&model, data, true) {
                    Ok((summary, predictions)) => {
                        write_json(
                            &run_dir
                                .join(format!("{}-predictions.json", state_data::GROUPS[group])),
                            &predictions,
                        )?;
                        groups.insert(state_data::GROUPS[group].into(), summary);
                    }
                    Err(e) => {
                        failure = Some(e.to_string());
                        break;
                    }
                }
            }
        }
        cp.parameters = saved(model.snapshot()?);
        cp.elapsed_seconds = elapsed_before + started.elapsed().as_secs_f64();
        let checkpoint_hash = write_json(&run_dir.join("final.json"), &cp)?;
        if cp.elapsed_seconds > 600. && failure.is_none() {
            failure =
                Some("600-second budget exceeded during evaluation/checkpoint finalization".into());
        }
        let matrix_weights: usize = SHAPES.iter().map(|(a, b)| a * b).sum();
        let biases: usize = SHAPES.iter().map(|(a, _)| a).sum();
        let result = json!({"seed":seed,"policy":policy_name,"mode":mode,"completed":failure.is_none() && cp.epoch==20 && groups.len()==5,"failure":failure,"updates":cp.step,"epochs":cp.epoch,"elapsed_seconds":cp.elapsed_seconds,"host_read_wait_seconds":host_wait,"checkpoint_sha256":checkpoint_hash,"groups":groups,"curves":cp.curves,"peak_rss_kib":training::peak_rss_kib(),"sampled_gpu_process_peak_mib":gpu_peak,"gpu_sample_interval_steps":20,"model_memory":{"latent_matrix_bytes":matrix_weights*4,"floating_bias_bytes":biases*4,"adam_moment_bytes":(matrix_weights+biases)*8,"gradient_payload_bytes":(matrix_weights+biases)*4,"packed_matrix_payload_bytes":SHAPES.iter().map(|(a,b)|(a*b).div_ceil(4)+a*4).sum::<usize>(),"excludes":"activations, decoded weights, tensors, allocator and runtime; actual sampled process memory reported separately"},"transfer_accounting":{"f32_input_features_uploaded_bytes_per_forward_batch128":128*(33*85+20+3*5+3)*4,"boolean_mask_values_uploaded_per_forward_batch128":128*33*3,"ternary_activation_host_read_bytes_per_forward_batch128":if mode=="ternary"{128*(2*33*85+20+3*5+3*64+64)*4}else{0},"ternary_matrix_host_read_and_decoded_upload_bytes_each_per_forward":if mode=="ternary"{(2*64*85+64*20+3*64*5+3*64*64+18*64)*4}else{0},"limitation":"explicit payload accounting, not PCIe instrumentation; excludes finite reductions, logits, snapshots and allocator transfers"}});
        write_json(&run_dir.join("run.json"), &result)?;
        receipt["runs"].as_array_mut().unwrap().push(result);
        write_json(
            &directory.join(format!(
                "receipt-after-{}.json",
                receipt["runs"].as_array().unwrap().len()
            )),
            &receipt,
        )?;
    }
    write_json(&directory.join("receipt.json"), &receipt)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn replay(compute: ComputeDevice) {
        let gpu = matches!(compute, ComputeDevice::Cuda(_));
        let plan = PlacementPlan { compute };
        let source = "fixture";
        let data = vec!["fixture-data".into(); 7];
        let dataset = state_data::dataset(0, 2);
        let directory =
            std::env::temp_dir().join(format!("tergon-state-replay-{}-{gpu}", std::process::id()));
        fs::create_dir(&directory).unwrap();
        for mode in ["float", "ternary"] {
            for policy_name in ["metadata-only", "enforced"] {
                let mut model = StateReasoner::restore(
                    &initial(17),
                    mode == "ternary",
                    policy(policy_name),
                    plan,
                )
                .unwrap();
                let mut rng = Rng(17 ^ 0x53485546464c45);
                let mut order = (0..32768).collect::<Vec<_>>();
                rng.shuffle(&mut order);
                let batch_at = |step: usize| {
                    let examples: Vec<_> = order[step * 128..step * 128 + 4]
                        .iter()
                        .map(|i| &dataset[*i % dataset.len()])
                        .collect();
                    (
                        examples.iter().map(|e| e.materialize()).collect::<Vec<_>>(),
                        examples.iter().map(|e| e.label).collect::<Vec<_>>(),
                    )
                };
                for step in 0..200 {
                    let (batch, labels) = batch_at(step);
                    model.step(&batch, &labels).unwrap();
                }
                let cp = Checkpoint {
                    schema: 1,
                    source_revision: source.into(),
                    config_sha256: "config".into(),
                    dataset_sha256: data.clone(),
                    seed: 17,
                    selection: None,
                    mode: mode.into(),
                    policy: policy_name.into(),
                    epoch: 0,
                    next_batch: 25600,
                    step: 200,
                    shuffle_state: rng.0,
                    order: order.clone(),
                    parameters: saved(model.snapshot().unwrap()),
                    elapsed_seconds: 1.,
                    epoch_loss: 1.,
                    epoch_examples: 25600,
                    curves: vec![],
                };
                validate(&cp, source, "config", &data).unwrap();
                let path = directory.join(format!("{mode}-{policy_name}.json"));
                write_json(&path, &cp).unwrap();
                let loaded = load(&path).unwrap();
                validate(&loaded, source, "config", &data).unwrap();
                let mut restored = StateReasoner::restore(
                    &snapshot(&loaded.parameters),
                    mode == "ternary",
                    policy(policy_name),
                    plan,
                )
                .unwrap();
                for step in 200..250 {
                    let (batch, labels) = batch_at(step);
                    let a = model.step(&batch, &labels).unwrap();
                    let b = restored.step(&batch, &labels).unwrap();
                    if gpu {
                        assert!((a - b).abs() <= 1e-5 + 1e-4 * a.abs());
                    } else {
                        assert_eq!(a.to_bits(), b.to_bits());
                    }
                }
                let a = model.snapshot().unwrap();
                let b = restored.snapshot().unwrap();
                for (a, b) in a.parameters.iter().zip(&b.parameters) {
                    assert_eq!(a.time, b.time);
                    for (a, b) in a
                        .values
                        .iter()
                        .chain(&a.moment_1)
                        .chain(&a.moment_2)
                        .zip(b.values.iter().chain(&b.moment_1).chain(&b.moment_2))
                    {
                        if gpu {
                            assert!((*a - *b).abs() <= 1e-5 + 1e-4 * a.abs());
                        } else {
                            assert_eq!(a.to_bits(), b.to_bits());
                        }
                    }
                }
                let (batch, _) = batch_at(249);
                for (a, b) in model
                    .logits(&batch)
                    .unwrap()
                    .iter()
                    .zip(restored.logits(&batch).unwrap())
                {
                    if gpu {
                        assert!((*a - b).abs() <= 1e-5 + 1e-4 * a.abs());
                    } else {
                        assert_eq!(a.to_bits(), b.to_bits());
                    }
                }
                let mut bad = loaded.clone();
                bad.shuffle_state += 1;
                assert!(validate(&bad, source, "config", &data).is_err());
                let mut bad = loaded.clone();
                bad.order[0] = bad.order[1];
                assert!(validate(&bad, source, "config", &data).is_err());
                let mut bad = loaded.clone();
                bad.selection = Some("29:enforced:float".into());
                assert!(validate(&bad, source, "config", &data).is_err());
                fs::write(&path, b"corrupt").unwrap();
                assert!(load(&path).is_err());
            }
        }
        fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn serialized_cpu_replay_step200_next50() {
        replay(ComputeDevice::Cpu);
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn serialized_cuda_replay_step200_next50() {
        replay(ComputeDevice::Cuda(0));
    }
    #[test]
    fn brier_definition_and_tie_rule() {
        assert_eq!(prediction(&[0.5, 0.5]), 0);
        let p = probabilities(&[0.; 18]).unwrap();
        let brier = p
            .iter()
            .enumerate()
            .map(|(i, p)| (p - f64::from(i == 3)).powi(2))
            .sum::<f64>();
        assert!((brier - 17. / 18.).abs() < 1e-12);
    }
}
