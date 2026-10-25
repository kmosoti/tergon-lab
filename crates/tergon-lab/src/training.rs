//! Frozen MNIST experiment, dataset/checkpoint I/O and receipts.
use super::hash;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{error::Error, fs, io::Write, path::Path, process::Command, time::Instant};
use tergon_engine::{
    ClassifierSnapshot, ComputeDevice, ParameterSnapshot, PlacementPlan, TinyClassifier,
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
const RECIPE: &str = include_str!("../../../experiments/mnist-adamw-v1.json");

/// SplitMix64 by Sebastiano Vigna (public domain), adapted to explicit saved state.
/// <https://prng.di.unimi.it/splitmix.c>
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn bounded(&mut self, n: usize) -> usize {
        let n = n as u64;
        let threshold = n.wrapping_neg() % n;
        loop {
            let r = self.next();
            if r >= threshold {
                return (r % n) as usize;
            }
        }
    }
    fn shuffle(&mut self, order: &mut [usize]) {
        for i in (1..order.len()).rev() {
            order.swap(i, self.bounded(i + 1));
        }
    }
}
fn initial(seed: u64) -> ClassifierSnapshot {
    let mut rng = Rng(seed);
    ClassifierSnapshot {
        parameters: [(128 * 784, 784), (128, 0), (10 * 128, 128), (10, 0)].map(|(len, fan)| {
            let values = if fan == 0 {
                vec![0.; len]
            } else {
                let bound = (6.0_f64 / fan as f64).sqrt();
                (0..len)
                    .map(|_| {
                        ((2. * ((rng.next() >> 11) as f64 / ((1_u64 << 53) as f64)) - 1.) * bound)
                            as f32
                    })
                    .collect()
            };
            ParameterSnapshot {
                values,
                moment_1: vec![],
                moment_2: vec![],
                time: 0,
            }
        }),
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedParameter {
    values: Vec<f32>,
    moment_1: Vec<f32>,
    moment_2: Vec<f32>,
    time: usize,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    schema: u32,
    source_revision: String,
    config_sha256: String,
    dataset: Value,
    run: usize,
    seed: u64,
    mode: String,
    epoch: usize,
    next_batch: usize,
    step: usize,
    shuffle_state: u64,
    order: Vec<usize>,
    parameters: [SavedParameter; 4],
    epoch_loss: f64,
    epoch_examples: usize,
    curves: Vec<Value>,
}
fn saved(parameters: ClassifierSnapshot) -> [SavedParameter; 4] {
    parameters.parameters.map(|p| SavedParameter {
        values: p.values,
        moment_1: p.moment_1,
        moment_2: p.moment_2,
        time: p.time,
    })
}
fn snapshot(parameters: &[SavedParameter; 4]) -> ClassifierSnapshot {
    ClassifierSnapshot {
        parameters: std::array::from_fn(|i| {
            let p = &parameters[i];
            ParameterSnapshot {
                values: p.values.clone(),
                moment_1: p.moment_1.clone(),
                moment_2: p.moment_2.clone(),
                time: p.time,
            }
        }),
    }
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    write_new(path, &bytes)?;
    write_new(
        &path.with_extension("sha256"),
        format!("{}\n", hash(&bytes)).as_bytes(),
    )
}
fn read_checkpoint(path: &Path) -> Result<Checkpoint> {
    if fs::metadata(path)?.len() > 32 * 1024 * 1024 {
        return Err("checkpoint exceeds bounded format size".into());
    }
    let bytes = fs::read(path)?;
    if fs::read_to_string(path.with_extension("sha256"))?.trim() != hash(&bytes) {
        return Err("checkpoint checksum mismatch".into());
    }
    Ok(serde_json::from_slice(&bytes)?)
}
fn u32_at(bytes: &[u8], i: usize) -> Result<usize> {
    Ok(u32::from_be_bytes(bytes.get(i..i + 4).ok_or("truncated IDX")?.try_into()?) as usize)
}
fn images(bytes: Vec<u8>, count: usize) -> Result<Vec<u8>> {
    if u32_at(&bytes, 0)? != 2051
        || u32_at(&bytes, 4)? != count
        || u32_at(&bytes, 8)? != 28
        || u32_at(&bytes, 12)? != 28
        || bytes.len() != 16 + count * 784
    {
        return Err("invalid image IDX".into());
    }
    Ok(bytes[16..].to_vec())
}
fn labels(bytes: Vec<u8>, count: usize) -> Result<Vec<usize>> {
    if u32_at(&bytes, 0)? != 2049
        || u32_at(&bytes, 4)? != count
        || bytes.len() != 8 + count
        || bytes[8..].iter().any(|y| *y >= 10)
    {
        return Err("invalid label IDX".into());
    }
    Ok(bytes[8..].iter().map(|y| usize::from(*y)).collect())
}
struct Dataset {
    train_x: Vec<u8>,
    train_y: Vec<usize>,
    test_x: Vec<u8>,
    test_y: Vec<usize>,
}
impl Dataset {
    fn load(dir: &Path, config: &Value) -> Result<Self> {
        let read = |name: &str| -> Result<Vec<u8>> {
            let bytes = fs::read(dir.join(name))?;
            if hash(&bytes)
                != config["dataset"][name]["sha256"]
                    .as_str()
                    .ok_or("missing dataset hash")?
            {
                return Err(format!("dataset hash mismatch: {name}").into());
            }
            Ok(bytes)
        };
        Ok(Self {
            train_x: images(read("train-images-idx3-ubyte")?, 60000)?,
            train_y: labels(read("train-labels-idx1-ubyte")?, 60000)?,
            test_x: images(read("t10k-images-idx3-ubyte")?, 10000)?,
            test_y: labels(read("t10k-labels-idx1-ubyte")?, 10000)?,
        })
    }
}
fn batch(x: &[u8], y: &[usize], indices: &[usize]) -> (Vec<f32>, Vec<usize>) {
    let mut inputs = Vec::with_capacity(indices.len() * 784);
    let mut targets = Vec::with_capacity(indices.len());
    for &i in indices {
        inputs.extend(
            x[i * 784..(i + 1) * 784]
                .iter()
                .map(|v| f32::from(*v) / 255.),
        );
        targets.push(y[i]);
    }
    (inputs, targets)
}
fn evaluate(
    model: &TinyClassifier,
    x: &[u8],
    y: &[usize],
    start: usize,
    end: usize,
) -> Result<(f64, f64)> {
    let mut correct = 0;
    let mut ce = 0.;
    for low in (start..end).step_by(128) {
        let indices: Vec<_> = (low..(low + 128).min(end)).collect();
        let (inputs, targets) = batch(x, y, &indices);
        let outputs = model.logits(&inputs)?;
        for (row, target) in outputs.as_chunks::<10>().0.iter().zip(targets) {
            if row.iter().any(|v| !v.is_finite()) {
                return Err("nonfinite evaluation logits".into());
            }
            let best = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .ok_or("empty logits")?
                .0;
            correct += usize::from(best == target);
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            ce += max
                + row
                    .iter()
                    .map(|v| (f64::from(*v) - max).exp())
                    .sum::<f64>()
                    .ln()
                - f64::from(row[target]);
        }
    }
    if !ce.is_finite() {
        return Err("nonfinite evaluation cross entropy".into());
    }
    Ok((
        correct as f64 / (end - start) as f64,
        ce / (end - start) as f64,
    ))
}
fn peak_rss_kib() -> Option<u64> {
    fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|l| {
            l.strip_prefix("VmHWM:")?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
}
fn gpu_memory_mib() -> Option<u64> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()?.lines().find_map(|l| {
        let (pid, mem) = l.split_once(',')?;
        if pid.trim().parse::<u32>().ok()? == std::process::id() {
            mem.trim().parse().ok()
        } else {
            None
        }
    })
}
fn compare(a: &ClassifierSnapshot, b: &ClassifierSnapshot) -> Result<f64> {
    let mut max = 0_f64;
    for (a, b) in a.parameters.iter().zip(&b.parameters) {
        if a.values.len() != b.values.len()
            || a.moment_1.len() != b.moment_1.len()
            || a.moment_2.len() != b.moment_2.len()
        {
            return Err("resume vector length mismatch".into());
        }
        if a.time != b.time {
            return Err("resume optimizer counter mismatch".into());
        }
        for (a, b) in a
            .values
            .iter()
            .chain(&a.moment_1)
            .chain(&a.moment_2)
            .zip(b.values.iter().chain(&b.moment_1).chain(&b.moment_2))
        {
            let error = (f64::from(*a) - f64::from(*b)).abs();
            max = max.max(error);
            if !error.is_finite() || error > 1e-5 + 1e-4 * f64::from(*a).abs() {
                return Err("GPU resume state diverged".into());
            }
        }
    }
    Ok(max)
}
fn source_revision() -> Result<String> {
    let dir = env!("CARGO_MANIFEST_DIR");
    super::command("git", &["-C", dir, "rev-parse", "HEAD"])
}
fn source_clean() -> Result<bool> {
    Ok(super::command(
        "git",
        &["-C", env!("CARGO_MANIFEST_DIR"), "status", "--porcelain"],
    )?
    .is_empty())
}

fn validate_checkpoint(
    cp: &Checkpoint,
    revision: &str,
    config_hash: &str,
    config: &Value,
) -> Result<()> {
    if cp.schema != 1
        || cp.source_revision != revision
        || cp.config_sha256 != config_hash
        || cp.dataset != config["dataset"]
        || cp.run >= 6
        || cp.seed != [17, 29, 43][cp.run / 2]
        || cp.mode != ["ternary", "float"][cp.run % 2]
        || cp.epoch > 10
        || cp.next_batch >= 55000
        || !cp.next_batch.is_multiple_of(128)
        || (cp.epoch == 10 && cp.next_batch != 0)
        || cp.order.len() != 55000
        || cp.step != cp.epoch * 430 + cp.next_batch / 128
        || cp.parameters.iter().any(|p| p.time != cp.step)
        || cp.epoch_examples != cp.next_batch
        || !cp.epoch_loss.is_finite()
        || cp.epoch_loss < 0.
        || (cp.next_batch == 0 && cp.epoch_loss != 0.)
        || cp.curves.len() != cp.epoch
    {
        return Err("incompatible checkpoint metadata/counters".into());
    }
    let mut rng = Rng(cp.seed);
    let mut expected: Vec<_> = (0..55000).collect();
    for _ in 0..cp.epoch + usize::from(cp.next_batch > 0) {
        expected = (0..55000).collect();
        rng.shuffle(&mut expected);
    }
    if cp.order != expected || cp.shuffle_state != rng.0 {
        return Err("checkpoint shuffle state/permutation mismatch".into());
    }
    Ok(())
}

pub(super) fn train(
    config_path: &Path,
    data_dir: &Path,
    out: &Path,
    resume: Option<&Path>,
) -> Result<()> {
    let config_bytes = fs::read(config_path)?;
    let config: Value = serde_json::from_slice(&config_bytes)?;
    if config != serde_json::from_str::<Value>(RECIPE)? {
        return Err("configuration differs from frozen mnist-adamw-v1 recipe".into());
    }
    let revision = source_revision()?;
    let config_hash = hash(&config_bytes);
    let dataset = Dataset::load(data_dir, &config)?;
    let checkpoint = resume.map(read_checkpoint).transpose()?;
    if let Some(cp) = &checkpoint {
        validate_checkpoint(cp, &revision, &config_hash, &config)?;
    }
    if !source_clean()? {
        return Err("commit experiment source before measured runs".into());
    }
    fs::create_dir(out)?;
    write_new(&out.join("config.json"), &config_bytes)?;
    let manifest = json!({"schema":1,"source_revision":revision,"source_dirty":false,"config_sha256":config_hash,
        "dataset":config["dataset"],"hardware":super::command("nvidia-smi",&["--query-gpu=name,uuid,driver_version,memory.total","--format=csv,noheader"] )?,
        "rustc":super::command("rustc",&["--version"] )?,"resume":resume.map(|p|p.display().to_string()),
        "runtime_boundary":"Per-run wallclock includes initialization, training, evaluation, checkpoints, resume comparison and process-memory sampling; excludes compilation and initial IDX loading",
        "memory_boundary":"VmHWM process-wide peak includes IDX, allocator, checkpoints, host tensors and runtime; GPU process memory sampled every20steps and evaluation, may miss transient peaks; no total-memory savings assertion"});
    write_json(&out.join("manifest.json"), &manifest)?;
    let first = checkpoint.as_ref().map_or(0, |c| c.run);
    let mut any_failed = false;
    for run in first..6 {
        let started = Instant::now();
        let seed = [17, 29, 43][run / 2];
        let mode = ["ternary", "float"][run % 2];
        let ternary = mode == "ternary";
        let run_dir = out.join(format!("{mode}-{seed}"));
        fs::create_dir(&run_dir)?;
        let plan = PlacementPlan {
            compute: ComputeDevice::Cuda(0),
        };
        let cp = checkpoint.as_ref().filter(|c| c.run == run);
        let mut model = TinyClassifier::restore(
            &cp.map_or_else(|| initial(seed), |c| snapshot(&c.parameters)),
            ternary,
            plan,
        )?;
        let mut rng = Rng(cp.map_or(seed, |c| c.shuffle_state));
        let mut order = cp.map_or_else(|| (0..55000).collect(), |c| c.order.clone());
        let mut epoch = cp.map_or(0, |c| c.epoch);
        let mut next = cp.map_or(0, |c| c.next_batch);
        let mut step = cp.map_or(0, |c| c.step);
        let mut resume_model: Option<(TinyClassifier, Vec<usize>, usize, usize)> = None;
        let mut resume_error = 0_f64;
        let mut resume_checked = 0;
        let mut curves = cp.map_or_else(Vec::new, |c| c.curves.clone());
        let mut gpu_peak = None;
        let mut total_step_seconds = 0.;
        let mut host_read_wait_seconds = 0.;
        let mut train_examples = 0_usize;
        let result = (|| -> Result<Value> {
            while epoch < 10 {
                if next == 0 {
                    order = (0..55000).collect();
                    rng.shuffle(&mut order);
                }
                let mut total_loss = cp.filter(|c| c.epoch == epoch).map_or(0., |c| c.epoch_loss);
                let mut examples = cp
                    .filter(|c| c.epoch == epoch)
                    .map_or(0, |c| c.epoch_examples);
                while next < 55000 {
                    if started.elapsed().as_secs_f64() > 1800. {
                        return Err("fixed 1800s run budget exceeded".into());
                    }
                    let end = (next + 128).min(55000);
                    let (inputs, targets) =
                        batch(&dataset.train_x, &dataset.train_y, &order[next..end]);
                    let tick = Instant::now();
                    let loss = model.step(&inputs, &targets)?;
                    total_step_seconds += tick.elapsed().as_secs_f64();
                    host_read_wait_seconds += model.last_host_read_wait_seconds();
                    if let Some((restarted, replay_order, replay_next, replay_step)) =
                        &mut resume_model
                    {
                        let replay_end = (*replay_next + 128).min(55000);
                        let (replay_inputs, replay_targets) = batch(
                            &dataset.train_x,
                            &dataset.train_y,
                            &replay_order[*replay_next..replay_end],
                        );
                        if replay_inputs != inputs
                            || replay_targets != targets
                            || *replay_step != step
                        {
                            return Err("resumed data/cursor mismatch".into());
                        }
                        let _: f32 =
                            TinyClassifier::step(restarted, &replay_inputs, &replay_targets)?;
                        *replay_next = replay_end;
                        *replay_step += 1;
                        resume_error =
                            resume_error.max(compare(&model.snapshot()?, &restarted.snapshot()?)?);
                        let a = model.logits(&inputs)?;
                        let b = restarted.logits(&inputs)?;
                        for (a, b) in a.iter().zip(&b) {
                            if !a.is_finite()
                                || !b.is_finite()
                                || (a - b).abs() > 1e-5 + 1e-4 * a.abs()
                            {
                                return Err("GPU resumed logits diverged".into());
                            }
                        }
                        resume_checked += 1;
                        if resume_checked == 50 {
                            resume_model = None;
                        }
                    }
                    total_loss += f64::from(loss) * targets.len() as f64;
                    examples += targets.len();
                    train_examples += targets.len();
                    next = end;
                    step += 1;
                    if step.is_multiple_of(20)
                        && let Some(memory) = gpu_memory_mib()
                    {
                        gpu_peak = Some(gpu_peak.map_or(memory, |p: u64| p.max(memory)));
                    }
                    if step == 200 {
                        let checkpoint = Checkpoint {
                            schema: 1,
                            source_revision: revision.clone(),
                            config_sha256: config_hash.clone(),
                            dataset: config["dataset"].clone(),
                            run,
                            seed,
                            mode: mode.into(),
                            epoch,
                            next_batch: next,
                            step,
                            shuffle_state: rng.0,
                            order: order.clone(),
                            parameters: saved(model.snapshot()?),
                            epoch_loss: total_loss,
                            epoch_examples: examples,
                            curves: curves.clone(),
                        };
                        let path = run_dir.join(format!("step-{step}.json"));
                        write_json(&path, &checkpoint)?;
                        if step == 200 {
                            // Exercise the actual JSON/checksum reader and host-state restore.
                            let restored = read_checkpoint(&path)?;
                            resume_model = Some((
                                TinyClassifier::restore(
                                    &snapshot(&restored.parameters),
                                    ternary,
                                    plan,
                                )?,
                                restored.order,
                                restored.next_batch,
                                restored.step,
                            ));
                        }
                    }
                }
                let (accuracy, ce) =
                    evaluate(&model, &dataset.train_x, &dataset.train_y, 55000, 60000)?;
                curves.push(json!({"epoch":epoch+1,"step":step,"train_loss":total_loss/examples as f64,"validation_accuracy":accuracy,"validation_loss":ce}));
                println!(
                    "{mode}-{seed} epoch{} validation{accuracy:.4} elapsed{:.1}s",
                    epoch + 1,
                    started.elapsed().as_secs_f64()
                );
                epoch += 1;
                next = 0;
                let checkpoint = Checkpoint {
                    schema: 1,
                    source_revision: revision.clone(),
                    config_sha256: config_hash.clone(),
                    dataset: config["dataset"].clone(),
                    run,
                    seed,
                    mode: mode.into(),
                    epoch,
                    next_batch: 0,
                    step,
                    shuffle_state: rng.0,
                    order: order.clone(),
                    parameters: saved(model.snapshot()?),
                    epoch_loss: 0.,
                    epoch_examples: 0,
                    curves: curves.clone(),
                };
                write_json(&run_dir.join(format!("step-{step}.json")), &checkpoint)?;
            }
            let (accuracy, ce) = evaluate(&model, &dataset.test_x, &dataset.test_y, 0, 10000)?;
            if started.elapsed().as_secs_f64() > 1800. {
                return Err(
                    "fixed 1800s run budget exceeded during evaluation/checkpointing".into(),
                );
            }
            Ok(
                json!({"test_accuracy":accuracy,"test_loss":ce,"target":if ternary{0.90}else{0.95},"target_met":accuracy>=if ternary{0.90}else{0.95}}),
            )
        })();
        let status = match &result {
            Ok(_) => "completed",
            Err(_) => "failed",
        };
        let receipt = json!({"schema":1,"run":run,"seed":seed,"mode":mode,"status":status,"result":result.as_ref().ok(),"failure":result.as_ref().err().map(ToString::to_string),"epochs_completed":epoch,"step":step,"curves":curves,
            "wall_seconds":started.elapsed().as_secs_f64(),"training_step_seconds":total_step_seconds,"host_read_wait_seconds":host_read_wait_seconds,"host_read_wait_boundary":"Overlaps step walltime; host read calls include pending device work, finite-reduction kernels, synchronization and transfer; not pure synchronization latency","trained_examples":train_examples,
            "peak_rss_kib":peak_rss_kib(),"sampled_gpu_process_peak_mib":gpu_peak,"gpu_sampling_interval_steps":20,
            "resume_check":{"at_step":200,"compared_updates":resume_checked,"max_state_abs_error":resume_error,"atol":1e-5,"rtol":1e-4},
            "numerical_payload_bytes":{"matrix_parameters":406528,"bias_parameters":552,"gradients":407080,"adamw_moments":814160,"decoded_ternary_weights":if ternary{406528}else{0},"immutable_packed_matrix_export":25960,"packed_export_residency":"Payload size of hypothetical immutable exports; packed matrices are not retained by the training model"},
            "transfers_per_update":{"input_h2d_bytes":"batch*784*4","ternary_weight_d2h_bytes":if ternary{406528}else{0},"ternary_weight_h2d_bytes":if ternary{406528}else{0},"ternary_input_validation_d2h_bytes":if ternary{"batch*(784+128)*4"}else{"0"},"finite_check_d2h_scalars":16,"loss_d2h_bytes":4},
            "synchronization":"loss read, 16 finite-check scalar reads, plus two weight and two activation reads per ternary forward; timed inside training_step_seconds. JSON checkpoints/evaluation/resume comparisons add host reads and are included in wall_seconds.",
            "limitations":["Decoded matrices, latent parameters, gradients, moments, host copies and allocator/runtime memory coexist; packed payload is not total training memory","PeakRSS is process high-water mark across runs, not isolated per-model allocation; sampled GPU memory can miss peaks","No energy, inference throughput or total-memory saving claim"]});
        write_json(&run_dir.join("result.json"), &receipt)?;
        // Keep all six fixed-recipe outcomes, including failures; do not retune.
        if let Err(error) = result {
            any_failed = true;
            eprintln!("{mode}-{seed} failed: {error}");
        }
    }
    if any_failed {
        return Err("one or more fixed-recipe runs failed; all outcomes preserved".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn idx_parser_rejects_truncation_and_invalid_labels() {
        assert!(images(vec![], 1).is_err());
        assert!(labels(vec![0; 8], 1).is_err());
        let mut bytes = vec![0, 0, 8, 1, 0, 0, 0, 1, 10];
        assert!(labels(bytes.clone(), 1).is_err());
        bytes[8] = 9;
        assert_eq!(labels(bytes, 1).unwrap(), vec![9]);
    }
    #[test]
    fn paired_initialization_and_shuffle_are_reproducible() {
        assert_eq!(initial(17), initial(17));
        assert_ne!(initial(17), initial(29));
        let mut rng = Rng(17);
        let mut a: Vec<_> = (0..31).collect();
        rng.shuffle(&mut a);
        let mut b: Vec<_> = (0..31).collect();
        Rng(17).shuffle(&mut b);
        assert_eq!(a, b);
        a.sort_unstable();
        assert_eq!(a, (0..31).collect::<Vec<_>>());
        assert_eq!(Rng(rng.0).next(), rng.next());
    }
    #[test]
    fn serialized_cpu_checkpoint_replays_50_updates_after_step200() {
        let config: Value = serde_json::from_str(RECIPE).unwrap();
        let dir = std::env::temp_dir().join(format!("tergon-cp-{}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        let path = dir.join("state.json");
        let plan = PlacementPlan {
            compute: ComputeDevice::Cpu,
        };
        for ternary in [true, false] {
            let mut model = TinyClassifier::restore(&initial(17), ternary, plan).unwrap();
            let inputs: Vec<_> = (0..784).map(|i| (i % 23) as f32 / 23.).collect();
            for _ in 0..200 {
                model.step(&inputs, &[3]).unwrap();
            }
            let mut rng = Rng(17);
            let mut order: Vec<_> = (0..55000).collect();
            rng.shuffle(&mut order);
            let cp = Checkpoint {
                schema: 1,
                source_revision: "fixture".into(),
                config_sha256: "config".into(),
                dataset: config["dataset"].clone(),
                run: usize::from(!ternary),
                seed: 17,
                mode: if ternary {
                    "ternary".into()
                } else {
                    "float".into()
                },
                epoch: 0,
                next_batch: 25600,
                step: 200,
                shuffle_state: rng.0,
                order,
                parameters: saved(model.snapshot().unwrap()),
                epoch_loss: 10.,
                epoch_examples: 25600,
                curves: vec![],
            };
            validate_checkpoint(&cp, "fixture", "config", &config).unwrap();
            write_json(&path, &cp).unwrap();
            let loaded = read_checkpoint(&path).unwrap();
            let mut restored =
                TinyClassifier::restore(&snapshot(&loaded.parameters), ternary, plan).unwrap();
            for _ in 0..50 {
                model.step(&inputs, &[3]).unwrap();
                restored.step(&inputs, &[3]).unwrap();
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
                    assert_eq!(a.to_bits(), b.to_bits());
                }
            }
            assert_eq!(
                model
                    .logits(&inputs)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                restored
                    .logits(&inputs)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            let mut bad = cp.clone();
            bad.config_sha256 = "wrong".into();
            assert!(validate_checkpoint(&bad, "fixture", "config", &config).is_err());
            let mut bad = cp.clone();
            bad.dataset = Value::Null;
            assert!(validate_checkpoint(&bad, "fixture", "config", &config).is_err());
            let mut bad = cp.clone();
            bad.next_batch += 1;
            assert!(validate_checkpoint(&bad, "fixture", "config", &config).is_err());
            let mut bad = cp.clone();
            bad.shuffle_state += 1;
            assert!(validate_checkpoint(&bad, "fixture", "config", &config).is_err());
            let mut bad = cp.clone();
            bad.order[0] = bad.order[1];
            assert!(validate_checkpoint(&bad, "fixture", "config", &config).is_err());
            fs::write(&path, b"corrupt").unwrap();
            assert!(read_checkpoint(&path).is_err());
            fs::remove_file(&path).unwrap();
            fs::remove_file(path.with_extension("sha256")).unwrap();
        }
        fs::remove_dir(dir).unwrap();
    }
}
