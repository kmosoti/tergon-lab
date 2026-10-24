//! Reproducible component experiments.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{error::Error, fs, io::Write, path::Path, process::Command};
use tergon_engine::{ComputeDevice, PlacementPlan, materialize};
use tergon_math::{PackedTernary, QuantSpec};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Experiment {
    schema: u32,
    hypothesis: String,
    baseline: String,
    seeds: Vec<u64>,
    weights_per_seed: usize,
    group_size: usize,
    budget: String,
    tolerance: f64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunManifest {
    schema: u32,
    kind: String,
    source_revision: String,
    source_dirty: bool,
    config_sha256: String,
    rustc: String,
    device: String,
    architecture: String,
    energy_joules: Option<f64>,
    limitations: Vec<String>,
    results: Vec<FixtureResult>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureResult {
    seed: u64,
    input_sha256: String,
    decoded_sha256: String,
    weights: usize,
    packed_weight_bytes: usize,
    scale_bytes: usize,
    dense_f32_bytes: usize,
    max_quantization_error: f64,
    max_bridge_error: f64,
    passed: bool,
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn float_hash(values: &[f32]) -> String {
    hash(
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn command(program: &str, args: &[&str]) -> Result<String, Box<dyn Error>> {
    let output = Command::new(program).args(args).output()?;
    if !output.status.success() {
        return Err(format!("{program} failed").into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn smoke(config: &Path, out: &Path) -> Result<(), Box<dyn Error>> {
    let bytes = fs::read(config)?;
    let exp: Experiment = serde_json::from_slice(&bytes)?;
    if exp.schema != 1
        || exp.seeds.is_empty()
        || exp.seeds.len() > 64
        || exp.weights_per_seed == 0
        || exp.weights_per_seed > 1_000_000
        || exp.tolerance != 0.0
    {
        return Err("unsupported experiment schema, size, or tolerance".into());
    }
    let spec = QuantSpec::new(exp.group_size)?;
    // Reserve the output before execution. Never replace an earlier result.
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out)?;
    let mut manifest = RunManifest {
        schema: 1,
        kind: "quantization-smoke".into(),
        source_revision: command("git", &["rev-parse", "HEAD"])?,
        source_dirty: !command("git", &["status", "--porcelain"])?.is_empty(),
        config_sha256: hash(&bytes),
        rustc: command("rustc", &["--version"])?,
        device: "burn-flex-cpu-f32".into(),
        architecture: std::env::consts::ARCH.into(),
        energy_joules: None,
        limitations: vec![
            "CPU component correctness only; no training or throughput conclusion".into(),
            "Payload bytes exclude allocator, tensor, and runtime overhead".into(),
            "No energy sensor available to this command".into(),
        ],
        results: Vec::new(),
    };
    for seed in exp.seeds {
        let mut state = seed;
        let input: Vec<f32> = (0..exp.weights_per_seed)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 32) as i32 as f32 / 65536.0
            })
            .collect();
        let packed = PackedTernary::quantize(&input, spec)?;
        let decoded: Vec<_> = packed.dequantize().collect();
        let tensor = materialize(
            &packed,
            1,
            input.len(),
            PlacementPlan {
                compute: ComputeDevice::Cpu,
            },
        )?;
        let actual = tensor
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| format!("{e:?}"))?;
        let max_bridge_error = decoded
            .iter()
            .zip(&actual)
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .fold(0.0, f64::max);
        manifest.results.push(FixtureResult {
            seed,
            input_sha256: float_hash(&input),
            decoded_sha256: float_hash(&decoded),
            weights: input.len(),
            packed_weight_bytes: packed.bytes().len(),
            scale_bytes: std::mem::size_of_val(packed.scales()),
            dense_f32_bytes: std::mem::size_of_val(input.as_slice()),
            max_quantization_error: input
                .iter()
                .zip(&decoded)
                .map(|(&a, &b)| (a as f64 - b as f64).abs())
                .fold(0.0, f64::max),
            max_bridge_error,
            passed: actual == decoded,
        });
    }
    output.write_all(&serde_json::to_vec_pretty(&manifest)?)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    if manifest.results.iter().any(|r| !r.passed) {
        return Err("bridge mismatch; failure saved".into());
    }
    println!(
        "{} fixtures passed; receipt: {}",
        manifest.results.len(),
        out.display()
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [cmd, config, out] if cmd == "smoke" => smoke(Path::new(config), Path::new(out)),
        _ => Err("usage: tergon-lab smoke <experiment.json> <new-receipt.json>".into()),
    }
}
