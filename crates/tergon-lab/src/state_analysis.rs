//! Predeclared paired-world bootstrap and conservative all-condition gate.
use crate::state_data::{GROUPS, Rng};
use crate::state_training::Prediction;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::{Path, PathBuf},
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn receipts(root: &Path, depth: usize, out: &mut Vec<PathBuf>) -> Result<()> {
    if depth > 3 {
        return Ok(());
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_symlink() {
            continue;
        }
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            receipts(&path, depth + 1, out)?;
        } else if entry.file_name() == "receipt.json" {
            out.push(path);
        }
    }
    Ok(())
}
fn checked_json(path: &Path) -> Result<Value> {
    let bytes = fs::read(path)?;
    if fs::read_to_string(path.with_extension("sha256"))?.trim() != crate::hash(&bytes) {
        return Err(format!("checksum mismatch {}", path.display()).into());
    }
    Ok(serde_json::from_slice(&bytes)?)
}
fn accuracy(rows: &[Prediction]) -> f64 {
    rows.iter().filter(|r| r.label == r.predicted).count() as f64 / rows.len() as f64
}
fn brier(rows: &[Prediction]) -> f64 {
    rows.iter().map(|r| r.brier).sum::<f64>() / rows.len() as f64
}
/// Read immutable per-condition evidence; require the full twelve-run matched experiment.
pub(crate) fn analyze(root: &Path, output: &Path) -> Result<()> {
    let mut paths = Vec::new();
    receipts(root, 0, &mut paths)?;
    paths.sort();
    let mut identity = None;
    let mut conditions = BTreeMap::new();
    let mut failures = Vec::new();
    for path in paths {
        let receipt = checked_json(&path)?;
        let current = json!({"source_revision":receipt["source_revision"],"config_sha256":receipt["config_sha256"],"dataset_sha256":receipt["dataset_sha256"]});
        if receipt["source_dirty"] != false || receipt["experiment"] != "typed-state-v1" {
            return Err("dirty or unsupported experiment receipt".into());
        }
        if let Some(expected) = &identity {
            if *expected != current {
                return Err("source/config/data identity mismatch across conditions".into());
            }
        } else {
            identity = Some(current);
        }
        for run in receipt["runs"].as_array().ok_or("missing runs")? {
            let seed = run["seed"].as_u64().ok_or("missing seed")?;
            let policy = run["policy"].as_str().ok_or("missing policy")?;
            let mode = run["mode"].as_str().ok_or("missing mode")?;
            if ![17, 29, 43].contains(&seed)
                || !["metadata-only", "enforced"].contains(&policy)
                || !["float", "ternary"].contains(&mode)
            {
                return Err("unexpected condition".into());
            }
            let key = (seed, policy.to_owned(), mode.to_owned());
            if conditions.contains_key(&key) {
                return Err("duplicate condition receipts".into());
            }
            if run["completed"] == true
                && (run["serialized_replay"]["completed"] != true
                    || run["serialized_replay"]["updates"] != 50
                    || run["serialized_replay"]["batch_size"] != 128
                    || run["serialized_replay"]["last_step"] != 250
                    || run["epochs"] != 20
                    || run["updates"] != 5120
                    || run["elapsed_seconds"]
                        .as_f64()
                        .is_none_or(|v| !v.is_finite() || !(0.0..=600.0).contains(&v)))
            {
                return Err("completed run lacks full bounded execution".into());
            }
            if run["completed"] != true {
                failures.push(
                    json!({"seed":seed,"policy":policy,"mode":mode,"failure":run["failure"]}),
                );
            }
            let mut groups = Vec::new();
            if run["completed"] == true {
                for name in GROUPS.iter().skip(2) {
                    let file = path
                        .parent()
                        .unwrap()
                        .join(format!("{seed}-{policy}-{mode}/{name}-predictions.json"));
                    let rows: Vec<Prediction> = serde_json::from_value(checked_json(&file)?)?;
                    if rows.len() != 2048
                        || rows.iter().any(|r| {
                            r.group != *name
                                || r.probabilities.len() != 18
                                || r.label >= 18
                                || r.predicted >= 18
                                || !r.brier.is_finite()
                        })
                    {
                        return Err("malformed prediction rows".into());
                    }
                    let mut worlds: BTreeMap<u64, BTreeSet<usize>> = BTreeMap::new();
                    for row in &rows {
                        if !worlds.entry(row.world).or_default().insert(row.branch) {
                            return Err("duplicate world/branch prediction".into());
                        }
                    }
                    if worlds.len() != 512
                        || worlds
                            .values()
                            .any(|branches| *branches != BTreeSet::from([0, 1, 2, 3]))
                    {
                        return Err("expected512 unique complete world clusters per stratum".into());
                    }
                    for row in &rows {
                        let sum = row.probabilities.iter().sum::<f64>();
                        let expected_brier = row
                            .probabilities
                            .iter()
                            .enumerate()
                            .map(|(i, p)| (p - f64::from(i == row.label)).powi(2))
                            .sum::<f64>();
                        let mut predicted = 0;
                        for i in 1..18 {
                            if row.probabilities[i] > row.probabilities[predicted] {
                                predicted = i;
                            }
                        }
                        if row
                            .probabilities
                            .iter()
                            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
                            || (sum - 1.).abs() > 1e-10
                            || (expected_brier - row.brier).abs() > 1e-10
                            || predicted != row.predicted
                            || row.branch > 3
                            || !(1..=3).contains(&row.hops)
                        {
                            return Err("invalid probability, prediction or Brier evidence".into());
                        }
                    }
                    groups.push(rows);
                }
            }
            conditions.insert(key, groups);
        }
    }
    if conditions.len() != 12 || !failures.is_empty() {
        let result = json!({"schema":1,"identity":identity,"conditions_present":conditions.len(),"failures":failures,"status":"inconclusive","gate_open":false,"reason":"all12 distinct complete matched conditions required"});
        return write_output(output, &result);
    }
    let get = |seed: u64, policy: &str, mode: &str| {
        &conditions[&(seed, policy.to_owned(), mode.to_owned())]
    };
    let reference = get(17, "metadata-only", "float");
    for condition in conditions.values() {
        for group in 0..5 {
            for (a, b) in reference[group].iter().zip(&condition[group]) {
                if (a.world, a.branch, a.label) != (b.world, b.branch, b.label) {
                    return Err("cross-seed paired prediction order/identity mismatch".into());
                }
            }
        }
    }
    let mut seed_results = Vec::new();
    let mut gate = true;
    let mut controls_pass = true;
    let mut gains = Vec::new();
    let mut world_differences = vec![vec![0.; 512]; 4];
    for seed in [17, 29, 43] {
        let float = get(seed, "metadata-only", "float");
        let metadata = get(seed, "metadata-only", "ternary");
        let enforced = get(seed, "enforced", "ternary");
        for condition in [float, metadata, enforced, get(seed, "enforced", "float")] {
            for group in 0..5 {
                let rows = &condition[group];
                for (reference, row) in metadata[group].iter().zip(rows) {
                    if (reference.world, reference.branch, reference.label)
                        != (row.world, row.branch, row.label)
                    {
                        return Err("paired prediction identity/order mismatch".into());
                    }
                }
            }
        }
        let ordinary_float = accuracy(&float[0]);
        controls_pass &= ordinary_float >= 0.9;
        let ordinary = accuracy(&enforced[0]);
        let ordinary_delta = ordinary - accuracy(&metadata[0]);
        let stress_enforced = enforced[1..].iter().map(|r| accuracy(r)).sum::<f64>() / 4.;
        let stress_metadata = metadata[1..].iter().map(|r| accuracy(r)).sum::<f64>() / 4.;
        let gain = stress_enforced - stress_metadata;
        let brier_delta = (0..5)
            .map(|g| brier(&enforced[g]) - brier(&metadata[g]))
            .sum::<f64>()
            / 5.;
        gains.push(gain);
        let passed = ordinary_float >= 0.9
            && ordinary >= 0.9
            && stress_enforced >= 0.85
            && gain > 0.
            && ordinary_delta >= -0.02
            && brier_delta <= 0.02;
        gate &= passed;
        seed_results.push(json!({"seed":seed,"metadata_float_ordinary":ordinary_float,"enforced_ternary_ordinary":ordinary,"enforced_ternary_stress":stress_enforced,"metadata_ternary_stress":stress_metadata,"stress_gain":gain,"ordinary_delta":ordinary_delta,"five_group_brier_delta":brier_delta,"per_seed_gate_pass":passed}));
        for group in 1..5 {
            for (world, (a, b)) in enforced[group]
                .as_chunks::<4>()
                .0
                .iter()
                .zip(metadata[group].as_chunks::<4>().0)
                .enumerate()
            {
                if a.iter().any(|r| r.world != a[0].world)
                    || b.iter().any(|r| r.world != a[0].world)
                {
                    return Err("prediction world clustering broken".into());
                }
                let delta = (0..4)
                    .map(|i| {
                        f64::from(a[i].label == a[i].predicted)
                            - f64::from(b[i].label == b[i].predicted)
                    })
                    .sum::<f64>()
                    / 4.;
                world_differences[group - 1][world] += delta / 3.;
            }
        }
    }
    let mean_gain = gains.iter().sum::<f64>() / 3.;
    let mut rng = Rng(20261025);
    let mut bootstrap = Vec::with_capacity(10000);
    for _ in 0..10000 {
        let mut sum = 0.;
        for stratum in &world_differences {
            for _ in 0..512 {
                sum += stratum[rng.bounded(512)];
            }
        }
        bootstrap.push(sum / (4. * 512.));
    }
    bootstrap.sort_by(f64::total_cmp);
    let low = percentile(&bootstrap, 0.025);
    let high = percentile(&bootstrap, 0.975);
    gate &= mean_gain >= 0.05 && low > 0.;
    let seed_variance = gains.iter().map(|g| (g - mean_gain).powi(2)).sum::<f64>() / 2.;
    let result = json!({"schema":1,"identity":identity,"conditions_present":12,"status":if controls_pass{"completed"}else{"inconclusive"},"control_pass":controls_pass,"conclusion":if !controls_pass{"competent baseline control failed"}else if gate{"numeric storage-pilot gate passed"}else{"completed negative gate result"},"gate_open":gate,"seed_results":seed_results,"mean_ternary_stress_gain":mean_gain,"seed_sample_standard_deviation":seed_variance.sqrt(),"paired_world_bootstrap":{"resamples":10000,"rng_seed":20261025,"unit":"whole world within stress stratum; model seeds averaged inside world","two_sided_95_percentile_interval":[low,high],"percentile_rule":"linear interpolation at (n-1)*p"}});
    write_output(output, &result)
}
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let rank = (sorted.len() - 1) as f64 * p;
    let i = rank.floor() as usize;
    let fraction = rank - i as f64;
    sorted[i] * (1. - fraction) + sorted[(i + 1).min(sorted.len() - 1)] * fraction
}
fn write_output(path: &Path, value: &Value) -> Result<()> {
    use std::io::Write;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn percentile_is_predeclared_linear_interpolation() {
        assert_eq!(percentile(&[0., 1., 2., 3.], 0.25), 0.75);
        assert_eq!(percentile(&[0., 1., 2., 3.], 1.), 3.);
    }
    fn fixture_json(path: &Path, value: &impl serde::Serialize) {
        let bytes = serde_json::to_vec(value).unwrap();
        fs::write(path.with_extension("sha256"), crate::hash(&bytes)).unwrap();
        fs::write(path, bytes).unwrap();
    }
    #[test]
    fn complete_paired_study_opens_gate_but_incomplete_or_duplicate_cannot() {
        let dir =
            std::env::temp_dir().join(format!("state-analysis-complete-{}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        let mut runs = Vec::new();
        for seed in [17_u64, 29, 43] {
            for policy in ["metadata-only", "enforced"] {
                for mode in ["float", "ternary"] {
                    let condition = dir.join(format!("{seed}-{policy}-{mode}"));
                    fs::create_dir(&condition).unwrap();
                    for (group, name) in GROUPS.iter().enumerate().skip(2) {
                        let rows: Vec<_> = (0..2048)
                            .map(|index| {
                                let branch = index % 4;
                                let label = [0, 1, 16, 17][branch];
                                let wrong = policy == "metadata-only"
                                    && mode == "ternary"
                                    && group > 2
                                    && index % 10 == 0;
                                let predicted = if wrong { (label + 1) % 18 } else { label };
                                let mut probabilities = vec![0.; 18];
                                probabilities[predicted] = 1.;
                                Prediction {
                                    group: (*name).into(),
                                    world: ((group as u64) << 48) | (index / 4) as u64,
                                    branch,
                                    hops: 1 + index / 4 % 3,
                                    label,
                                    predicted,
                                    probabilities,
                                    brier: if wrong { 2. } else { 0. },
                                    confidently_unsupported: wrong && predicted < 16,
                                    renamed_consistent: true,
                                    reordered_consistent: true,
                                }
                            })
                            .collect();
                        fixture_json(&condition.join(format!("{name}-predictions.json")), &rows);
                    }
                    runs.push(json!({"seed":seed,"policy":policy,"mode":mode,"completed":true,"epochs":20,"updates":5120,"elapsed_seconds":100.,"serialized_replay":{"completed":true,"updates":50,"batch_size":128,"last_step":250}}));
                }
            }
        }
        let mut receipt = json!({"experiment":"typed-state-v1","source_revision":"fixture","source_dirty":false,"config_sha256":"config","dataset_sha256":["train","val","a","b","c","d","e"],"runs":runs});
        fixture_json(&dir.join("receipt.json"), &receipt);
        let out = dir.join("complete.json");
        analyze(&dir, &out).unwrap();
        let result: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
        assert_eq!(result["gate_open"], true);
        assert_eq!(result["status"], "completed");
        assert!(
            (result["mean_ternary_stress_gain"].as_f64().unwrap() - 0.10009765625).abs() < 1e-12
        );
        assert!(
            result["paired_world_bootstrap"]["two_sided_95_percentile_interval"][0]
                .as_f64()
                .unwrap()
                > 0.
        );
        receipt["runs"][0]["completed"] = json!(false);
        fixture_json(&dir.join("receipt.json"), &receipt);
        let out = dir.join("incomplete.json");
        analyze(&dir, &out).unwrap();
        let result: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
        assert_eq!(result["gate_open"], false);
        assert_eq!(result["status"], "inconclusive");
        let duplicate = dir.join("duplicate");
        fs::create_dir(&duplicate).unwrap();
        fixture_json(&duplicate.join("receipt.json"), &receipt);
        assert!(analyze(&dir, &dir.join("duplicate-result.json")).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn missing_conditions_cannot_open_gate() {
        let dir = std::env::temp_dir().join(format!("state-analysis-{}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        let out = dir.join("result.json");
        analyze(&dir, &out).unwrap();
        let result: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
        assert_eq!(result["gate_open"], false);
        assert_eq!(result["status"], "inconclusive");
        fs::remove_dir_all(dir).unwrap();
    }
}
