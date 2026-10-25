//! Evidence outputs must never replace an earlier run.
use std::{fs, path::Path, process::Command};

#[test]
fn existing_result_is_preserved() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output =
        std::env::temp_dir().join(format!("tergon-cli-existing-{}.json", std::process::id()));
    let sentinel = b"preserve the original run\n";
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .unwrap();
    fs::write(&output, sentinel).unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_tergon-lab"))
        .current_dir(root)
        .args(["smoke", "experiments/quantization-v1.json"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(!status.status.success());
    assert_eq!(fs::read(&output).unwrap(), sentinel);
    fs::remove_file(output).unwrap();
}

#[test]
fn train_rejects_changed_recipe_and_missing_dataset_without_output() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = std::env::temp_dir().join(format!("tergon-train-cli-{}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let mut config: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("experiments/mnist-adamw-v1.json")).unwrap())
            .unwrap();
    config["epochs"] = serde_json::json!(1);
    let changed = dir.join("changed.json");
    fs::write(&changed, serde_json::to_vec(&config).unwrap()).unwrap();
    let output = dir.join("output");
    let result = Command::new(env!("CARGO_BIN_EXE_tergon-lab"))
        .current_dir(&root)
        .arg("train")
        .arg(&changed)
        .arg(&dir)
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("frozen"));
    assert!(!output.exists());
    let result = Command::new(env!("CARGO_BIN_EXE_tergon-lab"))
        .current_dir(&root)
        .args(["train", "experiments/mnist-adamw-v1.json"])
        .arg(&dir)
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!output.exists());
    fs::remove_file(changed).unwrap();
    fs::remove_dir(dir).unwrap();
}
