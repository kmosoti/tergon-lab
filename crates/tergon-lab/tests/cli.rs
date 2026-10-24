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
