//! Repository checks, independent of the experiment runtime.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn run(program: &str, args: &[&str]) -> Result<String> {
    let out = Command::new(program).args(args).output()?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned().into());
    }
    Ok(String::from_utf8(out.stdout)?)
}

fn architecture() -> Result<()> {
    let metadata: Value = serde_json::from_str(&run(
        "cargo",
        &["metadata", "--locked", "--no-deps", "--format-version", "1"],
    )?)?;
    let approved = [
        ("tergon-math", vec![]),
        ("tergon-engine", vec!["tergon-math", "burn"]),
        (
            "tergon-lab",
            vec![
                "tergon-math",
                "tergon-engine",
                "serde",
                "serde_json",
                "sha2",
            ],
        ),
        ("xtask", vec!["serde_json", "sha2"]),
    ];
    let packages = metadata["packages"].as_array().ok_or("missing packages")?;
    if packages.len() != approved.len() {
        return Err("workspace members changed: architecture review required".into());
    }
    for package in packages {
        let name = package["name"].as_str().ok_or("missing name")?;
        let allowed = &approved
            .iter()
            .find(|(n, _)| *n == name)
            .ok_or("unknown crate")?
            .1;
        for dep in package["dependencies"]
            .as_array()
            .ok_or("missing dependencies")?
        {
            if !allowed.contains(&dep["name"].as_str().ok_or("dependency name")?) {
                return Err(format!("unapproved dependency: {name} -> {}", dep["name"]).into());
            }
        }
        let manifest = fs::read_to_string(package["manifest_path"].as_str().ok_or("manifest")?)?;
        if !manifest.contains("[lints]\nworkspace = true") {
            return Err(format!("{name} must inherit workspace lints").into());
        }
    }
    for root in ["crates/tergon-math/src", "crates/tergon-engine/src"] {
        for path in files(Path::new(root))? {
            let source = fs::read_to_string(&path)?;
            for forbidden in ["std::fs", "std::net", "std::process", "std::io"] {
                if source.contains(forbidden) {
                    return Err(format!("{}: I/O belongs in lab", path.display()).into());
                }
            }
        }
    }
    Ok(())
}

fn files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut result = Vec::new();
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_symlink() {
            return Err(format!("symlink in checked tree: {}", path.display()).into());
        }
        if path.is_dir() {
            result.extend(files(&path)?);
        } else {
            result.push(path);
        }
    }
    result.sort();
    Ok(result)
}

fn docs() -> Result<()> {
    let mut paths = files(Path::new("docs"))?;
    paths.extend(["README.md", "AGENTS.md", "CONTRIBUTING.md", "NOTICE"].map(PathBuf::from));
    for path in paths {
        let content = fs::read_to_string(&path)?;
        for phrase in [
            "revolutionary",
            "unprecedented intelligence",
            "biological efficiency achieved",
        ] {
            if content.to_lowercase().contains(phrase) {
                return Err(format!("{}: unsupported promotional phrase", path.display()).into());
            }
        }
        let mut fenced = false;
        for line in content.lines() {
            if line.trim_start().starts_with("```") {
                fenced = !fenced;
                continue;
            }
            if fenced {
                continue;
            }
            let mut remainder = line;
            while let Some(start) = remainder.find("](") {
                remainder = &remainder[start + 2..];
                let Some(end) = remainder.find(')') else {
                    break;
                };
                let target = remainder[..end].split('#').next().unwrap_or("");
                if !target.is_empty() && !target.contains("://") && !target.starts_with("mailto:") {
                    let destination = path.parent().unwrap_or(Path::new(".")).join(target);
                    if !destination.exists() {
                        return Err(
                            format!("{}: broken local link {target}", path.display()).into()
                        );
                    }
                }
                remainder = &remainder[end + 1..];
            }
        }
    }
    Ok(())
}

fn evidence() -> Result<()> {
    let experiment: Value = serde_json::from_slice(&fs::read("experiments/quantization-v1.json")?)?;
    for field in ["hypothesis", "baseline", "seeds", "budget", "tolerance"] {
        if experiment.get(field).is_none() {
            return Err(format!("experiment lacks {field}").into());
        }
    }
    let inventory: Value = serde_json::from_slice(&fs::read("evidence/inventory.json")?)?;
    let items = inventory["files"]
        .as_array()
        .ok_or("evidence inventory lacks files")?;
    if items.is_empty() {
        return Err("evidence inventory is empty".into());
    }
    for item in items {
        let relative = item["path"].as_str().ok_or("inventory path")?;
        if Path::new(relative)
            .components()
            .any(|p| !matches!(p, std::path::Component::Normal(_)))
        {
            return Err("inventory paths must remain inside the repository".into());
        }
        let data = fs::read(relative)?;
        if item["sha256"] != format!("{:x}", Sha256::digest(&data))
            || item["bytes"] != data.len() as u64
        {
            return Err(format!("evidence mismatch: {relative}").into());
        }
    }
    Ok(())
}

fn contribution() -> Result<()> {
    let policy: Value = serde_json::from_slice(&fs::read("policy/contributions.json")?)?;
    let scopes: Value = serde_json::from_slice(&fs::read("policy/agent-scopes.json")?)?;
    if policy["schema"] != 1 || scopes["schema"] != 1 {
        return Err("unknown policy schema".into());
    }
    // Enabling this requires a separate implementation and identity-boundary
    // tests. A policy flag alone must never silently enable merges.
    if scopes["enabled"] != false
        || !scopes["scopes"]
            .as_array()
            .ok_or("missing scopes")?
            .is_empty()
    {
        return Err("scoped merging is not implemented or identity-validated yet".into());
    }
    for path in [
        ".github/pull_request_template.md",
        ".github/CODEOWNERS",
        "AGENTS.md",
        "CONTRIBUTING.md",
    ] {
        if !Path::new(path).is_file() {
            return Err(format!("missing contribution file: {path}").into());
        }
    }
    Ok(())
}

fn wiki_snapshot(directory: &Path) -> Result<Value> {
    let mut hashes = serde_json::Map::new();
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or("wiki filename")?;
        if name == ".git" || name == ".tergon-sync.json" {
            continue;
        }
        if path.is_symlink()
            || !path.is_file()
            || path.extension().and_then(|s| s.to_str()) != Some("md")
        {
            return Err(format!("unmanaged wiki content: {name}").into());
        }
        hashes.insert(
            name.to_owned(),
            Value::String(format!("{:x}", Sha256::digest(fs::read(&path)?))),
        );
    }
    Ok(Value::Object(hashes))
}

fn check_wiki_drift(directory: &Path) -> Result<()> {
    let snapshot = wiki_snapshot(directory)?;
    let previous = directory.join(".tergon-sync.json");
    if previous.is_symlink() {
        return Err("wiki receipt cannot be a symlink".into());
    }
    if previous.exists() {
        let receipt: Value = serde_json::from_slice(&fs::read(previous)?)?;
        if receipt["schema"] != 1 || receipt["files"] != snapshot {
            return Err("wiki drift: reconcile edits before publishing".into());
        }
    } else if snapshot.as_object().ok_or("wiki snapshot")?.len() != 1
        || fs::read_to_string(directory.join("Home.md"))?.trim() != "# Tergon Lab"
    {
        return Err("wiki is uninitialized or has unmanaged edits; refusing overwrite".into());
    }
    Ok(())
}

fn wiki_sync(directory: &Path) -> Result<()> {
    if !run("git", &["status", "--porcelain"])?.trim().is_empty() {
        return Err("publish only a clean reviewed source checkout".into());
    }
    run("git", &["fetch", "origin", "main"])?;
    let sha = run("git", &["rev-parse", "HEAD"])?.trim().to_owned();
    if sha != run("git", &["rev-parse", "origin/main"])?.trim() {
        return Err("wiki source must be current origin/main".into());
    }
    check_wiki_drift(directory)?;
    let old: Vec<String> = wiki_snapshot(directory)?
        .as_object()
        .ok_or("snapshot")?
        .keys()
        .cloned()
        .collect();
    let sources = files(Path::new("docs/wiki"))?;
    for source in &sources {
        let name = source.file_name().ok_or("wiki page filename")?;
        let content = format!(
            "{}\n---\nPublished from [source commit {sha}](https://github.com/kmosoti/tergon-lab/tree/{sha}/docs/wiki).\n",
            fs::read_to_string(source)?
        );
        fs::write(directory.join(name), content)?;
    }
    for name in old {
        if !Path::new("docs/wiki").join(&name).exists() {
            fs::remove_file(directory.join(name))?;
        }
    }
    let receipt = serde_json::json!({ "schema": 1, "source_revision": sha, "files": wiki_snapshot(directory)? });
    fs::write(
        directory.join(".tergon-sync.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!("wiki staged from {sha}; inspect the diff before committing");
    Ok(())
}

fn main() -> Result<()> {
    std::env::set_current_dir(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or("workspace root")?,
    )?;
    let task = std::env::args().nth(1).unwrap_or_else(|| "check".into());
    match task.as_str() {
        "architecture" => architecture()?, "docs" => docs()?,
        "evidence" => evidence()?, "contribution" => contribution()?,
        "check" => { architecture()?; docs()?; evidence()?; contribution()?; }
        "wiki-sync" => wiki_sync(Path::new(&std::env::args().nth(2).ok_or("wiki-sync needs a wiki checkout directory")?))?,
        _ => return Err("usage: cargo xtask [check|architecture|docs|evidence|contribution|wiki-sync <directory>]".into()),
    }
    println!("{task}: passed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wiki_accepts_only_bootstrap_or_unchanged_managed_pages() {
        let temp = std::env::temp_dir().join(format!("tergon-wiki-test-{}", std::process::id()));
        fs::create_dir(&temp).unwrap();
        fs::write(temp.join("Home.md"), "# Tergon Lab\n").unwrap();
        assert!(check_wiki_drift(&temp).is_ok());
        fs::write(temp.join("Home.md"), "reviewed page").unwrap();
        assert!(check_wiki_drift(&temp).is_err());
        let receipt = serde_json::json!({"schema": 1, "files": wiki_snapshot(&temp).unwrap()});
        fs::write(
            temp.join(".tergon-sync.json"),
            serde_json::to_vec(&receipt).unwrap(),
        )
        .unwrap();
        assert!(check_wiki_drift(&temp).is_ok());
        fs::write(temp.join("Home.md"), "unreviewed web edit").unwrap();
        assert!(check_wiki_drift(&temp).is_err());
        fs::write(temp.join("Home.md"), "reviewed page").unwrap();
        fs::write(temp.join("Extra.md"), "unmanaged page").unwrap();
        assert!(check_wiki_drift(&temp).is_err());
        fs::remove_file(temp.join("Extra.md")).unwrap();
        fs::remove_file(temp.join("Home.md")).unwrap();
        assert!(check_wiki_drift(&temp).is_err());
        fs::remove_dir_all(temp).unwrap();
    }
}
