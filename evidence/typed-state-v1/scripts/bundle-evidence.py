#!/usr/bin/env python3
"""Stage immutable typed-state evidence outside the repository after the frozen study.

This program copies observations and computes identities; it never runs training,
changes source, infers a gate from incomplete data, or rewrites an old finding.
"""
import argparse
import csv
import datetime
import gzip
import hashlib
import io
import itertools
import json
import pathlib
import platform
import shutil
import subprocess
import sys

GROUPS = ["train", "validation", "ordinary", "unseen-composition", "replace-retract", "conflicts-hypotheses", "distractors"]
CONDITIONS = list(itertools.product((17, 29, 43), ("metadata-only", "enforced"), ("float", "ternary")))
FROZEN = "4144184b1a83986f44feee36bcef3bfc0a086c84"
BINARY_SHA = "2cbc1f5106f048d7429eeb3b6b0d7f92ab2dba06fa29aecfc4324d843bdfa2cf"
ROOT_SCRIPTS = ["tergon-run-frozen-state.py", "tergon-independent-data-audit.py", "tergon-independent-state-results.py"]
SOURCE_FILES = [
    "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "experiments/typed-state-v1.json",
    "crates/tergon-engine/src/state.rs", "crates/tergon-engine/src/state_reasoner.rs",
    "crates/tergon-engine/src/state_reasoner_tests.rs", "crates/tergon-engine/src/training.rs",
    "crates/tergon-engine/src/ternary.rs", "crates/tergon-lab/src/state_data.rs",
    "crates/tergon-lab/src/state_training.rs", "crates/tergon-lab/src/state_analysis.rs",
    "crates/tergon-lab/src/main.rs", "docs/wiki/Typed-state-experiment.md", "docs/wiki/Typed-state-ADR.md",
]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path, checksum=False):
    raw = path.read_bytes()
    if checksum:
        assert path.with_suffix(".sha256").read_text().strip() == digest(raw), path
    return json.loads(raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=pathlib.Path, required=True)
    ap.add_argument("--study", type=pathlib.Path, required=True)
    ap.add_argument("--corpus", type=pathlib.Path, required=True)
    ap.add_argument("--preflight", type=pathlib.Path, required=True)
    ap.add_argument("--analysis", type=pathlib.Path, required=True)
    ap.add_argument("--independent-results", type=pathlib.Path)
    ap.add_argument("--checks", type=pathlib.Path, help="Optional post-study clean-freeze check logs/receipt")
    ap.add_argument("--runner-log", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--allow-incomplete", action="store_true", help="Preserve a fully attempted but incomplete/inconclusive study; never opens a gate")
    args = ap.parse_args()
    # Explicitly prevent staging anywhere in the active worktree or study.
    output = args.out.resolve()
    for frozen_root in (args.source.resolve(), args.study.resolve(), args.corpus.resolve()):
        assert not output.is_relative_to(frozen_root), "stage outside frozen source/study/corpus"
    execution = load(args.study / "execution.json")
    assert execution["source_revision"] == FROZEN
    assert execution["binary_sha256"] == BINARY_SHA
    corpus = load(args.corpus / "audit.json")
    assert corpus["source_revision"] == FROZEN and corpus["source_dirty"] is False
    assert corpus["config_sha256"] == execution["config_sha256"]
    independent_data = load(args.corpus / "independent-python-audit.json")
    assert independent_data["passed"] and independent_data["source_revision"] == FROZEN
    assert independent_data["audit_sha256"] == digest((args.corpus / "audit.json").read_bytes())
    assert independent_data["unique_worlds"] == 11776
    official = load(args.analysis)
    assert official["identity"]["source_revision"] == FROZEN
    assert official["identity"]["config_sha256"] == execution["config_sha256"]
    if args.analysis.with_suffix(".sha256").exists():
        load(args.analysis, checksum=True)
    data_hashes = [next(g["sha256"] for g in corpus["groups"] if g["group"] == name) for name in GROUPS]
    assert official["identity"]["dataset_sha256"] == data_hashes
    events = [json.loads(line) for line in (args.study / "progress.jsonl").read_text().splitlines()]
    finished = {e["condition"]: e for e in events if "finished_utc" in e}
    expected = {f"{seed}:{policy}:{mode}" for seed, policy, mode in CONDITIONS}
    assert set(finished) == expected and sum("finished_utc" in e for e in events) == 12, "wait for all twelve attempts"
    if not args.allow_incomplete:
        assert all(e["exit_code"] == 0 for e in finished.values())
        assert official["conditions_present"] == 12
    output.mkdir()
    sources = []
    script_root=pathlib.Path(__file__).resolve().parent

    def write(relative, raw, origin=None, role=None):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as f:
            f.write(raw)
        sources.append({"path": relative, "bytes": len(raw), "sha256": digest(raw),
                        **({"origin": origin} if origin else {}), **({"role": role} if role else {})})

    def copy(source, relative, role=None):
        write(relative, source.read_bytes(), str(source), role)

    def write_json(relative, value, role=None):
        write(relative, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode(), role=role)

    def copy_checked(source, relative, role=None):
        load(source, checksum=True)
        copy(source, relative, role)
        copy(source.with_suffix(".sha256"), str(pathlib.Path(relative).with_suffix(".sha256")), "original checksum sidecar")

    copy(args.study / "execution.json", "execution.json", "frozen fresh-process runner provenance")
    copy(args.study / "progress.jsonl", "progress.jsonl", "every attempt/start/end/exit retained")
    if args.runner_log:
        copy(args.runner_log, "runner.log", "outer runner output, including failures")
    copy(args.analysis, "analysis.json", "predeclared Rust aggregate result")
    write("analysis.sha256", (digest((args.analysis).read_bytes()) + "\n").encode())
    if args.independent_results:
        independent = load(args.independent_results)
        assert independent["passed"] is True
        assert independent["source_revision"] == FROZEN
        assert independent["official_analysis_sha256"] == digest(args.analysis.read_bytes())
        assert independent["gate_open"] == official["gate_open"]
        copy(args.independent_results, "independent-results-audit.json", "independent checkpoint/prediction/bootstrap verification")
    elif not args.allow_incomplete:
        raise AssertionError("complete study requires independent results verification")
    copy(args.corpus / "audit.json", "corpus/audit.json", "clean final generator/oracle/metadata/degree audit")
    copy(args.corpus / "independent-python-audit.json", "corpus/independent-python-audit.json", "independent event replay and split/degree audit")
    dataset_inventory = []
    for group in corpus["groups"]:
        path = args.corpus / (group["group"] + ".json")
        raw = path.read_bytes()
        assert digest(raw) == group["sha256"] and len(raw) == group["bytes"]
        dataset_inventory.append({"group": group["group"], "worlds": group["worlds"], "queries": group["queries"],
                                  "bytes": len(raw), "sha256": digest(raw), "local_path": str(path),
                                  "retention": "local JSON; bit-exact regeneration from frozen source/recipe"})
    write_json("corpus/dataset-inventory.json", {"schema": 1, "source_revision": FROZEN, "files": dataset_inventory})
    for name in ROOT_SCRIPTS:
        source = script_root / name
        copy(source, "scripts/" + name, "unaltered independent/runner script")
    assert digest((script_root / ROOT_SCRIPTS[0]).read_bytes()) == execution["runner_sha256"]
    if args.independent_results:
        assert digest((script_root / ROOT_SCRIPTS[2]).read_bytes()) == independent["script_sha256"]
    copy(pathlib.Path(__file__), "scripts/bundle-evidence.py", "deterministic artifact collection; no training")
    # Preserve all development logs, including failed ideal-f64 and initial TF32 probes.
    for path in sorted(args.preflight.iterdir()):
        if path.is_file() and path.suffix in (".txt", ".log", ".json"):
            copy(path, "validation/development/" + path.name, "development-only; never a formal learning result")
    if args.checks:
        for path in sorted(args.checks.iterdir()):
            if path.is_file():
                copy(path, "validation/required/" + path.name, "required final checks and provenance")
    else:
        write_json("validation/check-log-limitations.json", {
            "schema": 1, "source_revision": FROZEN,
            "note": "Precommit checks passed on the source tree committed as frozen SHA. Raw workspace/CUDA logs are retained in development/. Separate original stdout for fmt/base-clippy/rustdoc/xtask was not persisted; repeat the five checks after formal runs end and supply --checks before committing the bundle.",
            "original_required_outcomes": {"fmt": "pass", "clippy": "pass", "workspace_tests": "pass", "rustdoc": "pass", "xtask": "pass"},
            "recorded_in": "implementation PR17 validation narrative; no fabricated raw terminal log"})
    source_identities = []
    for path in SOURCE_FILES:
        command = ["git", "show", f"{FROZEN}:{path}"]
        result = subprocess.run(command, cwd=args.source, capture_output=True)
        if result.returncode and path == "rust-toolchain.toml":
            continue
        assert result.returncode == 0, (path, result.stderr)
        source_identities.append({"path": path, "bytes": len(result.stdout), "sha256": digest(result.stdout)})
        if path == "experiments/typed-state-v1.json":
            assert digest(result.stdout) == execution["config_sha256"]
            write("config.json", result.stdout, f"git:{FROZEN}:{path}", "verbatim frozen recipe")
    checkpoints, prediction_files, condition_summaries, curve_rows = [], [], [], []
    for seed, policy, mode in CONDITIONS:
        name = f"{seed}-{policy}-{mode}"
        directory = args.study / name
        event = finished[f"{seed}:{policy}:{mode}"]
        log = args.study / (name + ".log")
        if log.exists():
            copy(log, "conditions/" + name + "/process.log", "formal condition stdout/stderr")
        receipt_path = directory / "receipt.json"
        if not receipt_path.exists():
            assert args.allow_incomplete
            condition_summaries.append({"seed": seed, "policy": policy, "mode": mode, "exit_code": event["exit_code"], "receipt_missing": True})
            continue
        receipt = load(receipt_path, checksum=True)
        assert receipt["source_revision"] == FROZEN and receipt["source_dirty"] is False
        assert receipt["config_sha256"] == execution["config_sha256"] and receipt["dataset_sha256"] == data_hashes
        assert receipt["selection"] == f"{seed}:{policy}:{mode}"
        assert len(receipt["runs"]) == 1
        run = receipt["runs"][0]
        assert (run["seed"], run["policy"], run["mode"]) == (seed, policy, mode)
        if not args.allow_incomplete:
            assert run["completed"] and run["serialized_replay"]["completed"] and run["updates"] == 5120 and run["epochs"] == 20
        # Keep original relative nesting so receipts can be rehydrated unchanged.
        relative_root = "conditions/" + name
        copy_checked(receipt_path, relative_root + "/receipt.json", "full formal receipt: all metrics/curves/telemetry/replay")
        for path in sorted(directory.glob("receipt-after-*.json")):
            copy_checked(path, relative_root + "/" + path.name, "immutable intermediate outer receipt")
        for path in [directory / "config.json", directory / "config.sha256", directory / "data" / "audit.json"]:
            if path.exists():
                copy(path, relative_root + "/" + str(path.relative_to(directory)), "condition input/audit provenance")
        model_dir = directory / name
        for item in ("run.json", "replay.json"):
            path = model_dir / item
            if path.exists():
                copy_checked(path, relative_root + "/" + name + "/" + item, "per-condition results or actual full-batch replay proof")
        for group in GROUPS[2:]:
            path = model_dir / (group + "-predictions.json")
            if not path.exists():
                assert args.allow_incomplete
                continue
            rows = load(path, checksum=True)
            assert len(rows) == 2048
            raw = path.read_bytes()
            target = relative_root + "/" + name + "/" + path.name + ".gz"
            stream = io.BytesIO()
            with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0, compresslevel=9) as compressed:
                compressed.write(raw)
            packed = stream.getvalue()
            assert gzip.decompress(packed) == raw
            write(target, packed, str(path), "lossless original per-world predictions; gzip mtime0")
            copy(path.with_suffix(".sha256"), str(pathlib.Path(target.removesuffix(".gz")).with_suffix(".sha256")), "original uncompressed prediction checksum")
            prediction_files.append({"condition": name, "group": group, "path": target, "uncompressed_bytes": len(raw),
                                     "uncompressed_sha256": digest(raw), "gzip_bytes": len(packed), "gzip_sha256": digest(packed), "queries": len(rows)})
        for checkpoint_name in ("step-200.json", "final.json"):
            path = model_dir / checkpoint_name
            if not path.exists():
                assert args.allow_incomplete
                continue
            cp = load(path, checksum=True)
            raw = path.read_bytes()
            checkpoints.append({"condition": name, "path_relative_to_study": str(path.relative_to(args.study)),
                                "local_path": str(path), "bytes": len(raw), "sha256": digest(raw),
                                "source_revision": cp["source_revision"], "config_sha256": cp["config_sha256"],
                                "dataset_sha256": cp["dataset_sha256"], "step": cp["step"], "epoch": cp["epoch"], "next_batch": cp["next_batch"],
                                "selection": cp["selection"], "parameter_group_lengths": [len(p["values"]) for p in cp["parameters"]],
                                "parameter_counters": [p["time"] for p in cp["parameters"]], "payload_committed": False})
        group_metrics = {key: value for key, value in run.get("groups", {}).items()}
        condition_summaries.append({"seed": seed, "policy": policy, "mode": mode, "completed": run["completed"], "failure": run.get("failure"),
                                    "exit_code": event["exit_code"], "updates": run["updates"], "epochs": run["epochs"],
                                    "experiment_elapsed_seconds": run["elapsed_seconds"], "outer_process_wall_seconds": event["process_wall_seconds"],
                                    "peak_rss_kib": run.get("peak_rss_kib"), "sampled_gpu_process_peak_mib": run.get("sampled_gpu_process_peak_mib"),
                                    "serialized_replay": run.get("serialized_replay"), "groups": group_metrics})
        for curve in run["curves"]:
            val = curve["validation"]["all"]
            curve_rows.append({"seed": seed, "policy": policy, "mode": mode, "epoch": curve["epoch"], "step": curve["step"],
                               "train_cross_entropy": curve["train_cross_entropy"], "validation_accuracy": val["accuracy"],
                               "validation_brier": val["brier"], "elapsed_seconds": curve["elapsed_seconds"]})
    write_json("checkpoint-inventory.json", {"schema": 1, "payloads_committed": False, "retention": "large checkpoint payloads remain in ignored local study directory; hashes and shape/cursor inventory are committed", "study_local_path": str(args.study), "files": checkpoints})
    write_json("prediction-inventory.json", {"schema": 1, "compression": "gzip compresslevel9, empty filename, mtime0; decompression restores exact original bytes", "files": prediction_files})
    write_json("summary.json", {"schema": 1, "source_revision": FROZEN, "source_dirty": False, "config_sha256": execution["config_sha256"],
                                "binary_sha256": BINARY_SHA, "dataset_sha256": data_hashes, "official_status": official["status"],
                                "gate_open": official["gate_open"], "conditions": condition_summaries,
                                "timing_boundaries": ["experiment elapsed excludes compilation and corpus audit/generation", "fresh process per condition; per-run RSS includes data/runtime and temporary replay model", "outer process wall includes corpus audit/generation", "host-read wait overlaps computation/transfers; not isolated PCIe latency"],
                                "limitations": ["structured synthetic worlds,16entities,fixed64-wide reader,three model seeds", "TF32 CUDA arithmetic; ideal-f64 drift preserved separately", "hypothetical packed export is nonresident; process memory is measured separately", "GPU process samples can miss transients; allocator payload breakdown is not full residency", "no measured energy or general kernel/throughput claim", "checkpoint payloads retained locally with hashes"]})
    buffer = io.StringIO(newline="")
    if curve_rows:
        writer = csv.DictWriter(buffer, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    write("learning-curves.csv", buffer.getvalue().encode(), role="all20epochs per complete condition; no checkpoint selection")
    write_json("provenance.json", {"schema": 1, "source_revision": FROZEN, "release_binary_sha256": BINARY_SHA,
                                   "config_sha256": execution["config_sha256"], "dataset_sha256": data_hashes,
                                   "source_files": source_identities, "bundler_python": sys.version, "bundler_platform": platform.platform(),
                                   "bundled_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                   "formal_execution": execution, "official_analysis_sha256": digest(args.analysis.read_bytes()),
                                   "original_failure_preservation": "all development files copied byte-for-byte; first ideal-f64 GPU failures not replaced"})
    copy(next(p for p in (script_root / "tergon-rehydrate-state-evidence.py",script_root / "rehydrate-evidence.py") if p.exists()), "scripts/rehydrate-evidence.py", "restore exact prediction bytes/receipt layout; optional local checkpoints")
    readme = next((p for p in (script_root / "tergon-typed-state-evidence-README.md",script_root.parent / "README.md") if p.exists()),pathlib.Path("/missing-readme"))
    if readme.exists():
        copy(readme, "README.md", "artifact map and reproduction boundaries")
    # This manifest excludes itself to avoid a self-hash cycle. The repository-wide
    # evidence inventory will hash this manifest as well as every staged artifact.
    write_json("bundle-inventory.json", {"schema": 1, "root_in_repository": "evidence/typed-state-v1", "source_revision": FROZEN,
                                         "files": list(sources), "inventory_self_excluded": True})
    print(json.dumps({"stage": str(output), "source_revision": FROZEN, "conditions": len(condition_summaries),
                      "prediction_files": len(prediction_files), "checkpoint_inventory_entries": len(checkpoints),
                      "bytes_staged": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
                      "official_status": official["status"], "gate_open": official["gate_open"]}, indent=2))


if __name__ == "__main__":
    main()
