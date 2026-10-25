#!/usr/bin/env python3
"""Restore exact committed receipts/predictions into a new scratch study directory.

Checkpoint payloads are optional, remain local, and are accepted only if their
bytes and SHA256 match the committed checkpoint inventory. No training is run.
"""
import argparse
import gzip
import hashlib
import json
import pathlib
import shutil


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=pathlib.Path)
    parser.add_argument("new_directory", type=pathlib.Path)
    parser.add_argument("--checkpoints", type=pathlib.Path, help="Original ignored formal-study directory; copy verified checkpoint payloads")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    destination = args.new_directory.resolve()
    assert not destination.exists(), "never replace a prior reconstruction"
    assert not destination.is_relative_to(bundle), "reconstruct outside the committed bundle"
    inventory = json.loads((bundle / "bundle-inventory.json").read_text())
    for item in inventory["files"]:
        path = bundle / item["path"]
        assert path.is_relative_to(bundle) and not path.is_symlink()
        raw = path.read_bytes()
        assert len(raw) == item["bytes"] and sha(raw) == item["sha256"], str(path)
    destination.mkdir()
    for path in sorted((bundle / "conditions").rglob("*")):
        assert not path.is_symlink()
        if path.is_file() and path.suffix != ".gz":
            target = destination / path.relative_to(bundle / "conditions")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                output.write(path.read_bytes())
    predictions = json.loads((bundle / "prediction-inventory.json").read_text())
    for item in predictions["files"]:
        packed = (bundle / item["path"]).read_bytes()
        assert len(packed) == item["gzip_bytes"] and sha(packed) == item["gzip_sha256"]
        raw = gzip.decompress(packed)
        assert len(raw) == item["uncompressed_bytes"] and sha(raw) == item["uncompressed_sha256"]
        relative = pathlib.Path(item["path"]).relative_to("conditions")
        target = destination / str(relative).removesuffix(".gz")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output:
            output.write(raw)
        assert target.with_suffix(".sha256").read_text().strip() == sha(raw)
    for name in ("analysis.json", "analysis.sha256", "execution.json", "progress.jsonl"):
        with (destination / name).open("xb") as output:
            output.write((bundle / name).read_bytes())
    copied_checkpoints = 0
    if args.checkpoints:
        inventory = json.loads((bundle / "checkpoint-inventory.json").read_text())
        for item in inventory["files"]:
            relative = pathlib.Path(item["path_relative_to_study"])
            assert not relative.is_absolute() and ".." not in relative.parts
            raw = (args.checkpoints / relative).read_bytes()
            assert len(raw) == item["bytes"] and sha(raw) == item["sha256"], str(relative)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                output.write(raw)
            with target.with_suffix(".sha256").open("x") as checksum:
                checksum.write(item["sha256"] + "\n")
            copied_checkpoints += 1
    print(json.dumps({"destination": str(destination), "prediction_files": len(predictions["files"]),
                      "checkpoint_payloads_restored": copied_checkpoints,
                      "numerical_aggregate_recomputable": True,
                      "full_independent_checkpoint_audit_requires_local_payloads": not bool(args.checkpoints)}, indent=2))


if __name__ == "__main__":
    main()
