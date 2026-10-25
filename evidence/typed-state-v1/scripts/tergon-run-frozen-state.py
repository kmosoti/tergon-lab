"""Execute the twelve frozen conditions in separate processes, retaining every result."""
import datetime
import hashlib
import itertools
import json
import os
import pathlib
import subprocess
import sys
import time

source, binary, destination, expected_revision = sys.argv[1:]
source, binary, destination = map(pathlib.Path, (source, binary, destination))
config = source / "experiments/typed-state-v1.json"
binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
config_hash = hashlib.sha256(config.read_bytes()).hexdigest()

def git(*args):
    return subprocess.check_output(["git", *args], cwd=source, text=True).strip()

def identity():
    assert git("rev-parse", "HEAD") == expected_revision, "source changed"
    assert not git("status", "--porcelain"), "source became dirty"
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == binary_hash, "binary changed"
    assert hashlib.sha256(config.read_bytes()).hexdigest() == config_hash, "recipe changed"

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

identity()
destination.mkdir()
metadata = {"source_revision": expected_revision, "binary_sha256": binary_hash,
            "config_sha256": config_hash, "started_utc": utc(),
            "execution": "one fresh process per condition; sequential on cuda:0",
            "runner_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
            "process_wall_boundary": "includes dataset audit/generation; experiment elapsed in each receipt excludes them"}
(destination / "execution.json").write_text(json.dumps(metadata, indent=2) + "\n")
with (destination / "progress.jsonl").open("x") as events:
    for seed, policy, mode in itertools.product((17, 29, 43), ("metadata-only", "enforced"), ("float", "ternary")):
        identity()
        selection = f"{seed}:{policy}:{mode}"
        name = selection.replace(":", "-")
        event = {"condition": selection, "started_utc": utc()}
        print(json.dumps(event), flush=True)
        events.write(json.dumps(event) + "\n")
        events.flush()
        started = time.monotonic()
        with (destination / (name + ".log")).open("x") as log:
            result = subprocess.run([str(binary), "state-train", str(config), str(destination / name)],
                                    cwd=source, env={**os.environ, "TERGON_STATE_RUN": selection},
                                    stdout=log, stderr=subprocess.STDOUT)
        event = {"condition": selection, "finished_utc": utc(), "exit_code": result.returncode,
                 "process_wall_seconds": time.monotonic() - started}
        receipt = destination / name / "receipt.json"
        if receipt.exists():
            event["receipt_sha256"] = hashlib.sha256(receipt.read_bytes()).hexdigest()
            event["runs"] = [{key: run.get(key) for key in ("completed", "updates", "epochs", "elapsed_seconds", "failure")}
                             for run in json.loads(receipt.read_text())["runs"]]
        print(json.dumps(event), flush=True)
        events.write(json.dumps(event) + "\n")
        events.flush()
    identity()
    print(json.dumps({"study_finished_utc": utc()}), flush=True)
