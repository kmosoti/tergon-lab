"""Independent recomputation from immutable predictions; no Rust analysis helpers."""
import collections
import hashlib
import itertools
import json
import math
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
official_path = pathlib.Path(sys.argv[2])
groups = ["ordinary", "unseen-composition", "replace-retract", "conflicts-hypotheses", "distractors"]
seeds = [17, 29, 43]

def checked(path):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == path.with_suffix(".sha256").read_text().strip(), str(path)
    return json.loads(raw)

conditions, identities, summaries = {}, set(), []
for seed, policy, mode in itertools.product(seeds, ["metadata-only", "enforced"], ["float", "ternary"]):
    name = f"{seed}-{policy}-{mode}"
    directory = root / name
    receipt = checked(directory / "receipt.json")
    identities.add((receipt["source_revision"], receipt["config_sha256"], tuple(receipt["dataset_sha256"])))
    assert receipt["source_dirty"] is False
    assert len(receipt["runs"]) == 1
    run = receipt["runs"][0]
    assert (run["seed"], run["policy"], run["mode"]) == (seed, policy, mode)
    assert run["completed"] and run["updates"] == 5120 and run["epochs"] == 20
    assert 0 < run["elapsed_seconds"] <= 600
    replay = run["serialized_replay"]
    assert replay["completed"] and replay["at_step"] == 200 and replay["updates"] == 50
    assert replay["batch_size"] == 128 and replay["last_step"] == 250
    assert replay["indices_cursors_and_labels_match"]
    model_dir = directory / name
    cp200 = checked(model_dir / "step-200.json")
    final = checked(model_dir / "final.json")
    assert cp200["step"] == 200 and cp200["next_batch"] == 25600
    assert final["step"] == 5120 and final["epoch"] == 20 and final["next_batch"] == 0
    assert all(p["time"] == 5120 for p in final["parameters"])
    assert replay["checkpoint_sha256"] == hashlib.sha256((model_dir / "step-200.json").read_bytes()).hexdigest()
    assert run["checkpoint_sha256"] == hashlib.sha256((model_dir / "final.json").read_bytes()).hexdigest()
    worlds_by_group, group_metrics = {}, {}
    for group in groups:
        rows = checked(model_dir / (group + "-predictions.json"))
        assert len(rows) == 2048
        worlds = collections.defaultdict(dict)
        correct, briers = [], []
        for row in rows:
            assert row["group"] == group
            assert row["branch"] not in worlds[row["world"]]
            p = row["probabilities"]
            assert len(p) == 18 and all(math.isfinite(x) and 0 <= x <= 1 for x in p)
            assert abs(sum(p) - 1) < 1e-10
            assert row["predicted"] == max(range(18), key=p.__getitem__)
            brier = sum((x - (i == row["label"])) ** 2 for i, x in enumerate(p))
            assert abs(brier - row["brier"]) < 1e-10
            correct.append(row["predicted"] == row["label"])
            briers.append(brier)
            worlds[row["world"]][row["branch"]] = row
        assert len(worlds) == 512
        assert all(set(branches) == {0, 1, 2, 3} for branches in worlds.values())
        accuracy = sum(correct) / len(correct)
        brier = sum(briers) / len(briers)
        assert accuracy == run["groups"][group]["all"]["accuracy"]
        assert abs(brier - run["groups"][group]["all"]["brier"]) < 1e-12
        group_metrics[group] = {"accuracy": accuracy, "brier": brier}
        worlds_by_group[group] = worlds
    conditions[seed, policy, mode] = (worlds_by_group, group_metrics)
    summaries.append({"seed": seed, "policy": policy, "mode": mode, "groups": group_metrics,
                      "elapsed_seconds": run["elapsed_seconds"], "replay_max_parameter_error": replay["max_parameter_moment_absolute_error"]})
assert len(identities) == 1

reference = conditions[17, "metadata-only", "float"][0]
for worlds_by_group, _ in conditions.values():
    for group in groups:
        assert set(worlds_by_group[group]) == set(reference[group])
        for world, branches in worlds_by_group[group].items():
            assert {b: (r["label"], r["hops"]) for b, r in branches.items()} == {b: (r["label"], r["hops"]) for b, r in reference[group][world].items()}

seed_results = []
for seed in seeds:
    float_metrics = conditions[seed, "metadata-only", "float"][1]
    metadata = conditions[seed, "metadata-only", "ternary"][1]
    enforced = conditions[seed, "enforced", "ternary"][1]
    stress_m = sum(metadata[g]["accuracy"] for g in groups[1:]) / 4
    stress_e = sum(enforced[g]["accuracy"] for g in groups[1:]) / 4
    item = {"seed": seed, "metadata_float_ordinary": float_metrics["ordinary"]["accuracy"],
            "enforced_ternary_ordinary": enforced["ordinary"]["accuracy"],
            "enforced_ternary_stress": stress_e, "metadata_ternary_stress": stress_m,
            "stress_gain": stress_e - stress_m,
            "ordinary_delta": enforced["ordinary"]["accuracy"] - metadata["ordinary"]["accuracy"],
            "five_group_brier_delta": sum(enforced[g]["brier"] - metadata[g]["brier"] for g in groups) / 5}
    item["per_seed_gate_pass"] = (item["metadata_float_ordinary"] >= .9 and item["enforced_ternary_ordinary"] >= .9
                                   and stress_e >= .85 and item["stress_gain"] > 0
                                   and item["ordinary_delta"] >= -.02 and item["five_group_brier_delta"] <= .02)
    seed_results.append(item)

strata = []
for group in groups[1:]:
    stratum = []
    for world in sorted(reference[group]):
        seed_deltas = []
        for seed in seeds:
            m = conditions[seed, "metadata-only", "ternary"][0][group][world]
            e = conditions[seed, "enforced", "ternary"][0][group][world]
            seed_deltas.append(sum(int(e[b]["predicted"] == e[b]["label"]) - int(m[b]["predicted"] == m[b]["label"]) for b in range(4)) / 4)
        stratum.append(sum(d / 3 for d in seed_deltas))
    strata.append(stratum)

mask = (1 << 64) - 1
rng = 20261025
bootstrap = []
for _ in range(10000):
    total = 0.0
    for stratum in strata:
        for _ in range(512):
            rng = (rng + 0x9e3779b97f4a7c15) & mask
            x = ((rng ^ (rng >> 30)) * 0xbf58476d1ce4e5b9) & mask
            x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & mask
            x ^= x >> 31
            total += stratum[x % 512]
    bootstrap.append(total / 2048)
bootstrap.sort()

def quantile(p):
    at = 9999 * p
    low = math.floor(at)
    high = math.ceil(at)
    return bootstrap[low] + (bootstrap[high] - bootstrap[low]) * (at - low)

interval = [quantile(.025), quantile(.975)]
gains = [s["stress_gain"] for s in seed_results]
mean_gain = statistics.mean(gains)
control_pass = all(s["metadata_float_ordinary"] >= .9 for s in seed_results)
gate = all(s["per_seed_gate_pass"] for s in seed_results) and mean_gain >= .05 and interval[0] > 0
official = checked(official_path)
assert official["gate_open"] == gate
assert official["control_pass"] == control_pass
for expected, actual in zip(seed_results, official["seed_results"]):
    for key, value in expected.items():
        assert abs(value - actual[key]) < 1e-12, (key, value, actual[key])
assert abs(mean_gain - official["mean_ternary_stress_gain"]) < 1e-12
assert abs(statistics.stdev(gains) - official["seed_sample_standard_deviation"]) < 1e-12
assert all(abs(a - b) < 1e-12 for a, b in zip(interval, official["paired_world_bootstrap"]["two_sided_95_percentile_interval"]))
print(json.dumps({"method": "independent Python prediction/checkpoint validation and keyed whole-world bootstrap",
                  "passed": True, "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                  "official_analysis_sha256": hashlib.sha256(official_path.read_bytes()).hexdigest(),
                  "source_revision": next(iter(identities))[0], "control_pass": control_pass, "gate_open": gate,
                  "seed_results": seed_results, "mean_ternary_stress_gain": mean_gain,
                  "paired_95_interval": interval, "seed_sample_standard_deviation": statistics.stdev(gains),
                  "conditions": summaries}, indent=2))
