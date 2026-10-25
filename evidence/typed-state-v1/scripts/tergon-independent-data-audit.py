"""Independent corpus review; reads frozen JSON, never invokes Rust helpers."""
import collections
import hashlib
import json
import pathlib
import sys

directory = pathlib.Path(sys.argv[1])
receipt = json.loads((directory / "audit.json").read_text())
heldout = {(0, 0, 0), (1, 2, 4)}
summary = []
all_world_ids = set()

def replay(example):
    records = {}
    last_time = 0
    for event in example["events"]:
        action = event["action"]
        assertion = event.get("assertion")
        time = assertion["time"] if assertion else event["time"]
        assert time > last_time
        last_time = time
        if action in ("Retract", "Replace"):
            assert records[event["target"]][1]
            records[event["target"]][1] = False
        if assertion:
            assert assertion["id"] not in records
            records[assertion["id"]] = [assertion, True]
    ordered = list(records.values())
    order = example["record_order"]
    assert sorted(order) == list(range(len(ordered)))
    return [ordered[index] for index in order]

def answer_and_primitives(example, records):
    current = example["start"]
    duplicate = competing_hypothesis = hypothesis_only = False
    for relation in example["relations"]:
        observed = [a["object"] for a, active in records
                    if active and a["observed"] and a["subject"] == current
                    and a["relation"] == relation]
        hypothetical = {a["object"] for a, active in records
                        if active and not a["observed"] and a["subject"] == current
                        and a["relation"] == relation}
        values = set(observed)
        duplicate |= len(observed) > len(values)
        competing_hypothesis |= bool(hypothetical - values)
        hypothesis_only |= not values and bool(hypothetical)
        if not values:
            return 16, (duplicate, competing_hypothesis, hypothesis_only)
        if len(values) > 1:
            return 17, (duplicate, competing_hypothesis, hypothesis_only)
        current = values.pop()
    return current, (duplicate, competing_hypothesis, hypothesis_only)

for group in receipt["groups"]:
    name = group["group"]
    raw = (directory / (name + ".json")).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == group["sha256"]
    assert len(raw) == group["bytes"]
    examples = json.loads(raw)
    worlds = collections.defaultdict(list)
    primitives = collections.Counter()
    paths = collections.Counter()
    for example in examples:
        assert example["group"] == name
        assert sorted(collections.Counter(example["types"]).values()) == [4, 4, 4, 4]
        records = replay(example)
        result, flags = answer_and_primitives(example, records)
        assert result == example["label"], (name, example["world"], example["branch"])
        for key, flag in zip(("agreeing_duplicate", "competing_hypothesis", "hypothesis_only_gap"), flags):
            primitives[key] += int(flag)
        primitives["update_context"] += any(e["action"] != "Assert" for e in example["events"])
        paths[tuple(example["relations"])] += 1
        if name in ("train", "validation"):
            assert tuple(example["relations"]) not in heldout
        if name == "unseen-composition":
            assert tuple(example["relations"]) in heldout
        nuisance = tuple((a["id"], a["relation"], a["source"], a["time"], active,
                          a["observed"], example["types"][a["subject"]],
                          example["types"][a["object"]]) for a, active in records)
        worlds[example["world"]].append((example, nuisance))
    for world, branches in worlds.items():
        assert world not in all_world_ids
        all_world_ids.add(world)
        assert len(branches) == 4
        assert len({nuisance for _, nuisance in branches}) == 1
        assert len({(e["start"], tuple(e["relations"]), tuple(e["types"])) for e, _ in branches}) == 1
        labels = collections.Counter("known" if e["label"] < 16 else e["label"] for e, _ in branches)
        assert labels == {"known": 2, 16: 1, 17: 1}
        assert len({e["label"] for e, _ in branches if e["label"] < 16}) == 2
        by_branch = {e["branch"]: e for e, _ in branches}
        assert set(by_branch) == {0, 1, 2, 3}
        def observed_degrees(example):
            return collections.Counter((a["subject"], a["relation"])
                                       for a, active in replay(example)
                                       if active and a["observed"])
        assert observed_degrees(by_branch[0]) == observed_degrees(by_branch[3])
    if name in ("train", "validation"):
        assert all(primitives[k] > 0 for k in ("agreeing_duplicate", "competing_hypothesis", "hypothesis_only_gap", "update_context"))
        assert {(0,), (0, 0), (1,), (1, 2), (2,), (2, 4), (4,)}.issubset(paths)
    assert len(worlds) == group["worlds"]
    assert len(examples) == group["queries"]
    summary.append({"group": name, "worlds": len(worlds), "queries": len(examples),
                    "primitive_occurrences": dict(primitives), "distinct_paths": len(paths)})
    del examples, raw, worlds

print(json.dumps({"method": "independent Python event replay and nuisance/split audit",
                  "source_revision": receipt["source_revision"],
                  "source_dirty": receipt.get("source_dirty", "unrecorded"),
                  "config_sha256": receipt.get("config_sha256"),
                  "audit_sha256": hashlib.sha256((directory / "audit.json").read_bytes()).hexdigest(),
                  "unique_worlds": len(all_world_ids), "groups": summary, "passed": True}, indent=2))
