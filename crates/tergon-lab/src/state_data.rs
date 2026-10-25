//! Frozen world generator and independently implemented event-log oracle.
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use tergon_engine::state::{Assertion, EntityType, Event, Evidence, Query, Relation, State};

pub(crate) const TYPES: [EntityType; 4] = [
    EntityType::Person,
    EntityType::Item,
    EntityType::Container,
    EntityType::Place,
];
pub(crate) const GROUPS: [&str; 7] = [
    "train",
    "validation",
    "ordinary",
    "unseen-composition",
    "replace-retract",
    "conflicts-hypotheses",
    "distractors",
];
/// SplitMix64; public-domain Vigna reference, rejection-sampled bounded draws.
pub(crate) struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn bounded(&mut self, n: usize) -> usize {
        let n = n as u64;
        let t = n.wrapping_neg() % n;
        loop {
            let x = self.next();
            if x >= t {
                return (x % n) as usize;
            }
        }
    }
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            x.swap(i, self.bounded(i + 1));
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SavedAssertion {
    pub id: u32,
    pub subject: usize,
    pub relation: usize,
    pub object: usize,
    pub source: usize,
    pub time: u32,
    pub observed: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "action", deny_unknown_fields)]
pub(crate) enum SavedEvent {
    Assert {
        assertion: SavedAssertion,
    },
    Retract {
        target: u32,
        time: u32,
    },
    Replace {
        target: u32,
        assertion: SavedAssertion,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Example {
    pub world: u64,
    pub branch: usize,
    pub group: String,
    pub types: [usize; 16],
    pub events: Vec<SavedEvent>,
    pub record_order: Vec<usize>,
    pub start: usize,
    pub relations: Vec<usize>,
    pub label: usize,
}
impl SavedAssertion {
    fn engine(&self) -> Assertion {
        Assertion {
            id: self.id,
            subject: self.subject,
            relation: Relation::ALL[self.relation],
            object: self.object,
            source: self.source,
            time: self.time,
            evidence: if self.observed {
                Evidence::Observed
            } else {
                Evidence::Hypothesis
            },
        }
    }
}
impl Example {
    pub fn materialize(&self) -> (State, Query) {
        let events: Vec<_> = self
            .events
            .iter()
            .map(|e| match e {
                SavedEvent::Assert { assertion } => Event::Assert(assertion.engine()),
                SavedEvent::Retract { target, time } => Event::Retract {
                    target: *target,
                    time: *time,
                },
                SavedEvent::Replace { target, assertion } => Event::Replace {
                    target: *target,
                    assertion: assertion.engine(),
                },
            })
            .collect();
        let mut state = State::reduce(self.types.map(|t| TYPES[t]), &events)
            .expect("generator emits valid logs");
        let records = state.records.clone();
        state.records = self
            .record_order
            .iter()
            .map(|i| records[*i].clone())
            .collect();
        (
            state,
            Query {
                start: self.start,
                relations: self.relations.iter().map(|r| Relation::ALL[*r]).collect(),
            },
        )
    }
    /// Independent reference: replay a map of live raw log assertions, then set traversal.
    /// Does not call the engine reducer, eligibility rule, relation signatures or symbolic baseline.
    pub fn oracle(&self) -> usize {
        let mut live = BTreeMap::new();
        for event in &self.events {
            match event {
                SavedEvent::Assert { assertion } => {
                    live.insert(assertion.id, assertion);
                }
                SavedEvent::Retract { target, .. } => {
                    assert!(live.remove(target).is_some());
                }
                SavedEvent::Replace { target, assertion } => {
                    assert!(live.remove(target).is_some());
                    live.insert(assertion.id, assertion);
                }
            }
        }
        let mut entity = self.start;
        for relation in &self.relations {
            let destinations: BTreeSet<_> = live
                .values()
                .filter(|a| a.observed && a.subject == entity && a.relation == *relation)
                .map(|a| a.object)
                .collect();
            if destinations.is_empty() {
                return 16;
            }
            if destinations.len() > 1 {
                return 17;
            }
            entity = *destinations.first().unwrap();
        }
        entity
    }
    pub fn renamed(&self, permutation: &[usize; 16]) -> Self {
        let mut result = self.clone();
        for (old, new) in permutation.iter().enumerate() {
            result.types[*new] = self.types[old];
        }
        result.start = permutation[self.start];
        if result.label < 16 {
            result.label = permutation[result.label];
        }
        for event in &mut result.events {
            if let SavedEvent::Assert { assertion } | SavedEvent::Replace { assertion, .. } = event
            {
                assertion.subject = permutation[assertion.subject];
                assertion.object = permutation[assertion.object];
            }
        }
        result
    }
}
fn heldout(chain: &[usize]) -> bool {
    chain == [0, 0, 0] || chain == [1, 2, 4]
}
fn different(rng: &mut Rng, types: &[usize; 16], entity: usize) -> usize {
    let choices: Vec<_> = (0..16)
        .filter(|id| *id != entity && types[*id] == types[entity])
        .collect();
    choices[rng.bounded(choices.len())]
}
fn assertion(
    rng: &mut Rng,
    id: u32,
    subject: usize,
    relation: usize,
    object: usize,
    observed: bool,
    time: u32,
) -> SavedAssertion {
    SavedAssertion {
        id,
        subject,
        relation,
        object,
        source: rng.bounded(4),
        time,
        observed,
    }
}
/// Four related counterfactual branches, clustered by world. No answer-derived model features.
pub(crate) fn world(group: usize, index: usize) -> Vec<Example> {
    let world = ((group as u64) << 48) | index as u64;
    let mut rng = Rng(0x5354415445310000 ^ world.wrapping_mul(0xd1342543de82ef95));
    let mut permutation = std::array::from_fn::<_, 16, _>(|i| i);
    rng.shuffle(&mut permutation);
    let mut types = [0; 16];
    for i in 0..16 {
        types[permutation[i]] = i / 4;
    }
    let mut attempts = 0;
    let (nodes, relations) = loop {
        attempts += 1;
        assert!(
            attempts < 100_000,
            "bounded chain-generation rejection exceeded"
        );
        let hops = if group == 3 { 3 } else { 1 + rng.bounded(3) };
        let mut nodes = vec![rng.bounded(16)];
        let mut chain = Vec::new();
        for _ in 0..hops {
            let candidates: Vec<_> = Relation::ALL
                .iter()
                .enumerate()
                .flat_map(|(r, rel)| {
                    (0..16).filter_map({
                        let nodes = &nodes;
                        move |dst| {
                            (rel.accepts(TYPES[types[*nodes.last().unwrap()]], TYPES[types[dst]])
                                && !nodes.contains(&dst))
                            .then_some((r, dst))
                        }
                    })
                })
                .collect();
            if candidates.is_empty() {
                break;
            }
            let (r, dst) = candidates[rng.bounded(candidates.len())];
            chain.push(r);
            nodes.push(dst);
        }
        if chain.len() == hops && heldout(&chain) == (group == 3) {
            break (nodes, chain);
        }
    };
    let critical = rng.bounded(relations.len());
    let subject = nodes[critical];
    let relation = relations[critical];
    let object = nodes[critical + 1];
    let other_subjects: Vec<_> = (0..16)
        .filter(|id| {
            types[*id] == types[subject]
                && !relations
                    .iter()
                    .enumerate()
                    .any(|(i, r)| *r == relation && nodes[i] == *id)
        })
        .collect();
    let other_subject = other_subjects[rng.bounded(other_subjects.len())];
    let other_object = different(&mut rng, &types, object);
    // All branches have identical record budgets and counts of observed/hypothetical/inactive records.
    let mut count = if group == 6 {
        24 + rng.bounded(9)
    } else {
        8 + rng.bounded(9)
    };
    let history = group == 4 || rng.bounded(2) == 0;
    let nuisance = relations.len();
    let duplicate = nuisance + 1;
    let mut update_targets = vec![critical, relations.len() - 1, nuisance, duplicate];
    update_targets.sort_unstable();
    update_targets.dedup();
    rng.shuffle(&mut update_targets);
    count = count.max(relations.len() + 3 + if history { update_targets.len() } else { 0 });
    let base_count = count - if history { update_targets.len() } else { 0 };
    let mut base = Vec::new();
    for (i, r) in relations.iter().enumerate() {
        base.push(assertion(
            &mut rng,
            (base.len() + 1) as u32,
            nodes[i],
            *r,
            nodes[i + 1],
            true,
            (base.len() + 1) as u32,
        ));
    }
    base.push(assertion(
        &mut rng,
        (nuisance + 1) as u32,
        other_subject,
        relation,
        other_object,
        true,
        (nuisance + 1) as u32,
    ));
    // Agreeing duplicate support is preserved as a distinct observed assertion.
    base.push(assertion(
        &mut rng,
        (base.len() + 1) as u32,
        subject,
        relation,
        object,
        true,
        (base.len() + 1) as u32,
    ));
    // Incorrect hypotheses challenge the queried operation in every split, including
    // hypothesis-only support at the Unknown branch's missing key.
    base.push(assertion(
        &mut rng,
        (base.len() + 1) as u32,
        subject,
        relation,
        other_object,
        false,
        (base.len() + 1) as u32,
    ));
    if group == 5 {
        for (i, r) in relations.iter().enumerate() {
            if base.len() == base_count {
                break;
            }
            let destination = different(&mut rng, &types, nodes[i + 1]);
            base.push(assertion(
                &mut rng,
                (base.len() + 1) as u32,
                nodes[i],
                *r,
                destination,
                false,
                (base.len() + 1) as u32,
            ));
        }
    }
    let mut attempts = 0;
    while base.len() < base_count {
        attempts += 1;
        assert!(
            attempts < 100_000,
            "bounded distractor-generation rejection exceeded"
        );
        let s = rng.bounded(16);
        let r = rng.bounded(5);
        let o = rng.bounded(16);
        if !Relation::ALL[r].accepts(TYPES[types[s]], TYPES[types[o]])
            || relations
                .iter()
                .enumerate()
                .any(|(i, rel)| nodes[i] == s && *rel == r)
        {
            continue;
        }
        let observed = if group == 5 {
            rng.bounded(2) == 0
        } else {
            rng.bounded(4) != 0
        };
        base.push(assertion(
            &mut rng,
            (base.len() + 1) as u32,
            s,
            r,
            o,
            observed,
            (base.len() + 1) as u32,
        ));
    }
    let mut chronology: Vec<_> = (0..base_count).collect();
    rng.shuffle(&mut chronology);
    let alternate_final = different(&mut rng, &types, *nodes.last().unwrap());
    let mut ids: Vec<_> = (1..=count as u32).collect();
    rng.shuffle(&mut ids);
    let mut order: Vec<_> = (0..count).collect();
    rng.shuffle(&mut order);
    let mut branches = Vec::new();
    for branch in 0..4 {
        let mut assertions = base.clone();
        match branch {
            0 => {}
            1 => assertions[relations.len() - 1].object = alternate_final,
            2 => assertions[critical].subject = other_subject,
            3 => assertions[duplicate].object = other_object,
            _ => unreachable!(),
        }
        if branch == 1 && critical == relations.len() - 1 {
            assertions[duplicate].object = alternate_final;
        }
        if branch == 2 {
            assertions[duplicate].subject = other_subject;
        }
        let initial = if history { &base } else { &assertions };
        let mut events: Vec<_> = chronology
            .iter()
            .enumerate()
            .map(|(time, i)| {
                let mut assertion = initial[*i].clone();
                assertion.time = (time + 1) as u32;
                SavedEvent::Assert { assertion }
            })
            .collect();
        if history {
            for (j, target) in update_targets.iter().enumerate() {
                let mut changed = assertions[*target].clone();
                changed.id = (base_count + j + 1) as u32;
                changed.time = (base_count + 2 * (j + 1)) as u32;
                if branch == 2 && (*target == critical || *target == duplicate) {
                    events.push(SavedEvent::Retract {
                        target: (target + 1) as u32,
                        time: changed.time - 1,
                    });
                    events.push(SavedEvent::Assert { assertion: changed });
                } else {
                    events.push(SavedEvent::Replace {
                        target: (target + 1) as u32,
                        assertion: changed,
                    });
                }
            }
        }
        // Shared random identity and slot permutations preserve nuisance equality across branches.
        for event in &mut events {
            match event {
                SavedEvent::Assert { assertion } => assertion.id = ids[assertion.id as usize - 1],
                SavedEvent::Replace { target, assertion } => {
                    *target = ids[*target as usize - 1];
                    assertion.id = ids[assertion.id as usize - 1];
                }
                SavedEvent::Retract { target, .. } => *target = ids[*target as usize - 1],
            }
        }
        let mut example = Example {
            world,
            branch,
            group: GROUPS[group].into(),
            types,
            events,
            record_order: order.clone(),
            start: nodes[0],
            relations: relations.clone(),
            label: 0,
        };
        example.label = example.oracle();
        let expected = match branch {
            0 | 1 => true,
            2 => example.label == 16,
            3 => example.label == 17,
            _ => false,
        };
        assert!(
            expected && (branch >= 2 || example.label < 16),
            "generator construction failed {world}/{branch}: {}",
            example.label
        );
        assert_eq!(
            example
                .materialize()
                .0
                .answer(&example.materialize().1)
                .unwrap()
                .class(),
            example.label
        );
        branches.push(example);
    }
    rng.shuffle(&mut branches);
    branches
}
pub(crate) fn dataset(group: usize, worlds: usize) -> Vec<Example> {
    (0..worlds).flat_map(|i| world(group, i)).collect()
}

fn assert_nuisance_equal(examples: &[Example]) {
    let signature = |example: &Example| {
        let (state, _) = example.materialize();
        state
            .records
            .iter()
            .map(|r| {
                let a = &r.assertion;
                (
                    a.id,
                    a.relation.index(),
                    a.source,
                    a.time,
                    r.active,
                    a.evidence == Evidence::Observed,
                    state.types[a.subject].index(),
                    state.types[a.object].index(),
                )
            })
            .collect::<Vec<_>>()
    };
    let known = examples
        .iter()
        .find(|e| e.branch == 0)
        .unwrap()
        .materialize()
        .0;
    let conflict = examples
        .iter()
        .find(|e| e.branch == 3)
        .unwrap()
        .materialize()
        .0;
    let degree = |state: &State| {
        let mut counts = BTreeMap::new();
        for r in &state.records {
            if r.active && r.assertion.evidence == Evidence::Observed {
                *counts
                    .entry((r.assertion.subject, r.assertion.relation.index()))
                    .or_insert(0) += 1;
            }
        }
        counts
    };
    assert_eq!(
        degree(&known),
        degree(&conflict),
        "known/conflict cardinality shortcut"
    );
    let expected = signature(&examples[0]);
    for e in &examples[1..] {
        assert_eq!(
            signature(e),
            expected,
            "nuisance metadata differ by branch in world {}",
            e.world
        );
    }
}

/// Generate, independently audit and persist the frozen corpus before any training.
pub(crate) fn audit(directory: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    use std::{fs, io::Write, time::Instant};
    fs::create_dir(directory)?;
    let mut receipt = Vec::new();
    let mut seed = Rng(20261025);
    for (group, name) in GROUPS.iter().enumerate() {
        let worlds = match group {
            0 => 8192,
            1 => 1024,
            _ => 512,
        };
        let started = Instant::now();
        let data = dataset(group, worlds);
        let generation_seconds = started.elapsed().as_secs_f64();
        for cluster in data.as_chunks::<4>().0 {
            assert_nuisance_equal(cluster);
        }
        let mut counts = [[0usize; 3]; 4];
        let mut inactive_contexts = 0usize;
        let mut agreeing_duplicate_contexts = 0usize;
        let mut query_related_hypothesis_contexts = 0usize;
        let mut retraction_contexts = 0usize;
        let mut metadata = serde_json::Map::new();
        let started = Instant::now();
        for example in &data {
            let (state, query) = example.materialize();
            assert_eq!(state.answer(&query)?.class(), example.oracle());
            inactive_contexts += usize::from(state.records.iter().any(|r| !r.active));
            retraction_contexts += usize::from(
                example
                    .events
                    .iter()
                    .any(|e| matches!(e, SavedEvent::Retract { .. })),
            );
            agreeing_duplicate_contexts +=
                usize::from(state.records.iter().enumerate().any(|(i, r)| {
                    r.active
                        && r.assertion.evidence == Evidence::Observed
                        && state.records[..i].iter().any(|s| {
                            s.active
                                && s.assertion.evidence == Evidence::Observed
                                && (
                                    r.assertion.subject,
                                    r.assertion.relation,
                                    r.assertion.object,
                                ) == (
                                    s.assertion.subject,
                                    s.assertion.relation,
                                    s.assertion.object,
                                )
                        })
                }));
            query_related_hypothesis_contexts += usize::from(state.records.iter().any(|r| {
                r.active
                    && r.assertion.evidence == Evidence::Hypothesis
                    && query.relations.contains(&r.assertion.relation)
            }));
            counts[query.relations.len()][if example.label < 16 {
                0
            } else {
                example.label - 15
            }] += 1;
            let mut p = std::array::from_fn(|i| i);
            seed.shuffle(&mut p);
            let renamed = example.renamed(&p);
            assert_eq!(renamed.oracle(), renamed.label);
            let mut records = state.records.clone();
            records.reverse();
            assert_eq!(
                State {
                    types: state.types,
                    records
                }
                .answer(&query)?
                .class(),
                example.label
            );
            // A label-conditioned audit exposes nuisance/status shortcuts before training.
            let outcome = if example.label < 16 {
                "known"
            } else if example.label == 16 {
                "unknown"
            } else {
                "conflict"
            };
            for relation in 0..5 {
                let active = state
                    .records
                    .iter()
                    .filter(|r| {
                        r.active
                            && r.assertion.evidence == Evidence::Observed
                            && r.assertion.relation.index() == relation
                    })
                    .count();
                let key = format!("{outcome}/active-observed/relation-{relation}/count-{active}");
                let value = metadata.entry(key).or_insert(serde_json::json!(0));
                *value = serde_json::json!(value.as_u64().unwrap() + 1);
            }
        }
        let symbolic_seconds = started.elapsed().as_secs_f64();
        let contexts: Vec<_> = data.iter().map(Example::materialize).collect();
        let state_record_allocation_bytes = contexts
            .iter()
            .map(|(state, _)| {
                state.records.capacity() * std::mem::size_of::<tergon_engine::state::Record>()
            })
            .sum::<usize>();
        let query_relation_allocation_bytes = contexts
            .iter()
            .map(|(_, query)| query.relations.capacity() * std::mem::size_of::<Relation>())
            .sum::<usize>();
        let inline_context_bytes = contexts.capacity() * std::mem::size_of::<(State, Query)>();
        let symbolic_started = Instant::now();
        let mut checksum = 0usize;
        for (state, query) in &contexts {
            checksum = checksum.wrapping_add(std::hint::black_box(state.answer(query)?.class()));
        }
        let isolated_symbolic_seconds = symbolic_started.elapsed().as_secs_f64();
        assert_eq!(checksum, data.iter().map(|e| e.label).sum::<usize>());

        let bytes = serde_json::to_vec(&data)?;
        let filename = format!("{name}.json");
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join(&filename))?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        receipt.push(serde_json::json!({"group":name,"worlds":worlds,"queries":data.len(),"sha256":super::hash(&bytes),"bytes":bytes.len(),"outcome_by_hop":counts,"symbolic_accuracy":1.0,"exact_nuisance_equality_worlds":worlds,"known_conflict_equal_degree_worlds":worlds,"primitive_coverage":{"inactive_contexts":inactive_contexts,"explicit_retraction_contexts":retraction_contexts,"agreeing_duplicate_contexts":agreeing_duplicate_contexts,"query_relation_hypothesis_contexts":query_related_hypothesis_contexts},"generation_seconds":generation_seconds,"symbolic_and_metamorphic_audit_seconds":symbolic_seconds,"isolated_symbolic_query_seconds":isolated_symbolic_seconds,"isolated_query_boundary":"validated-context answer API includes its validation; excludes event reduction, generation, serialization and metamorphic checks","symbolic_live_storage":{"inline_state_query_bytes":inline_context_bytes,"record_capacity_bytes":state_record_allocation_bytes,"query_relation_capacity_bytes":query_relation_allocation_bytes,"excludes":"allocator bookkeeping, event logs, and report/dataset storage"},"active_observed_histogram":metadata}));
    }
    let bytes = serde_json::to_vec_pretty(
        &serde_json::json!({"schema":1,"source_revision":super::command("git",&["rev-parse","HEAD"])?,"source_dirty":!super::command("git",&["status","--porcelain"])?.is_empty(),"config_sha256":super::hash(include_bytes!("../../../experiments/typed-state-v1.json")),"generator_sha256":super::hash(include_bytes!("state_data.rs")),"groups":receipt,"limitations":["combined audit timing includes validation/materialization/renaming/reversal; isolated query timing reported separately","serialized JSON bytes are data storage, not tensor or allocator memory"]}),
    )?;
    fs::write(directory.join("audit.json"), bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generator_semantics_and_cluster_balance() {
        let mut seen = BTreeSet::new();
        for group in 0..7 {
            for i in 0..128 {
                let examples = world(group, i);
                assert_nuisance_equal(&examples);
                assert!(seen.insert(examples[0].world));
                let mut outcomes = [0; 3];
                let n = examples[0].record_order.len();
                for example in examples {
                    assert_eq!(example.record_order.len(), n);
                    assert!(n >= 8 && n <= if group == 6 { 32 } else { 16 });
                    outcomes[if example.label < 16 {
                        0
                    } else {
                        example.label - 15
                    }] += 1;
                    let (mut state, query) = example.materialize();
                    assert_eq!(state.answer(&query).unwrap().class(), example.oracle());
                    state.records.reverse();
                    assert_eq!(state.answer(&query).unwrap().class(), example.label);
                    assert_eq!(heldout(&example.relations), group == 3);
                    let mut rng = Rng(i as u64 + 9);
                    let mut p = std::array::from_fn(|i| i);
                    rng.shuffle(&mut p);
                    let renamed = example.renamed(&p);
                    assert_eq!(
                        renamed.oracle(),
                        if example.label < 16 {
                            p[example.label]
                        } else {
                            example.label
                        }
                    );
                }
                assert_eq!(outcomes, [2, 1, 1]);
            }
        }
    }
    #[test]
    fn known_counterfactual_changes_answer() {
        for group in 0..7 {
            for i in 0..30 {
                let e = world(group, i);
                let a = e.iter().find(|e| e.branch == 0).unwrap();
                let b = e.iter().find(|e| e.branch == 1).unwrap();
                assert_ne!(a.label, b.label);
            }
        }
    }
}
