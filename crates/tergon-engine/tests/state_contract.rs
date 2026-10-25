//! Semantic, provenance and bounded-input contract fixtures.
use tergon_engine::state::*;
fn types() -> [EntityType; 16] {
    std::array::from_fn(|i| {
        [
            EntityType::Person,
            EntityType::Item,
            EntityType::Container,
            EntityType::Place,
        ][i / 4]
    })
}
fn a(id: u32, s: usize, r: Relation, o: usize) -> Assertion {
    Assertion {
        id,
        subject: s,
        relation: r,
        object: o,
        source: 0,
        time: id,
        evidence: Evidence::Observed,
    }
}
fn q() -> Query {
    Query {
        start: 4,
        relations: vec![Relation::Owner, Relation::At],
    }
}
#[test]
fn missing_conflict_duplicates_and_hypotheses() {
    let mut events = vec![Event::Assert(a(1, 4, Relation::Owner, 0))];
    assert_eq!(
        State::reduce(types(), &events)
            .unwrap()
            .answer(&q())
            .unwrap(),
        Answer::Unknown
    );
    events.push(Event::Assert(a(2, 0, Relation::At, 12)));
    events.push(Event::Assert(a(3, 4, Relation::Owner, 0)));
    assert_eq!(
        State::reduce(types(), &events)
            .unwrap()
            .answer(&q())
            .unwrap(),
        Answer::Entity(12)
    );
    let mut h = a(4, 4, Relation::Owner, 1);
    h.evidence = Evidence::Hypothesis;
    events.push(Event::Assert(h));
    assert_eq!(
        State::reduce(types(), &events)
            .unwrap()
            .answer(&q())
            .unwrap(),
        Answer::Entity(12)
    );
    events.push(Event::Assert(a(5, 4, Relation::Owner, 1)));
    // Conflict terminates immediately, even when the conflicting branch has no later edge.
    assert_eq!(
        State::reduce(types(), &events)
            .unwrap()
            .answer(&q())
            .unwrap(),
        Answer::Conflict
    );
}
#[test]
fn replacement_retraction_are_explicit_and_id_specific() {
    let events = vec![
        Event::Assert(a(1, 4, Relation::Owner, 0)),
        Event::Assert(a(2, 4, Relation::Owner, 0)),
        Event::Replace {
            target: 1,
            assertion: a(3, 4, Relation::Owner, 1),
        },
        Event::Retract { target: 2, time: 4 },
    ];
    let state = State::reduce(types(), &events).unwrap();
    assert_eq!(state.records.iter().filter(|r| r.active).count(), 1);
    assert_eq!(
        state
            .answer(&Query {
                start: 4,
                relations: vec![Relation::Owner]
            })
            .unwrap(),
        Answer::Entity(1)
    );
    assert!(!state.eligible(0, Relation::Owner, ReadPolicy::Enforced));
    assert!(state.eligible(0, Relation::At, ReadPolicy::MetadataOnly));
    assert!(!state.eligible(2, Relation::At, ReadPolicy::Enforced));
    let events = vec![
        Event::Assert(a(1, 4, Relation::Owner, 0)),
        Event::Assert(a(2, 4, Relation::Owner, 1)),
    ];
    assert_eq!(
        State::reduce(types(), &events)
            .unwrap()
            .answer(&Query {
                start: 4,
                relations: vec![Relation::Owner]
            })
            .unwrap(),
        Answer::Conflict
    );
}
#[test]
fn malformed_boundaries() {
    assert!(
        Query {
            start: 16,
            relations: vec![Relation::Link]
        }
        .validate()
        .is_err()
    );
    assert!(
        Query {
            start: 0,
            relations: vec![]
        }
        .validate()
        .is_err()
    );
    assert!(
        Query {
            start: 0,
            relations: vec![Relation::At; 4]
        }
        .validate()
        .is_err()
    );
    let mut invalid = a(1, 4, Relation::At, 12);
    assert!(State::reduce(types(), &[Event::Assert(invalid.clone())]).is_err());
    invalid.relation = Relation::Owner;
    invalid.object = 16;
    assert!(State::reduce(types(), &[Event::Assert(invalid)]).is_err());
    assert!(State::reduce(types(), &[Event::Retract { target: 3, time: 1 }]).is_err());
    assert!(
        State::reduce(
            types(),
            &[
                Event::Assert(a(1, 4, Relation::Owner, 0)),
                Event::Assert(a(1, 4, Relation::Owner, 0))
            ]
        )
        .is_err()
    );
    assert!(State::reduce(types(), &vec![Event::Retract { target: 1, time: 1 }; 97]).is_err());
    let events: Vec<_> = (1..=33)
        .map(|id| Event::Assert(a(id, 4, Relation::Owner, 0)))
        .collect();
    assert_eq!(State::reduce(types(), &events), Err(StateError::Capacity));
    let mut wrong = types();
    wrong[0] = EntityType::Item;
    assert!(State::reduce(wrong, &[]).is_err());
}
#[test]
fn relation_signatures_and_order_invariance() {
    assert!(Relation::Inside.accepts(EntityType::Container, EntityType::Place));
    assert!(!Relation::Owner.accepts(EntityType::Container, EntityType::Person));
    let mut state = State::reduce(
        types(),
        &[
            Event::Assert(a(1, 4, Relation::Owner, 0)),
            Event::Assert(a(2, 0, Relation::At, 12)),
        ],
    )
    .unwrap();
    let answer = state.answer(&q()).unwrap();
    state.records.reverse();
    assert_eq!(state.answer(&q()).unwrap(), answer);
}
