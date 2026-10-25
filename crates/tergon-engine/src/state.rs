//! Bounded typed event worlds and deterministic activity reduction for issue 14.
use std::{error::Error, fmt};

/// Entity count in the frozen experiment.
pub const ENTITIES: usize = 16;
/// Maximum retained assertions, excluding the NULL memory slot.
pub const MEMORY_SLOTS: usize = 32;
/// Concrete entity kinds; exactly four of each occur in a world.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntityType {
    /// Human actor.
    Person,
    /// Portable owned object.
    Item,
    /// Nestable container.
    Container,
    /// Named location.
    Place,
}
/// Directed relation operations. Type signatures are independent of entity IDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Relation {
    /// Item/container inside a container/place.
    Inside,
    /// Item owned by a person.
    Owner,
    /// Person located at a place.
    At,
    /// Person carries an item.
    Carries,
    /// Directed place-to-place connection.
    Link,
}
impl Relation {
    /// Fixed vocabulary order used by input encoding.
    pub const ALL: [Self; 5] = [
        Self::Inside,
        Self::Owner,
        Self::At,
        Self::Carries,
        Self::Link,
    ];
    /// Vocabulary index, independent of labels and paths.
    pub fn index(self) -> usize {
        Self::ALL.iter().position(|r| *r == self).unwrap()
    }
    /// Declared source and destination type compatibility.
    pub fn accepts(self, source: EntityType, destination: EntityType) -> bool {
        use EntityType::*;
        match self {
            Self::Inside => {
                matches!(source, Item | Container) && matches!(destination, Container | Place)
            }
            Self::Owner => source == Item && destination == Person,
            Self::At => source == Person && destination == Place,
            Self::Carries => source == Person && destination == Item,
            Self::Link => source == Place && destination == Place,
        }
    }
}
impl EntityType {
    /// Fixed type vocabulary index.
    pub fn index(self) -> usize {
        match self {
            Self::Person => 0,
            Self::Item => 1,
            Self::Container => 2,
            Self::Place => 3,
        }
    }
}
/// Provenance status. A hypothesis never establishes an observed fact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evidence {
    /// Report accepted as observed evidence.
    Observed,
    /// Unverified proposal, excluded from symbolic truth.
    Hypothesis,
}
/// Immutable assertion; identity, source and logical time are retained.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assertion {
    /// Unique assertion identity within this log; nonzero.
    pub id: u32,
    /// Source entity ID.
    pub subject: usize,
    /// Directed relation.
    pub relation: Relation,
    /// Destination entity ID.
    pub object: usize,
    /// Source identity from a fixed four-source vocabulary.
    pub source: usize,
    /// Monotone event time (ties rejected).
    pub time: u32,
    /// Observed assertion or unsupported hypothesis.
    pub evidence: Evidence,
}
/// Log action. Replacement retracts exactly the named assertion then asserts a new one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// Retain a new assertion.
    Assert(Assertion),
    /// Deactivate an existing active assertion, at a strictly later time.
    Retract {
        /// Target assertion identity.
        target: u32,
        /// Event time.
        time: u32,
    },
    /// Atomic explicit replacement; the new assertion must have a fresh identity.
    Replace {
        /// Target assertion identity.
        target: u32,
        /// New assertion, including event time.
        assertion: Assertion,
    },
}
/// Assertion plus activity derived once from the event log, shared by both readers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    /// Original assertion metadata.
    pub assertion: Assertion,
    /// Whether this assertion survives all explicit updates.
    pub active: bool,
}
/// A validated immutable context. Records may be reordered without changing semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State {
    /// Entity types indexed by randomized ID.
    pub types: [EntityType; ENTITIES],
    /// Retained assertions, including inactive claims.
    pub records: Vec<Record>,
}
/// One to three named relation operations from an entity ID.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Query {
    /// Starting entity ID.
    pub start: usize,
    /// Named operations, in execution order.
    pub relations: Vec<Relation>,
}
/// Semantic result, encoded into the same unrestricted 18 classes by both models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Answer {
    /// A uniquely supported entity ID.
    Entity(usize),
    /// No active observed destination.
    Unknown,
    /// Two or more distinct observed destinations at any step.
    Conflict,
}
impl Answer {
    /// Fixed decoder class, 0..15 entities, 16 Unknown, 17 Conflict.
    pub fn class(self) -> usize {
        match self {
            Self::Entity(id) => id,
            Self::Unknown => 16,
            Self::Conflict => 17,
        }
    }
}
/// The only neural-reader intervention; it never depends on answers or supporting paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadPolicy {
    /// Mask padding only; all present metadata remains visible.
    MetadataOnly,
    /// Also mask inactive, hypothetical and operation-incompatible records.
    Enforced,
}
/// Malformed input rejected before tensor creation or unbounded work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateError {
    /// Invalid entity/type/source/identity/time or relation signature.
    InvalidRecord,
    /// Event log or assertion count exceeds fixed bounds.
    Capacity,
    /// Retraction/replacement does not identify a current active assertion.
    InvalidTarget,
    /// Query is empty, too long, or has an invalid starting ID.
    InvalidQuery,
}
impl fmt::Display for StateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl Error for StateError {}
impl Query {
    /// Validate the public query boundary without consulting the answer.
    pub fn validate(&self) -> Result<(), StateError> {
        if self.start >= ENTITIES || self.relations.is_empty() || self.relations.len() > 3 {
            Err(StateError::InvalidQuery)
        } else {
            Ok(())
        }
    }
}
impl State {
    /// Replay at most 96 strictly time-ordered events; never infer replacement from recency.
    pub fn reduce(types: [EntityType; ENTITIES], events: &[Event]) -> Result<Self, StateError> {
        if events.len() > 96 {
            return Err(StateError::Capacity);
        }
        let mut state = Self {
            types,
            records: Vec::new(),
        };
        let mut time = 0;
        for event in events {
            let (target, assertion, now) = match event {
                Event::Assert(a) => (None, Some(a), a.time),
                Event::Retract { target, time } => (Some(*target), None, *time),
                Event::Replace { target, assertion } => {
                    (Some(*target), Some(assertion), assertion.time)
                }
            };
            if now <= time {
                return Err(StateError::InvalidRecord);
            }
            if let Some(a) = assertion {
                if state.records.len() == MEMORY_SLOTS {
                    return Err(StateError::Capacity);
                }
                if a.id == 0
                    || a.subject >= ENTITIES
                    || a.object >= ENTITIES
                    || a.source >= 4
                    || state.records.iter().any(|r| r.assertion.id == a.id)
                    || !a.relation.accepts(types[a.subject], types[a.object])
                {
                    return Err(StateError::InvalidRecord);
                }
            }
            if let Some(id) = target {
                let r = state
                    .records
                    .iter_mut()
                    .find(|r| r.assertion.id == id && r.active)
                    .ok_or(StateError::InvalidTarget)?;
                r.active = false;
            }
            if let Some(a) = assertion {
                state.records.push(Record {
                    assertion: a.clone(),
                    active: true,
                });
            }
            time = now;
        }
        state.validate()?;
        Ok(state)
    }
    /// Validate materialized records, including distinct IDs and four entities of each type.
    pub fn validate(&self) -> Result<(), StateError> {
        if self.records.len() > MEMORY_SLOTS {
            return Err(StateError::Capacity);
        }
        let mut counts = [0; 4];
        for t in self.types {
            counts[t.index()] += 1;
        }
        if counts != [4; 4] {
            return Err(StateError::InvalidRecord);
        }
        for (i, r) in self.records.iter().enumerate() {
            let a = &r.assertion;
            if a.id == 0
                || a.time == 0
                || a.subject >= ENTITIES
                || a.object >= ENTITIES
                || a.source >= 4
                || !a
                    .relation
                    .accepts(self.types[a.subject], self.types[a.object])
                || self.records[..i].iter().any(|r| r.assertion.id == a.id)
            {
                return Err(StateError::InvalidRecord);
            }
        }
        Ok(())
    }
    /// Exact symbolic baseline over this reduced context. Duplicate agreeing support is legal.
    pub fn answer(&self, query: &Query) -> Result<Answer, StateError> {
        self.validate()?;
        query.validate()?;
        let mut current = query.start;
        for relation in &query.relations {
            let mut destination = None;
            for record in &self.records {
                let a = &record.assertion;
                if record.active
                    && a.evidence == Evidence::Observed
                    && a.subject == current
                    && a.relation == *relation
                {
                    if destination.is_some_and(|d| d != a.object) {
                        return Ok(Answer::Conflict);
                    }
                    destination = Some(a.object);
                }
            }
            match destination {
                Some(id) => current = id,
                None => return Ok(Answer::Unknown),
            }
        }
        Ok(Answer::Entity(current))
    }
    /// Eligibility of one retained assertion for an operation; never filters by a learned current entity.
    pub fn eligible(&self, slot: usize, relation: Relation, policy: ReadPolicy) -> bool {
        self.records.get(slot).is_some_and(|r| {
            policy == ReadPolicy::MetadataOnly
                || (r.active
                    && r.assertion.evidence == Evidence::Observed
                    && r.assertion.relation == relation)
        })
    }
}
