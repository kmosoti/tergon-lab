# Typed state v1 boundary decision

Issue [#14](https://github.com/kmosoti/tergon-lab/issues/14) introduces concrete
`State`, `Event`, `Assertion`, `Query`, `Answer`, `ReadPolicy` and reasoner/snapshot
APIs in engine. These types serve only the bounded
[typed-state protocol](Typed-state-experiment.md); they are not a generic memory
service. Engine owns validation, reduction, pure symbolic execution, feature
encoding, tensor execution and optimizer state. Lab owns generated event logs,
independent raw-log oracle, serialization, dataset partitions, checkpoints and
receipts. Math and all existing numerical contracts remain unchanged.

Keeping provenance/activity in reduced records makes the two reader treatments
consume identical inputs. A separate oracle implementation prevents the learned
reader or reducer from manufacturing its own evaluation labels. Retaining all
assertions preserves evidence after replacement/retraction. Explicit read policy
changes only eligibility and never the decoder, shape or supporting subject.

Snapshots contain ordered plain-host parameter/moment vectors and update counters;
lab's versioned JSON checkpoint also records run identity and shuffle/training
position. Restore validates complete shapes, finite values, nonnegative moments
and common counters before constructing a model. No unsafe code, new dependency,
trait framework, implicit device substitution or stable public interchange format
is introduced. New APIs/formats and numerical fixtures require owner review.
