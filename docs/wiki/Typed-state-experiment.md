# Typed state experiment v1

Issue [#14](https://github.com/kmosoti/tergon-lab/issues/14) tests whether explicit
read contracts improve final-answer reasoning in a small ternary memory network.
This protocol and the [recipe](../../experiments/typed-state-v1.json) must be
committed before formal runs. A negative result completes the experiment; failed
baseline controls or incomplete runs are inconclusive. Nothing here changes a
runtime default. The conditional storage pilot is
[#15](https://github.com/kmosoti/tergon-lab/issues/15).

The [boundary decision](Typed-state-ADR.md) records API and snapshot ownership.

## Semantic contract and generator

There are 16 entity IDs, randomly permuted per world, with four each of Person,
Item, Container and Place. Signatures are inside (Item|Container) ->
(Container|Place), owner Item -> Person, at Person -> Place, carries Person ->
Item, and link Place -> Place. IDs encode neither type nor role. Relation
composition takes 1–3 named operations from a starting ID. An absent active
observed destination returns Unknown immediately; distinct destinations return
Conflict immediately, even if a later operation is missing. Agreeing duplicates
are legal. Hypotheses never establish a fact.

Assertions retain unique randomized record IDs, source (four exchangeable
sources), strictly increasing event times, and evidence status. Explicit
retraction deactivates exactly one active assertion ID; explicit replacement
atomically deactivates that assertion and retains a new assertion with fresh ID.
Recency alone does not replace anything. The shared deterministic reducer retains
all assertions, marks activity, rejects malformed signatures/IDs/events, and caps
logs at 96 events and memory at 32 assertions. All models receive the same reduced
records. The independent lab oracle replays a raw-event map and traverses sets;
it does not call the engine reducer, baseline, masks or type signatures.

Worlds have one random valid backbone, with four related counterfactual branches:
two different uniquely supported answers, one missing step, and one conflict.
Every context includes distinct agreeing observed support at a queried key;
the missing branch redirects both copies into an irrelevant decoy key. This
prevents learning Conflict from claim count alone.
All four branches share the query and common distractors; they have identical
record counts and exactly identical per-slot relation/type/status/source/time/ID
metadata, apart from the subject/object ID changes being tested. One branch redirects a
critical subject; another redirects an irrelevant compatible assertion to create
conflict. Random branch order prevents positions from revealing outcomes.
Assertion IDs, initial chronology and memory slot ordering are randomized
independently of paths. A world and every branch/metamorphic variant stay in the
same split and bootstrap cluster. No branch/world/outcome identity enters model
features. Each split has exactly 50% known, 25% Unknown and 25% Conflict.

Train 8,192 worlds, validation 1,024 worlds, and 512 worlds in each of ordinary,
unseen composition, replacement/retraction, conflicts/hypotheses and distractors.
Training and validation exclude inside/inside/inside and owner/at/link; unseen
composition contains these triples. Their shorter constituents occur in training.
Normal contexts have 8–16 assertions, distractor contexts 24–32. Every tensor
uses 32 padded slots plus NULL. Hypothesis probability is 1/4 for background
assertions and 1/2 in the conflicts/hypotheses stratum. Background records avoid
backbone subject/relation keys, so they cannot accidentally change oracle labels.
Half of train/validation/ordinary/other worlds and every replacement/retraction
stress world retain inactive assertions through explicit updates, teaching the
activity primitive before testing stress. Every context has an incorrect active
query-related hypothesis, including hypothesis-only support at an Unknown missing
key; the hypothesis stress adds further such claims where capacity permits.
History branches apply identical update targets and timestamps in every branch,
preserving stale query-related claims with explicit updates;
missing branches use a retraction followed by a new irrelevant assertion.

`state-audit <new-dataset-dir>` persists the corpus with per-group hashes, an
independent symbolic baseline, label/hop balance, metadata histograms, record-order
and renaming checks. The receipt separates combined audit timing from isolated symbolic query timing
over materialized contexts. The latter includes public-API validation and excludes
generation, reduction and serialization. Live symbolic storage counts inline
state/query values and record/relation vector capacity, excluding allocator
bookkeeping; serialized corpus bytes are reported separately. Receipts identify
source revision/dirty state and config/generator hashes.

## Reader and numerical contract

The tied-hop reader adapts [Sukhbaatar et al., End-To-End Memory Networks
(2015)](https://arxiv.org/abs/1503.08895). Record features are subject ID one-hot16,
object ID one-hot16, relation one-hot5, subject/object types one-hot4 each, source
one-hot4, record identity one-hot32, logical time /96, active bit, observed bit,
and NULL bit: 85 floats. Query features are start ID one-hot16 and its type
one-hot4. No oracle answer, supporting path, conflict indicator, world family,
branch ID or label enters these features.

Six linear maps with floating biases project keys/values 85->64, query20->64,
relation5->64, shared update64->64, decoder64->18. At each requested operation,
`z = h + R(operation)`, `p = softmax(K z / sqrt(64))`,
`read = sum_slot p_slot V_slot`, `h = ReLU(U z + read)`.
There is one semantic hop per relation. Fixed-shape execution computes three
candidate hops for every batch and freezes state after each query’s actual
length; timings include this padding work. The same decoder produces 18 unrestricted logits.
No answer/type restriction is applied to output. Padding is always masked; NULL
has its own input bit and is always eligible. Metadata-only attends every present
record; enforced additionally excludes inactive, hypothetical and current-relation
incompatible records. Neither policy masks by a predicted/ground-truth subject.

Each of seeds17,29,43 has metadata-only/enforced x float/ternary conditions with
identical initial latent matrices, biases and batch ordering. All matrices use
the existing rowwise quantization/identity STE or all use f32; activations remain
f32. Existing numerical tolerances and MNIST behavior remain unchanged. Mean
final-answer cross entropy is the only training objective. AdamW lr .001, betas
.9/.999, epsilon1e-8, matrix decay .01, bias decay0, no clipping/schedule.
Batch128,20 epochs, final checkpoints, no best-validation selection.

`state-train <config> <new-run-dir> [--resume <checkpoint>]` executes the twelve
frozen conditions. Optional `TERGON_STATE_RUN=17:metadata-only:float` selects one
exact approved seed/policy/mode, is strictly validated and recorded verbatim and
parsed, and changes no data/order/initialization/hyperparameter. Aggregate analysis
requires all twelve distinct matching source/config/data conditions. Resume rejects
conflicting selection and carries elapsed execution budget across processes.

Each run has 600 seconds excluding compilation and dataset generation. Record
incomplete runs and errors; do not extend budgets after observing outcomes.
Checkpoints carry parameters, moments, counters, data/config/source identity,
shuffle RNG state/order and training position. Serialized replay after step200
covers the next50 updates: CPU bitwise; NVIDIA GPU `1e-5 + 1e-4*abs(reference)`.

## Analysis and gate

Report per-world predictions/probabilities and accuracy by stratum/hop/outcome.
Multiclass Brier is `sum_18 (p-y)^2`, averaged over examples, range0..2, without
class-count normalization. Confidently unsupported means an entity prediction
with probability >=.9 that is not the uniquely supported symbolic answer.
Report entity-renaming equivariance, record-order consistency, and paired answers
after counterfactual changes. Record checkpoint/source/config/data hashes, curves,
execution/host-read times, memory/transfer accounting, hardware and failures.

Primary comparison equally weights the four stress strata. Resample whole worlds
within each stress stratum 10,000 times with SplitMix64 seed20261025. Average the
three model seeds inside each sampled world; use the two-sided95% percentile
interval and report seed variation separately. The storage pilot opens only if
all twelve runs complete and ALL conditions hold:

- Metadata-only float ordinary accuracy >=90% in every seed.
- Enforced ternary ordinary >=90% and aggregate stress >=85% in every seed.
- Mean ternary stress gain >=5 percentage points, positive in every seed, and
  positive lower bound of the paired world-bootstrap interval.
- Enforced-minus-metadata ternary ordinary accuracy >=-2 percentage points in
  every seed; equally weighted five-stratum Brier worsening <=.02 in every seed.

Independent scalar forward/backward and masked-gradient fixtures, semantic and
malformed-input tests, serialized replay, all five repository checks and explicit
RTX3060 Laptop CUDA fixtures precede formal execution. Incomplete runs cannot
open the gate. Learned storage, HDC, learned type discovery, ternary activations,
optimizers/kernels/offload/distribution remain excluded from this main experiment.
