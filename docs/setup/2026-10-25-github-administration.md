# GitHub administration receipt: 2026-10-25 UTC

Related issue: [App boundary #8](https://github.com/kmosoti/tergon-lab/issues/8).
This receipt records a later setup session; earlier publication failures remain
in [PUBLISHING](../PUBLISHING.md). Observations below are a dated snapshot, not a
claim that pending identities or scoped merging are implemented.

## Source and authority

A fresh clone of current main was used at
`66b84af7750366f77ee157a31aaa9ef289dc8cfb`, containing merged
[foundation PR #9](https://github.com/kmosoti/tergon-lab/pull/9).
Its [main verification run](https://github.com/kmosoti/tergon-lab/actions/runs/40143351571)
succeeded. No genuine main CI failure was found. The original repository checkout
and the main/Wiki histories were preserved. The bootstrap publication script was
not rerun.

The owner explicitly authorized this administration session using existing `gh`
owner access. That credential is an OAuth credential, not a contributor App token.
Its repository metadata reports admin access; successful mutations were followed
by GET/readback verification. No credential values are recorded here. Owner
credentials must not be treated as evidence of restricted agent permissions.

## Applied and verified controls

[Repository settings](https://github.com/kmosoti/tergon-lab/settings) match every
field in policy/repository-settings.json: Issues and Wiki enabled; squash enabled;
merge commits, rebases, and automatic merging disabled; merged branches deleted;
squash title/body taken from PR title/body.

[Main ruleset 25801689](https://github.com/kmosoti/tergon-lab/rules/25801689)
is active on `~DEFAULT_BRANCH`. There were no existing rulesets to reconcile.
Readback and the effective main rules endpoint confirm:

- No bypass actors, branch deletion, or non-fast-forward updates.
- Linear history, PR-only changes, and squash-only PR merges.
- One approving review with code-owner review, stale-review dismissal, and
  resolved review discussions required.
- Strict required checks: rust, architecture, numerics, evidence, docs,
  contribution, and owner-gate; enforced on branch creation too.
- All seven expected sources use integration ID `15368`, matched against actual
  Actions check runs on main and the foundation PR. No App verifier is installed
  by this session, and the expected source was not switched to a hypothetical App.

GitHub added `required_reviewers: []` and
`require_extra_approval_for_unattributed_changes: true` to the returned PR rule.
Every field specified by repository policy matches the returned configuration.
The separate last-push-approval option remains false as specified in policy.

The [owner-review environment](https://github.com/kmosoti/tergon-lab/settings/environments)
has ID `25242324110`, reviewer-protection rule ID `69997710`, and exactly one
required reviewer: User `kmosoti` (ID `47609243`). Readback confirms
`can_admins_bypass: false` and `prevent_self_review: false`.
Environment approval permits the solo-owner workflow; GitHub PR review still
forbids author self-approval. Neither PR approval nor environment approval was
performed for the production PRs during this session.

The trusted base workflow checks that the environment names kmosoti as reviewer,
executes no PR checkout/code, and compares current head, base, open state, and
draft state after environment approval. Its PR concurrency group cancels old
runs on head updates. Strict checks require an up-to-date base. Source inspection
supports these statements; full stale-approval/base-change tests remain pending.

## Wiki publication and navigation finding

[Publish reviewed wiki run 40145046052](https://github.com/kmosoti/tergon-lab/actions/runs/40145046052)
succeeded from current main. The [Wiki](https://github.com/kmosoti/tergon-lab/wiki)
now contains all 10 canonical docs/wiki pages.

- Source revision: `66b84af7750366f77ee157a31aaa9ef289dc8cfb`.
- Wiki publication commit: `be04df38af560f0d3046243976cda819e23d57f6`.
- Every published page identifies that source revision in its footer.
- `.tergon-sync.json` exists; its source_revision, page set, and SHA-256 file
  hashes were independently checked against the Wiki checkout.
- Existing Wiki commit `a54586fd0d26e8601a87f7938c0ae1094c549365` remains an ancestor.
  No direct edits were discarded and no drift bypass/reset was used.

Navigation did not pass acceptance: Home's relative `.md` links render as
`wiki/Architecture.md` and equivalent targets that redirect back to Home.
[PR #10](https://github.com/kmosoti/tergon-lab/pull/10) changes the nine topic links
to canonical Wiki page URLs. All nine proposed destinations returned HTTP 200,
retained their canonical URLs, and had the intended page titles. The currently
published Home still has the defect. Owner-reviewed merge of #10 and another
publication from current main are required to complete navigation acceptance.

## Boundary observations

Tests used the isolated public
[disposable repository](https://github.com/kmosoti/tergon-lab-admin-boundary-20261025),
with ruleset `25801756`, no bypass actors, matching main rules, matching
owner-review protection, and wildcard CODEOWNERS. Only owner credentials were
available. The fixture omits Actions workflows and production runtime code;
therefore it expects seven missing checks and cannot exercise workflow execution
or numerical correctness. It is retained for inspection, with test PRs unmerged.

| Probe | Observed result | Limit |
| --- | --- | --- |
| Direct main write | Git GH013: PR required and seven checks missing | Owner credential, not contributor token |
| Non-fast-forward main push | Git GH013: cannot force-push, PR/checks required | Disposable main only |
| Main deletion | Git rejected deleting the current/default branch | Does not independently isolate the deletion ruleset rule |
| [Fixture PR #1](https://github.com/kmosoti/tergon-lab-admin-boundary-20261025/pull/1) author approval | HTTP 422: cannot approve your own PR | Separate from environment self-review |
| Forged owner-gate check run | HTTP 403: must authenticate via a GitHub App | OAuth denial; contributor App denial still untested |
| Fixture PR #1 merge without review/checks | HTTP 405: approving review and seven checks required | No merge occurred |
| [Fixture PR #2](https://github.com/kmosoti/tergon-lab-admin-boundary-20261025/pull/2) with unreviewed governance/numerical-contract documents | HTTP 405: approving review and seven checks required | Universal review gate, not semantic scope classification |
| Copy trusted workflow into fixture | Push rejected: OAuth App lacks workflow scope | No permission expansion was attempted |
| Production PR #10 head update | Old waiting gate cancelled; replacement waits for kmosoti | No approval was granted or reused |

The head-update observation compares
[cancelled run 40145238522](https://github.com/kmosoti/tergon-lab/actions/runs/40145238522)
for `869407014c3ce7addcb10a0da0b5c38fce59b68b` with
[waiting run 40145270574](https://github.com/kmosoti/tergon-lab/actions/runs/40145270574)
for `9cdd8c675e01d6d5f4b04306ceae3edd19bd2fdf`.
The latter pending-deployments response names only kmosoti and reports that the
owner can approve, confirming the configured solo-owner environment path.

Pending tests: contributor-token main/check-forgery denials, broker refusal to
issue verifier credentials to contributors, approval reuse after an already
approved head/base changes, expired/revoked tokens, and installation removal.
These are not reported as passes. No trusted scoped classifier/merger exists;
policy/agent-scopes.json remains unchanged with enabled false and no scopes.

## App owner handoff

No App was registered, installed, or issued credentials by this session. Existing
private registrations/installation IDs could not be enumerated with this OAuth
credential: the installation-list API requires App authentication. No browser
surface was available. App ID and installation ID are therefore **unverified and
pending**, not assumed absent or fabricated.

The owner can complete registration without giving the agent a key:

1. Open [New GitHub App](https://github.com/settings/apps/new); first inspect
   existing private Apps to avoid duplicate registration. Use an available name
   such as `kmosoti-tergon-lab-automation`, homepage
   `https://github.com/kmosoti/tergon-lab`, disable webhooks, and allow installation
   only on this account. A polling integration needs no callback or webhook URL.
2. Grant only repository Contents write, Pull requests write, Issues write,
   Checks write, Actions read, and Metadata read. Grant no Administration,
   Workflows, Secrets, user permissions, or ruleset bypass.
3. Install with **Only select repositories**, initially `tergon-lab`
   (repository ID `1432666512`). Record the non-secret App/installation IDs in a
   later receipt. Explicitly select the disposable repository only when running
   identity tests; do not silently use all-repository access.
4. Do not generate/download a private key into an agent workspace. No broker or
   trusted key store is configured. Provision a trusted issuer outside agent
   workspaces as separate work before generating/storing the key directly there.
5. Have that issuer narrow every installation token to the intended repository
   ID and exactly its policy/app-token-profiles.json role permissions. Contributors
   get no checks write, administration, workflows, verifier token, or private key.
   Registering one App does not itself separate token roles: all share its identity.
6. Run and record the pending identity/expiry tests before claiming isolation.
   Implement the trusted classifier/merger separately; keep scoped merging off.

## Local verification and remaining review

The five required commands passed on the foundation checkout and on PR #10's
final documentation revision and this setup-documentation revision, using Rust
1.99.1 on Fedora x86_64 CPU:

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps
cargo xtask check
```

No GPU checks were run; this session changes administration and documentation,
not numerical behavior, device placement, or runtime APIs.
[PR #10 verification run 40145271779](https://github.com/kmosoti/tergon-lab/actions/runs/40145271779)
also passed all six unprivileged CI checks; owner-gate is waiting for approval.

The navigation and receipt PRs are owner-authored with the current credential.
GitHub will not let kmosoti approve them as PR author. Arrange a contributor-authored
replacement PR preserving the commits/links, or another valid code-owner review
arrangement, before merging. Do not weaken the review rule to resolve this.
The owner must separately approve the environment for the current head/base.
Neither PR will be merged by this setup session.
