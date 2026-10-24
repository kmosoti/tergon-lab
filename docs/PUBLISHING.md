# Publication status and recovery

The implementation is committed locally. On 2026-10-24, both GitHub GraphQL
`createRepository` and REST `POST /user/repos` returned HTTP 403, "Resource not
accessible by integration", for kmosoti/tergon-lab. The public repository, remote
PR, issues, ruleset, and environment were therefore not created in this session.
Repository/admin permission must be available before those steps can succeed.

The owner can create an **empty public** repository named `tergon-lab`, without
a generated README or license, and grant the publishing credential access. The
publisher also needs permission to configure repository settings, environments,
and rulesets. This one-time administrative credential must not be reused as the
future contributor agent token.

From the clean local checkout, inspect and run:

```bash
bash -n scripts/publish.sh
bash scripts/publish.sh
```

The script publishes the bootstrap main and foundation branch, applies squash-only
settings and the no-bypass main ruleset, configures owner-review, creates bounded
milestone issues without duplicating exact titles, and opens the foundation PR.
It stops on any API error, does not force-push, rejects a conflicting remote main,
and never merges or releases. These remote operations remain untested because
repository creation was denied; review returned settings after execution.

The baseline workflow and CODEOWNERS are already in the local bootstrap commit.
The ruleset requires a code-owner review in addition to seven checks, so an
owner-authored PR cannot be self-approved. Use the App contributor identity to
reopen the same branch after setup, preserving the initial PR link and commit
history. Keep scoped merging disabled until the identity tests pass.

[App registration](wiki/GitHub-App-setup.md) and
[Wiki initialization](wiki/Wiki-contributions.md) are separate owner setup steps.
No App key, App installation, native Wiki, or remote protection is claimed to exist.

An offline Git bundle can preserve every local branch and commit:

```bash
git bundle create ../tergon-lab.bundle --all
git bundle verify ../tergon-lab.bundle
```

Restore with:

```bash
git clone -b feat/1-component-foundation tergon-lab.bundle tergon-lab
cd tergon-lab
git branch main origin/main
git remote set-url origin https://github.com/kmosoti/tergon-lab.git
```

The bundle includes code, compact evidence,
and history, but excludes target/, runs/, and the original multi-gigabyte archive.
