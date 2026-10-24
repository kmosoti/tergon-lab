# Publication status and remaining setup

The owner created the public repository and Wiki on 2026-10-24. Source and
history are pushed, eight component/experiment issues are open, and the
[foundation PR #9](https://github.com/kmosoti/tergon-lab/pull/9) is ready for review.
The original GitHub initialization commit is preserved. The Wiki initialization
commit is also preserved; full pages publish from reviewed main after merge.

The connected integration can push branches and create PRs/issues. Repository
settings PATCH, environment PUT, and ruleset POST each returned HTTP 403,
"Resource not accessible by integration". No branch ruleset or protected
owner-review environment has been applied. A passing Actions job alone does not
establish owner approval while these settings are missing.

The owner must apply policy/repository-settings.json, policy/github-ruleset.json,
and an owner-review environment with kmosoti as required reviewer and administrator
bypass disabled. Use a separately authorized administrative credential; keep it
out of contributor agent sessions. The prepared bootstrap script stops on API
errors and never merges or releases. Running it again with administration access
will reconcile settings and existing exact-title issues.

The ruleset configuration requires a code-owner review plus seven checks. GitHub
forbids an author approving their own PR. Once the contributor App is installed,
reopen the unchanged implementation branch as an App-authored PR so the solo owner
can approve it, retaining the original PR link. Scoped merging remains disabled.

[App registration](wiki/GitHub-App-setup.md) is still pending. The native Wiki is
initialized. Its [publication workflow](wiki/Wiki-contributions.md) becomes usable
from main after the foundation is reviewed and merged. No App installation or
remote protection is claimed to exist.

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
