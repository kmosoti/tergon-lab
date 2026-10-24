#!/usr/bin/env bash
# Administrative bootstrap only. Does not merge PRs or publish releases.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
repo=kmosoti/tergon-lab
branch=feat/1-component-foundation
test -z "$(git status --porcelain)" || { echo 'Commit or preserve local changes first.' >&2; exit 1; }
test "$(git remote get-url origin)" = "https://github.com/$repo.git"
test "$(gh api "repos/$repo" --jq .private)" = false || {
  echo "Create the empty public $repo repository and authorize repository administration first." >&2
  exit 1
}
remote_main=$(git ls-remote --heads origin main | cut -f1)
if test -z "$remote_main"; then
  test -z "$(git ls-remote origin)" || { echo 'Remote is not empty; refusing bootstrap.' >&2; exit 1; }
  git push -u origin main
elif test "$remote_main" != "$(git rev-parse main)"; then
  echo 'Remote main differs from the bootstrap. Reconcile it before publishing.' >&2
  exit 1
fi
gh api --method PATCH "repos/$repo" --input policy/repository-settings.json >/dev/null
owner_id=$(gh api users/kmosoti --jq .id)
env_config=$(mktemp)
pr_body=$(mktemp)
trap 'rm -f "$env_config" "$pr_body"' EXIT
jq -n --argjson id "$owner_id" '{wait_timer:0, prevent_self_review:false,
  reviewers:[{type:"User",id:$id}], can_admins_bypass:false}' > "$env_config"
gh api --method PUT "repos/$repo/environments/owner-review" --input "$env_config" >/dev/null
ruleset_id=$(gh api "repos/$repo/rulesets" --jq '.[] | select(.name == "main: reviewed and verified") | .id')
if test -n "$ruleset_id"; then
  gh api --method PUT "repos/$repo/rulesets/$ruleset_id" --input policy/github-ruleset.json >/dev/null
else
  gh api --method POST "repos/$repo/rulesets" --input policy/github-ruleset.json >/dev/null
fi
git push -u origin "$branch"
foundation_issue=
for body in .github/project-issues/*.md; do
  title=$(sed -n '1s/^# //p' "$body")
  existing=$(gh issue list --repo "$repo" --state all --limit 100 --json title,number |
    jq -r --arg title "$title" '.[] | select(.title == $title) | .number')
  if test -z "$existing"; then
    issue_url=$(gh issue create --repo "$repo" --title "$title" --body-file "$body")
    existing=${issue_url##*/}
  fi
  case "$body" in *01-foundation.md) foundation_issue=$existing ;; esac
done
cat docs/foundation-pr.md > "$pr_body"
printf '\nRefs #%s\n' "$foundation_issue" >> "$pr_body"
if ! gh pr view "$branch" --repo "$repo" --json number >/dev/null 2>&1; then
  gh pr create --repo "$repo" --base main --head "$branch" \
    --title 'feat: establish verified Rust ternary component foundation' --body-file "$pr_body"
fi
echo 'Published source, issues, PR, and protection configuration. Verify remote rules and checks.'
echo 'App registration, independent review identity, and first Wiki page still require owner setup.'
