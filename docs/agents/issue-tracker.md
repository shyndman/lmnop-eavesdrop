# Issue tracker: GitHub

Issues and specs live in GitHub Issues for `shyndman/lmnop-eavesdrop`.
Use `gh` from the repository root.

## Operations

- Create: `gh issue create --title "..." --body "..."`
- Read: `gh issue view <number> --json number,title,body,labels,comments`
- List: `gh issue list --state open --json number,title,body,labels,comments`
- Comment: `gh issue comment <number> --body "..."`
- Label: `gh issue edit <number> --add-label "..." --remove-label "..."`
- Close: `gh issue close <number> --comment "..."`
- Link a child: `gh issue edit <parent> --add-sub-issue <child>`

For multiline bodies, use `--body-file` with a file or standard input.

Publishing to the issue tracker means creating a GitHub issue.
Fetching a relevant ticket means reading its issue and comments.

## Pull requests as a triage surface

**PRs as a request surface: no.**

## Wayfinding

- Map: one issue labelled `wayfinder:map`, containing Notes,
  Decisions-so-far, and Fog.
- Children: sub-issues labelled `wayfinder:<type>`, where type is
  `research`, `prototype`, `grilling`, or `task`.
- Blocking: use GitHub native issue dependencies:
  `gh api --method POST repos/{owner}/{repo}/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`.
  Obtain the database ID with
  `gh api repos/{owner}/{repo}/issues/<blocker> --jq .id`.
- Where native relationships are unavailable, list children in the map,
  add `Part of #<map>` to children, and record `Blocked by: #<number>`
  at the top of blocked tickets.
- Frontier: the first open child in map order with no open blockers
  and no assignee. Native `issue_dependencies_summary.blocked_by`
  counts open blockers.
- Claim: `gh issue edit <number> --add-assignee @me`.
- Resolve: comment with the answer, close the child, and append a gist
  and link to the map's Decisions-so-far.
