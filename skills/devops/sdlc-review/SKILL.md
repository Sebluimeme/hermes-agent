---
name: sdlc-review
description: Review Kanban handoffs and route verified outcomes.
version: 1.1.0
author: Jakub Wolniewicz (@frizikk) + Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [kanban, review, quality, verification]
    category: devops
    requires_toolsets: [kanban]
environments:
  - kanban
---

# SDLC Review Skill

Independently verify work handed from a Kanban implementation run to the review lane, then approve it, request changes, or escalate. This skill reviews the deliverable and its evidence; it does not take over the implementer's work.

## When to Use

Use this skill when all of the following are true:

- the dispatcher spawned you for a task claimed from the `review` lane;
- an implementer submitted a `review_requested` handoff;
- the task needs an independent verdict before it can be completed.

Do not use it for a separate downstream review card. A downstream card is ordinary implementation work with a review-oriented specification and completes through its own lifecycle.

## Prerequisites

- A Kanban worker context with the current task and run identifiers.
- Native Kanban tools: `kanban_show`, `kanban_comment`, `kanban_completion_ready`, `kanban_complete`, `kanban_request_changes`, `kanban_defer_review`, and `kanban_block`.
- Workspace access through `read_file`, `search_files`, and `terminal` when the deliverable is code.
- The task's original specification, acceptance criteria, handoff summary, and prior run history must be available through `kanban_show`.

## How to Run

This skill is loaded automatically by the review dispatcher. Start with `kanban_show` before inspecting files or choosing a verdict.

1. Read the task specification and the latest `review_requested` handoff.
2. Inspect the actual deliverable and run relevant verification.
3. Choose exactly one verdict: approve, request changes, or escalate.
4. Record concrete evidence in the terminal Kanban transition.

## Quick Reference

| Verdict | When | Final action |
|---|---|---|
| Approve | Acceptance criteria and verification pass | `kanban_complete` |
| Request changes | Correctable implementation defects remain | `kanban_comment`, then `kanban_request_changes` |
| Escalate | A human decision or external prerequisite is required | `kanban_block` |

A requested-changes transition returns the task to its original implementer. When that implementer requests review again without naming a reviewer, the persisted reviewer provenance routes the re-review back to the same reviewer profile.

## Review Lenses

Vary how you look at the work on each round instead of repeating the same inspection. Decorrelated lenses catch different defect classes: a cold read of the artifact surfaces design and correctness problems that the implementer's narrative would have framed away, execution surfaces claims that do not reproduce, and a strict contract audit surfaces quiet scope drift. Repeating the round-1 lens on round 3 mostly re-finds what round 1 already found.

Determine the current round from the history the task record already gives you: count the `changes_requested` entries in the "Prior attempts on this task" section of your worker context (also visible as prior runs in `kanban_show`). The current review round is that count plus one. Round 1 therefore shows zero `changes_requested` attempts; round 2 shows one; and so on.

| Round | Lens | How to apply it |
|---|---|---|
| 1 | Artifact | Read the diff or deliverable cold, before the implementer's summary. Form an independent judgment, then compare it against the handoff narrative and investigate every mismatch. |
| 2 | Execution | Check out the work and actually run it via `terminal`: build, test, and exercise the reported behavior yourself. Verify each handoff claim empirically instead of re-reading the artifact. |
| 3+ | Contract | Re-read the ORIGINAL task body and acceptance criteria, then audit the deliverable strictly against them. Also verify that every item from every prior `kanban_request_changes` round actually landed. |

The baseline duties in the Procedure section still apply on every round; the lens sets which inspection you lead with and weight most heavily.

### Lens variation for ad-hoc review fan-outs

The same principle applies outside the Kanban review lane. When spawning multiple parallel reviewers via `delegate_task`, give each reviewer a different lens — one diff-only brief, one full-context brief, one checkout-and-run brief — rather than identical briefs. Identical briefs produce correlated verdicts and duplicate findings; varied briefs cover more defect classes for the same review spend.

## Procedure

### 1. Orient from the durable task record

Call `kanban_show` and identify:

- the original task body and acceptance criteria;
- the latest implementation summary and structured metadata;
- changed files, commit identifiers, and test evidence;
- comments and decisions from earlier runs;
- findings from prior review rounds.

Treat the handoff as a claim to verify, not as proof that the work is correct.

### Web visual candidates

When the latest handoff contains `metadata.visual_review.required=true`, the
review is a two-stage visual gate in addition to the ordinary code checks:

1. Stay in the reviewer role; never edit the candidate. Before looking at the
   screenshots, establish the exact candidate diff and its pre-change baseline.
   Write down which visible surfaces the candidate can actually have changed.
2. Load every handoff screenshot with `vision_analyze`. The active Coder model
   sees the pixels natively; do not call Gemini for this intermediate review.
   Check the expected behavior and regressions attributable to the candidate:
   desktop/mobile consistency, overflow, truncation, contrast, empty states and
   the key interaction state. A visible issue that is demonstrably pre-existing
   or outside the task/diff is non-blocking; record it separately if useful, but
   never send the current implementer into unrelated work.
3. If a correctable defect caused by the candidate or violating an acceptance
   criterion remains, use `kanban_request_changes` immediately.
4. Only after Coder's native verdict is PASS, run exactly one final independent
   Gemini call over the whole desktop/mobile set:

   ```text
   python3 /home/seb/.hermes/scripts/gemini_review_image.py \
       <desktop.png> <mobile.png> --task-id <current-task-id> \
       --attendu "<observable acceptance result + exact visible change scope; say explicitly when no visible change is expected>"
   ```

   The command prints the evidence path. It reuses a prior PASS when the exact
   screenshot hashes and expectation are unchanged, so retries do not spend a
   second Gemini request.
5. `VERDICT: PROBLEME` means `kanban_request_changes` only when the reported
   problem maps to the stated acceptance result or candidate change scope. The
   final prompt is deliberately scope-bound so a pre-existing, unrelated page
   issue does not become a false rejection. Never edit unrelated UI merely to
   satisfy the visual gate. A temporary failure prints `NEXT_RETRY_AT`. Call
   `kanban_defer_review(reason=..., retry_at=...)`: the dispatcher keeps the
   task in review and resumes this same reviewer session automatically. Use a
   human block only for a non-transient prerequisite that actually needs a
   decision.
6. On `VERDICT: OK`, call `kanban_completion_ready`, then `kanban_complete`
   with:

   ```text
   metadata={
     "visual_review": {
       "coder_verdict": "PASS",
       "gemini_evidence": "<printed evidence path>"
     }
   }
   ```

The completion gate verifies that this run belongs to Coder, that native vision
loaded both exact screenshot hashes, and that the Gemini PASS was produced for
the same files. A prose-only visual claim cannot complete the card.

### 2. Compare requested behavior with delivered behavior

Map every acceptance criterion to concrete implementation or output evidence. Note omissions, changed semantics, and unrelated scope before deciding whether to run deeper checks.

For code work:

1. Use `read_file` and `search_files` to inspect the changed paths and their callers.
2. Use `terminal` to inspect the diff and run the project's existing focused tests, lint, type checks, or build commands.
3. Exercise the reported failure path and at least one ordinary control path when practical.
4. Check error handling, edge cases, concurrency boundaries, data preservation, security boundaries, and cross-platform behavior relevant to the change.
5. Confirm that tests assert behavior rather than merely snapshotting source text or constants.

Repository cleanliness is decided by the canonical `kanban_complete` guard
against the task's recorded spawn baseline. Do not run a raw cleanliness
script without that baseline and then ask the implementer to delete, ignore,
move, or commit foreign dirty paths. A path owned by another task and already
present when this candidate/reviewer spawned is not a candidate defect while
it remains byte-identical. Let `kanban_completion_ready`/`kanban_complete`
evaluate the durable baseline; if the gate itself rejects, report its exact
path and reason.

For non-code work:

1. Inspect the complete deliverable rather than only its summary.
2. Check correctness, completeness, formatting, and provenance.
3. Validate referenced URLs or external facts with the appropriate native tools when they affect the verdict.

### 3. Choose one verdict

#### Approve

Approve only when the acceptance criteria are satisfied and the evidence is sufficient. Call:

```text
kanban_complete(
    summary="Reviewed and approved. <what was verified>",
    metadata={"review_outcome": "approved", "reviewer_checks": [...]}
)
```

Include the exact checks that passed and any bounded caveat that does not block acceptance.

#### Request changes

Use this for specific, correctable defects. First record actionable findings:

```text
kanban_comment(
    task_id="<current-task-id>",
    body="Changes requested:\n1. <file or artifact + defect>\n2. <required correction>",
)
```

Then return the same task to its implementer:

```text
kanban_request_changes(
    reason="<concise summary of the required corrections>"
)
```

State where the defect is, how it reproduces, why it violates the task, and what minimum outcome would resolve it. The transition does not use blocker recurrence accounting.

#### Escalate

Use escalation only when the reviewer and implementer cannot resolve the problem without a human decision or external prerequisite:

```text
kanban_block(
    reason="escalation: <decision or prerequisite required>"
)
```

Explain the blocked decision and the smallest information needed to continue.

### 4. Preserve role separation

Do not edit the implementation while acting as reviewer. Request changes and let the implementer produce the next candidate; then independently verify that candidate in the next review run.

## Pitfalls

- **Rubber-stamping:** A passing handoff summary is not independent evidence.
- **Reviewer implementation:** Editing the deliverable hides ownership and weakens the re-review boundary.
- **Vague findings:** “Needs work” does not give the implementer a reproducible correction target.
- **Style-only blocking:** Do not request changes for preference-level nits when behavior and repository standards are satisfied.
- **Baseline-free visual rejection:** A screenshot can expose a real page issue
  that the reviewed candidate did not introduce. Confirm the exact diff and
  pre-change baseline before returning implementation work; unrelated findings
  are not defects of this card.
- **Baseline-free dirty check:** Never reject a candidate from a raw repository
  check that lacks the task's spawn baseline, and never clean another task's
  screenshots, artifacts, gitlinks, or uncommitted work to make the tree look
  clean.
- **Skipping prior rounds:** Re-review must confirm both the requested corrections and preservation of previously passing behavior.
- **Using blockers for ordinary rework:** Correctable defects belong in `kanban_request_changes`; reserve `kanban_block` for genuine external blockers or human decisions.
- **Completing without evidence:** Every approval summary must name the checks or artifacts actually inspected.

## Verification

Before submitting the verdict, confirm:

- [ ] `kanban_show` was read for the current task and run.
- [ ] Every acceptance criterion was mapped to evidence.
- [ ] The actual deliverable was inspected.
- [ ] Relevant focused checks were run or an explicit reason was recorded when execution was impossible.
- [ ] Prior requested changes were re-tested on re-review.
- [ ] Unrelated regressions and scope changes were considered.
- [ ] The verdict uses exactly one terminal action.
- [ ] The summary contains concrete, non-secret evidence.
- [ ] No implementation files were edited by the reviewer.
