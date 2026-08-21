# 03_REVIEW_GOVERNANCE_LOCK

**Date:** 20 August 2026  
**Status:** `BINDING REVIEW PROCESS`  
**Applies from:** S00 onward

---

# 1. Purpose

This file governs how architecture and implementation stages are reviewed.

Goals:

- genuine cross-vendor independence;
- minimal anchoring;
- minimal review noise;
- mechanical handling of disagreements;
- strict scope control;
- finite review loops;
- no reviewer may silently redesign the study.

---

# 2. Reviewer order

For every major stage:

```text
ARCHITECT / IMPLEMENTER
        ↓
CLAUDE BLIND REVIEW
        ↓
CODEX BLIND REVIEW
        ↓
RECONCILIATION
        ↓
ARCHITECT RESPONSE
        ↓
MECHANICAL ADJUDICATION
```

Claude and Codex must review independently before seeing each other's findings.

---

# 3. Blindness rule

## Claude blind review

Claude receives:

```text
authority documents
stage plan / implementation
diff
tests
raw outputs
```

Claude does not receive Codex findings.

## Codex blind review — Phase 1

Codex receives:

```text
authority documents
stage plan / implementation
diff
tests
raw outputs
```

Codex does not receive:

```text
Claude review
architect self-review
previous adjudication commentary
```

## Codex Phase 2 reconciliation

Only after both blind reviews are frozen, Codex receives Claude's review.

It answers only:

```text
1. What did Claude find that Codex missed?
2. What did Codex find that Claude missed?
3. Which findings overlap?
4. Are any findings duplicates under different wording?
```

It does not create a new full review.

---

# 4. Traceability rule

Every planned directory, command, module, manifest, service, abstraction, workflow, or test lane must cite at least one authority source:

```text
00_MEASUREMENT_SPEC_v1.9_FINAL_CLOSED.md §...
01_EXECUTION_STACK_LOCK_v2.md §...
02_PRE_EXECUTION_IMPLEMENTATION_CORRECTIONS.md C...
03_REVIEW_GOVERNANCE_LOCK.md §...
```

If no authority source exists:

```text
UNJUSTIFIED_SCOPE
```

The reviewer must flag it.

No component is justified merely because it is "best practice."

---

# 5. Finding-validity bar

A `BLOCKER` or `MAJOR` must contain a concrete counterexample expressible as at least one of:

```text
command sequence
file state
configuration state
data-flow path
specific invariant violation
specific reproducibility failure
specific scientific-claim corruption path
```

If the reviewer cannot provide one, severity automatically becomes:

```text
NOTE
```

No eloquent hypothetical may remain a BLOCKER/MAJOR.

---

# 6. Minimal-comment rule

Reviewers must optimize for **signal, not volume**.

Do not output:

```text
praise
style preferences
renaming suggestions
optional refactors
generic best practices
duplicated findings
"consider..." comments with no concrete failure
```

Only report findings that materially affect:

```text
scientific correctness
data leakage
statistical validity
reproducibility
provenance
cache correctness
result integrity
scope/time risk
ability to satisfy an authority requirement
```

### Finding budget

Default maximum per blind review:

```text
BLOCKER: unlimited if real
MAJOR:   max 6
MINOR:   max 4
NOTE:    max 2
```

If more exist, merge duplicates into one root-cause finding.

The target is usually:

```text
0–5 substantive findings
```

not a long comment list.

---

# 7. Scope-creep test

Every reviewer must explicitly answer:

> Is every planned component traceable to an authority requirement?

Any untraceable component becomes:

```text
UNJUSTIFIED_SCOPE
```

Severity:

```text
MAJOR
```

if it adds significant implementation/maintenance burden;

```text
MINOR
```

if trivial.

---

# 8. Hardware-unknown rule

A plan must not fabricate values requiring real hardware.

Use:

```text
TBD_REQUIRES_HARDWARE
```

for values such as:

```text
exact CUDA version
actual PyTorch/CUDA compatibility choice
batch size
throughput
VRAM peak
H100-specific environment result
```

The plan must specify:

```text
procedure that determines the value
artifact/file that records the value
acceptance test
```

---

# 9. Read-only enforcement

A reviewer's read-only state must be enforced operationally, not only requested.

Preferred:

```text
separate git worktree
pinned commit
filesystem read-only where practical
no production write permissions
```

Reviewer-generated diagnostics go only to:

```text
reviews/<stage>/scratch/
```

and never become production source automatically.

---

# 10. Architect response rule

The architect/implementer must respond to every unique finding with exactly one status:

```text
FIX
REJECT
DEFER
```

## FIX

Must include:

```text
specific plan/code delta
affected files
test proving closure
```

## REJECT

Must include:

```text
specific authority citation
or
specific technical evidence
showing the finding is wrong
```

## DEFER

Allowed only for non-BLOCKER findings.

Must include:

```text
ticket ID
reason
target stage
reviews/<stage>/deferred.md
```

A BLOCKER cannot be deferred.

---

# 11. Mechanical adjudication rule

The adjudicator does not choose a preferred technical opinion.

It checks process closure only.

Rules:

```text
Any unresolved BLOCKER -> BLOCKED

Severity disagreement between blind reviewers ->
use the HIGHER severity until resolved by FIX or REJECT evidence

Finding without required architect response ->
INCOMPLETE

REJECT without authority citation / technical evidence ->
INCOMPLETE

DEFER without ticket ->
INCOMPLETE

All findings coherently answered and no unresolved BLOCKER ->
CLEARED
```

Output exactly:

```text
SXX_ADJUDICATION = CLEARED | BLOCKED | INCOMPLETE
```

The adjudicator checks:

```text
coherence
responsiveness
evidence presence
authority traceability
```

It does not invent a fourth technical opinion.

---

# 12. Review-loop termination

Maximum:

```text
2 review rounds per stage
```

## Round 1

Blind Claude + blind Codex.

Fix/reject/defer.

## Round 2

Review only the changed material and unresolved Round-1 findings.

After Round 2:

```text
unresolved BLOCKER -> stage BLOCKED
unresolved MAJOR   -> stage BLOCKED unless formally REJECTED with evidence
new MINOR/NOTE     -> deferred ticket
```

Do not start Round 3 for non-BLOCKER findings.

This prevents review recursion.

---

# 13. Plan-length cap

`STAGE_PLAN.md` must be:

```text
<= 1,200 lines
```

If longer:

```text
PLAN_TOO_LARGE
```

and it must be reduced before review.

A plan longer than 1,200 lines is treated as implementation disguised as planning.

---

# 14. Requirement-source rule

Anything introduced only in a task prompt and absent from authority documents must be marked:

```text
PROMPT_ONLY_REQUIREMENT
```

The architect must either:

```text
cite the authority that actually mandates it
or
exclude it
```

This specifically protects against an execution prompt silently changing:

```text
primary privacy endpoint
statistical estimator
data split
model family
attack definition
review rule
```

---

# 15. Overlap diagnostic

After blind reviews, compute:

```text
CLAUDE_ONLY findings
CODEX_ONLY findings
OVERLAP findings
```

Do not interpret high overlap as proof of correctness.

Use overlap only as a process diagnostic.

If both reviewers independently miss a known planted architecture/control defect in a review benchmark:

```text
REVIEW_PROCESS_FAIL
```

and the stage cannot clear.

---

# 16. Final stage rule

The review system exists to remove high-impact mistakes, not to maximize comments.

The desired terminal state is:

```text
few findings
all concrete
all traceable
all closed
no unresolved BLOCKER
no unjustified scope
```
