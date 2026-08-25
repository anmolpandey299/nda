# 04_EXECUTION_ACCELERATION_AMENDMENT

**Project:** Privacy Leakage / Recoverability in Model Merging
**Date:** 25 August 2026
**Status:** `BINDING ENGINEERING / REVIEW CADENCE AMENDMENT`
**Unchanged authority:** `00_MEASUREMENT_SPEC_v1.9_FINAL_CLOSED.md`, `02_PRE_EXECUTION_IMPLEMENTATION_CORRECTIONS.md`
**Amends:** `01_EXECUTION_STACK_LOCK_v2.md` §40, §50 cadence and `03_REVIEW_GOVERNANCE_LOCK.md` §2, §12 — for `S01`–`S10` only
**Applies from:** `S01`. `S00` is CLOSED at `db9ed078702804b6aa581d74910c6f0e902b5d29` and is not reopened.

# 1. Purpose

The per-stage ceremony of `01 §40`/`§50` — architecture → implementation → blind Claude
review → blind Codex review → reconciliation → architect response → adjudication → tag,
repeated ten times — cannot be executed inside the remaining experimental calendar. For
`S01`–`S10` it is replaced by accelerated component-level implementation and review, which
changes **engineering and review cadence only**. The experiment stays frozen.

# 2. Frozen scientific surface — untouched by this amendment

```text
RQ1, RQ2                                        00 §1
hypotheses H1–H4, falsification outcomes        00 §38, §39
threat model, attacker observation, recovery    00 §3.6, §3.7
protected objects PO1–PO3                       00 §4
attacks — base family and pooled lineage        00 §14, §15
datasets, partitions, split/freeze semantics    00 §6, §33
canary design, inclusion, membership            00 §7
M_PRIMARY = common-FPR TPR@1%FPR                00 §16; 02 §C1
tie-safe fixed-FPR estimator                    02 §C2, §C8
privacy-endpoint eligibility                    00 §18
R_priv, R_func                                  00 §19, §23A
D_L, G_L, D_R, G_R                              00 §20.4–§20.7
statistical thresholds and tolerances           00 §34B.1A; 02 §C3–§C5
dry-run assertions DRY-A…H, DRY-CACHE           00 §34B.1; 02 §C4
model panel and pinned revisions                01 §8, §8G
LoRA rank / common target policy                00 §8.3; 01 §8B
merge operators O1–O3, TIES status              00 §10
recovery baselines C1, C2, C2B, C3, C4          00 §25
DP boundary, DP regime, DP audit, DP arm        00 §5, §8.2, §21, §37
gates and stop rules                            00 §24, §24A, §28, §41
no-result-peeking boundary                      01 §29
evidentiary requirements                        01 §16, §26, §35, §36
```

No number, definition, threshold, gate, ordering or stop rule in `00` or `02` is altered,
relaxed or reinterpreted here.

# 3. Authority basis

```text
00 §34C.1   21–27 Aug 2026 is a NON-COMPRESSIBLE P0-PRE software window;
            P1_LAUNCH_DEADLINE = 2026-09-03T18:00:00+01:00 does not slide
00 §34C.3   CALENDAR_COMPRESSED_MVRS is entered when implementation progress
            shows the dated schedule cannot protect the MVRS
01 §47      an execution-lock change is permitted for actual implementation
            impossibility — the impossibility relied on here
```

The compression applies to **ceremony**, never to **gates**. The `00 §34C.1` `SOFTWARE_SLIP`
rule stays binding in full: nothing below permits skipping a dry-run scenario, weakening a
tolerance, removing a negative control, bypassing cross-fit validation, disabling cache
invalidation, or opening a real evaluation outcome to preserve a date.

# 4. Precedence

```text
00 and 02      win over this document in every conflict
this document  wins over 01 §40, 01 §50 cadence and 03 §2, §12 loop shape, S01–S10 only
01 and 03      otherwise remain in force verbatim
```

# 5. Technical dependency order retained

```text
S01 → S02 → S03 → S04 → S05 → S06 → S07 → S08 → S09 → S10
```

exactly as `01 §39` and `01 §50`. Blocks group stages for **review cadence**; they never
authorise out-of-order implementation, and no stage's code is used by a later stage before
its block review closes.

`S02B MODEL_COMPATIBILITY_GATE` (`01 §8C`) is unchanged and still gates any real membership
result: its thirteen checks close across Blocks B–D, the whole gate by the pre-data freeze.

# 6. Implementation blocks

```text
BLOCK A   S01 provenance/config + S02 deterministic fixtures
          config resolver, SHA256 utils, model/data/run manifests, RUN_ID, artifact hashing,
          immutable run directory, tiny fixtures/corpus/LoRA  [AUTH: 01 §13–§17, §36, §39]

BLOCK B   S03 canonical scorer/cache/cross-fit + S04 statistical engine
          parameterised scorer, reference loss, Min-K%, content-addressed cache, cross-fit
          controller, pooled transforms, eligibility, R_priv, R_func, D_L/D_R, G_L/G_R,
          isotonic, bootstrap, e50, residuals, DRY-A…H  [AUTH: 00 §34A, §34B; 01 §39; 02 §C1–§C7]

BLOCK C   S05 data/canary pipeline + S06 training/DP plumbing
          acquisition, normalisation, dedup, splits, canaries, blind split/dedup check,
          manifesting; base loader, LoRA r=32, three seeds, no-canary control, DP smoke,
          checkpoint/adaptor manifests  [AUTH: 00 §6, §7, §8, §9, §33; 01 §30, §39]

BLOCK D   S07 merge + S08 recovery
          linear/task arithmetic, DARE, conditional O3 SVD truncation; C1 exact inversion,
          C2, C2B, conditional O3 recovery, conditioning  [AUTH: 00 §10, §24–§26; 01 §39]

BLOCK E   S09 registry/orchestrator + S10 integrated H100 benchmark
          P0/P1 registry as data/config, launcher invents no cells; 1,000-sequence
          benchmark, throughput/batch/memory/projections frozen  [AUTH: 00 §0.2, §36; 01 §39]
```

# 7. Per-block workflow

```text
IMPLEMENT
   → DETERMINISTIC TESTS
   → ONE INDEPENDENT CODE REVIEW
   → FIX VALID BLOCKER/MAJOR FINDINGS
   → RE-RUN RELEVANT + REGRESSION TESTS
   → CONTINUE
```

**DETERMINISTIC TESTS** — the `01 §22` lanes in authority order; an empty lane reports
`NOT_RUN(<reason>)`, never `PASS`.

**ONE INDEPENDENT CODE REVIEW** — one reviewer session that did not write the code,
read-only from a separate worktree or pinned commit, writing only to `reviews/<block>/`
[AUTH: 01 §3.3; 03 §9]. The `03 §5` validity bar and `03 §6` comment budget apply unchanged.

**FIX VALID BLOCKER/MAJOR FINDINGS** — every unique finding gets exactly one of
`FIX`/`REJECT`/`DEFER` with the `03 §10` evidence; a `BLOCKER` cannot be deferred. Validity
is decided by §9.

**RE-RUN RELEVANT + REGRESSION TESTS** — the lanes the fix touches plus every previously
passing lane the change can reach.

**CONTINUE** — proceed to the next component. No adjudication ceremony between components.

# 8. Second-review trigger

A second independent review of a block occurs **only** when at least one holds:

1. a fix materially changed a scientific mechanism — scorer, cross-fitting, ROC/fixed-FPR
   estimator, cache identity, split or canary membership, merge or recovery mathematics,
   DP accounting, statistical estimator or threshold;
2. a `BLOCKER` or `MAJOR` remains unresolved after the fix pass.

Do not automatically trigger another review. A clean re-run is not a trigger. A reviewer
wishing to look again is not a trigger.

Maximum two review rounds per block [AUTH: 03 §12]. After round 2: unresolved `BLOCKER` →
block `BLOCKED`; unresolved `MAJOR` → `BLOCKED` unless formally `REJECT`ed with evidence;
new `MINOR`/`NOTE` → deferred ticket in `reviews/<block>/deferred.md`.

# 9. Review relevance rule

A finding is **blocking** only if it can plausibly change:

```text
1. a reported scientific number
2. an experimental condition
3. privacy or statistical validity
4. train / calibration / evaluation separation
5. model, data, config or environment attribution
6. result reproducibility
7. merge or recovery semantics
8. DP accounting or interpretation
9. an explicit frozen scientific requirement
```

```text
BLOCKING EXAMPLES
wrong ROC estimator                 held-out leakage
broken cross-fitting                stale cache
wrong model/config/data attributed  wrong LoRA target
wrong merge coefficient             recovery uses unavailable hidden information
bootstrap destroys pairing          wrong canary membership
incorrect DP accounting

MUST NOT STOP EXECUTION BY THEMSELVES
naming preference                   optional refactor
abstraction/generalisation request  more elegant API
dashboard                           stage-management framework
global code-coverage target         cosmetic documentation change
speculative future-proofing
```

A finding failing the nine-point test is recorded at most as `MINOR`/`NOTE` and deferred.
`03 §5` still applies: without a concrete counterexample, severity is `NOTE`.

# 10. Anti-platform rule

Do not build:

```text
stageflow software                    research dashboards
autonomous multi-agent orchestration  generic workflow platforms
experiment-management services        unnecessary database/service infrastructure
custom model-serving infrastructure — unless the S10 benchmark demonstrates it is necessary
```

The repository exists to execute the experiment. Any such component is `UNJUSTIFIED_SCOPE`
[AUTH: 03 §7] and is out of scope by `01 §49`.

# 11. Formal freeze points

Global adjudication happens here, not after every engineering stage. Each freeze point
requires the full `01 §26` field set, the `01 §45` package, independent review, and a verdict.

```text
11.1  PRE-DATA CODE FREEZE
      Blocks A–E closed; S02B MODEL_COMPATIBILITY_GATE closed (01 §8C);
      SCORER_ENGINE = VERIFIED, ANALYSIS_DRY_RUN = PASS,
      CROSSFIT_NEGATIVE_CONTROL = PASS, CACHE_ASSERTIONS = PASS (00 §35);
      00 §34B.2 artifacts written; pre-data items of the 00 §42 checklist confirmed;
      02 §C6 flags evaluated. Only then may a real evaluation outcome be opened.
11.2  P0-A MEASUREMENT-POWER GATE
      00 §28 eligibility per arm; §28.3 STOP_PRIVACY_GRID / §28.4 limited continuation.
11.3  MVRS / P0 RESULTS FREEZE
      00 §34C.2 completion status; 00 §43 artifact set complete.
11.4  P1 LAUNCH GATE — only if P1 is activated
      00 §34C.1 hard drop-dead; 00 §36 registry; 00 §40 expansion rules.
11.5  FINAL RESULTS FREEZE
      00 §34C.1 empirical-results freeze; no new experiment family after this point.
11.6  REPRODUCIBILITY RELEASE
      environment lock, run manifests, failed-run preservation (01 §12, §16, §36).
```

Individual components still receive independent code review before scientific use.

# 12. Result-peeking rule — preserved unchanged

Do not use real privacy outcomes while implementing or choosing:

```text
attack   scorer   threshold   pooling   recovery   operator   statistical method
```

Real outcome inspection begins only after the relevant pre-data code is frozen [AUTH: 01
§29; 00 §34B.3, §34C.1]. A peek discovered after the fact is recorded as `SPEC_DEVIATION` /
`IMPLEMENTATION_REPAIR`. Accelerated cadence is never a reason to open an outcome early.

# 13. H100 rule

Use the H100 for:

```text
genuine GPU compatibility   training   DP-LoRA
merge/recovery at research scale   scoring   benchmark   evidentiary experiments
```

Do not spend H100 time on ordinary coding, review, documentation or platform work
[AUTH: 01 §6.1, §9, §49].

# 14. Evidence requirements retained

Cadence changes; evidence content does not. Each block produces the `01 §26` fields and the
`01 §45` layout once per **block** instead of once per stage, under `stage_acceptance/<block>/`.
Unchanged: run manifests (no valid manifest → `NON_EVIDENTIARY`, `01 §16`), immutable run
directories and failed-run preservation (`01 §36`), branch/tag rules (`01 §27`), and `01 §46`.

# 15. Review governance — retained and suspended

```text
retained   03 §5 validity bar   03 §6 comment budget   03 §7 scope-creep test
           03 §9 read-only      03 §10 architect response   03 §13 plan cap
           03 §14 requirement-source rule   03 §16 terminal state
suspended  03 §2 blind two-vendor pair + reconciliation, for S01–S10 blocks
           03 §15 overlap diagnostic, for S01–S10 blocks
           01 §40 per-stage Codex list and per-stage adjudication list
restored   in full at every §11 freeze point
```

# 16. What this amendment does not authorise

```text
weakening any P0-PRE software gate to hold a date     00 §34C.1
a second scoring / cross-fit / fixed-FPR path         00 §34A.2, §34A.5; 02 §C2
material constants in source instead of configs/**    01 §17
a notebook as an execution path                       01 §18
an evidentiary run from a dirty production tree       01 §35(3), §36
overwriting a failed run directory                    01 §36
commits to main / experiment-frozen; history rewrite  01 §27
drive-by refactors outside the active block           01 §27, §41
research-model substitution                           01 §8C–§8G
result-driven tuning of any threshold                 01 §3.2; 00 §34B.1A
```

# 17. Requirement-source record

Introduced by architect instruction on 25 August 2026 under `01 §47` and `00 §34C.3`; not by a
reviewer request, a model suggestion, or an inconvenient result. It introduces no scientific
requirement, so the `03 §14` protected list — primary privacy endpoint, statistical estimator,
data split, model family, attack definition — is untouched.

**Registration note.** This document is deliberately absent from `specs/SPEC_HASHES.json`:
that file's key set is asserted by the closed S00 acceptance test
`tests/unit/test_s00_invariants.py::test_a2_spec_integrity`, and S00 evidence is not
reopened. Only `spec_sha256` (`00`) and `execution_lock_sha256` (`01`) are RUN_ID inputs
[AUTH: 01 §15], so this document's hash is not an evidentiary input. Registering it is a
separate reviewed change if the architect wants one.
