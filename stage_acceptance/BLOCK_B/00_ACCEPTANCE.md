# Block B Acceptance

## 1. Identity

```text
BLOCK                         = BLOCK_B
STAGES                        = S03 (scorer/cache/cross-fit) + S04 (statistical engine)
BLOCK_A_IMPLEMENTATION_COMMIT = dcee3c7df4ee5508c0735100c20540bf2e742ce1
BLOCK_B_IMPLEMENTATION_COMMIT = 22e6747cef87556a1d7b6c3e2ac36d7994c44586
                                "feat(block-b): freeze scoring and statistical measurement engine"
BRANCH                        = stage/predata-accelerated
HEAD                          = 22e6747cef87556a1d7b6c3e2ac36d7994c44586
DATE                          = 2026-08-26
```

## 2. Bounded objective

Block B implements and freezes the pre-data statistical/measurement engine. No real record,
model, outcome or GPU was involved.

```text
S03  canonical scorer interface · reference-calibrated loss · Min-K% ·
     canonical fixed-FPR estimator · cache · cross-fitting ·
     natural permanent holdout · pooled-lineage scoring
S04  privacy eligibility · R_priv / R_func · D_L / D_R · G_L / G_R ·
     paired endpoint bootstrap · seed-balanced isotonic analysis · e50 ·
     operator residual analysis · DRY-H · DRY-H4 · DRY-G · positive controls
```

## 3. Authority

```text
specs/00_MEASUREMENT_SPEC_v1.9_FINAL_CLOSED.md       scientific authority
specs/01_EXECUTION_STACK_LOCK_v2.md                  engineering authority
specs/02_PRE_EXECUTION_IMPLEMENTATION_CORRECTIONS.md binding clarifications
specs/03_REVIEW_GOVERNANCE_LOCK.md                   binding review process
specs/04_EXECUTION_ACCELERATION_AMENDMENT.md         engineering/review cadence
```

`04` changes implementation and review **cadence only** — no frozen scientific question,
estimand, threshold, split, attack, operator or gate [AUTH: 04 §2, §4].

## 4. Frozen primary endpoint

```text
M(V) = ROC-interpolated TPR at exactly FPR = 0.01, from held-out scores
       [AUTH: 00 §7.4(7), §16, §34A.2(8); 02 §C1, §C2]

TIE POLICY
  operating points exist only at DISTINCT score values
  every observation sharing an exact score is admitted atomically
  no arbitrary ordering exists inside a tie group
  the attainable threshold, its realised FPR and its attainable TPR are reported apart
    from the interpolated M(V), so no TPR is attributed to a nonexistent threshold

M0     = 0.01
R_priv = [M(V) - M0] / [M(A) - M0]     UNCLAMPED, eligible conditions only
```

## 5. Final scientific invariants

```text
one canonical fixed-FPR implementation; every view, arm, replicate and DRY test calls it
score direction is explicit: higher score = more member-like
reference score s_ref = loss_reference - loss_target [00 §14.1]
Min-K% is the declared secondary score family [00 §14.2]
natural calibration and evaluation partitions are permanently disjoint; no rotation
canary outer cross-fitting makes every selection, threshold and fit calibration-only
each fold's held-out scores are mapped onto a common scale from CALIBRATION statistics
no evaluation label reaches fitting, selection, pooling or thresholding
cache identity binds score family, reference backend, artifact, revision, tokenizer, ordered
  record set, precision, max sequence length, code hash, config and environment [02 §C7]
the resolved config is a canonical immutable snapshot; digest and constants come from that
  one object, and a config-A / SHA-B pairing is refused
normalized scientific reporting is eligibility-gated; the raw algebra is private
raw D_L and D_R remain the primary within-cell statistics [00 §20.4, §20.6]
the paired bootstrap recomputes the fixed-FPR endpoints inside every replicate
isotonic analysis is seed-balanced over EXACT registered rungs; no re-binning
no e50 is extrapolated beyond the calibrated support
the operator residual baseline is disjoint from the operator points judged against it
```

## 6. Final test evidence

```text
UNIT        = 617 passed          FORMAT     = PASS
INTEGRATION = 240 passed          LINT       = PASS
SYNTHETIC   =  60 passed          TYPECHECK  = PASS (mypy strict)
                                  INVARIANTS = 0 violations
TRUE_EFFECT_DETECTABLE       = TRUE
NULL_EFFECT_NOT_MANUFACTURED = TRUE
LEAKAGE_PATH_DETECTED        = TRUE
```

## 7. DRY-H4 — fixed-FPR estimator calibration [AUTH: 02 §C3]

```text
sample sizes         = 200, 400, 800, 2048, 5000
simulations per size = 1000 fixed seeds
analytic TPR@1%FPR   = 0.372081
```

The frozen calibration completed successfully: bias, absolute error, RMSE and the 2.5/50/97.5
percentiles are recorded at every size, the tied-score stress agrees with an independent
distinct-threshold reference, and a null separation manufactures no power.

## 8. DRY-G — family-wise null calibration

```text
GATING_AUTHORITY   = 02 §C4
N_FWER_CALIBRATION = 2000
c_FWER             = 2.673644675629078

810200 = PASS      810201 = PASS      810202 = FAIL
DRY_G              = PASS_WITH_INVESTIGATION

810202 empirical T_max percentile = 96.15%
810202 within empirical support   = TRUE
implementation bug identified     = FALSE
DRY_G_INVESTIGATION               = CLOSED
```

The 00 §34B.1 200-trial marginal bank is `NON_GATING_DIAGNOSTIC`: it decides nothing.

## 9. Positive controls [AUTH: 02 §C5]

```text
DRY-B regenerated threshold = 0.237151     DRY-B out-of-bank detection = 0.980
DRY-H1 regenerated bound    = 0.066945     DRY-H1 out-of-bank success  = 0.990
```

Both banks were regenerated under the final estimator. The in-bank percentile is arithmetic,
not power — the figures above are measured on held-out seeds outside the bank.

## 10. Configuration

```text
ACTIVE_MATERIAL_CONFIG_FIELDS = 53   (7 scoring · 10 analysis · 36 dry-run)
INERT_ACTIVE_CONFIG_FIELDS    = 0
```

Every material scientific constant is version-controlled configuration under `configs/**`
[AUTH: 01 §17]; each of the 53 fields is proven by mutation to control its owning path, and
the canonical snapshot is built once, is immutable, and supplies both semantics and digest.

## 11. Backend boundary

```text
BACKEND_INTEGRATED = FALSE
SUITE_SCOPE        = STATISTICAL_STACK_ONLY
P0_PRE_READY       = FALSE
```

Block B runs on deterministic synthetic backends. The production backend must still validate
actual precision and max-sequence/truncation binding, the backend contract, a synthetic-suite
rerun, cache assertions, H100 smoke and the 1,000-sequence benchmark [AUTH: 02 §C6; 00 §34C.1].
These are **deferred**, not waived, requirements.

## 12. Dependency-deferred item

```text
DRY_D_STATUS = NOT_RUN_DEPENDENCY(S07_S08)
```

DRY-D reads Δ_tail_priv over O3 SVD-truncation merges and their recoveries, which are S07/S08
objects [AUTH: 00 §34B.1 DRY-D, §28B]. This does not reopen Block B; it must run once those
dependencies exist, before final P0-PRE readiness.

## 13. Review history

Block B underwent initial implementation, independent adversarial reviews, a bounded repair
round, changed-surface verification and final architect adjudication [AUTH: 04 §7, §8].
Findings affecting **cache identity, natural holdout, cross-fold scaling, paired bootstrap,
isotonic rungs, config binding, normalized-reporting guards, positive controls, DRY-H4 and
DRY-G** were resolved before acceptance.

```text
NO FURTHER BLOCK B GENERAL REVIEW IS REQUIRED.
```

## 14. Safety and exposure

```text
REAL_SCIENTIFIC_DATA_INSPECTED  = FALSE
REAL_PRIVACY_OUTCOMES_INSPECTED = FALSE
H100_USED                       = FALSE
S00_EVIDENCE_PRESERVED          = TRUE
BLOCK_A_PRODUCTION_PRESERVED    = TRUE
```

No P0-PRE artifact was generated; the `01 §29` / `04 §12` peeking boundary is intact.

## 15. Scope boundaries

Block B acceptance does **not** certify:

```text
S05 real data acquisition, splits or canaries    S06 LoRA / DP training
S07 merge implementation                         S08 recovery implementation
production HF/PEFT backend                       H100 performance
P0 or P1 scientific findings                     the research hypotheses themselves
```
It certifies the **pre-data statistical and scoring measurement implementation**.

## 16. Verdict

```text
BLOCK_B_STATUS=ACCEPTED
BLOCK_B_IMPLEMENTATION_COMMIT=22e6747cef87556a1d7b6c3e2ac36d7994c44586
UNRESOLVED_BLOCKER=NONE
UNRESOLVED_MAJOR=NONE
REAL_SCIENTIFIC_DATA_INSPECTED=FALSE
REAL_PRIVACY_OUTCOMES_INSPECTED=FALSE
H100_USED=FALSE
BACKEND_INTEGRATED=FALSE
P0_PRE_READY=FALSE
DRY_D_STATUS=NOT_RUN_DEPENDENCY(S07_S08)
NEXT_BLOCK=BLOCK_C
NEXT_BLOCK_SCOPE=S05 data/canary + S06 training/DP
```
