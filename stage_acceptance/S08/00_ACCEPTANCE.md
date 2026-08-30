# S08 Acceptance — Recovery

## 1. Identity and scope

```text
STAGE_ID = S08 · STAGE_NAME = RECOVERY · STATUS = ACCEPTED
S08_IMPLEMENTATION_COMMIT = 2fadc5a5eec4ac800288d867982b9f5842ef5ae2
                            "feat(s08): freeze recovery engine and dry-d calibration"
BRANCH = stage/s08-recovery · DATE = 2026-08-30T13:24:09+01:00
BASE_ACCEPTED_STATE    = stage-s07-v1
BASE_ACCEPTANCE_COMMIT = eaa43015684d8980944a1b536db7c3ed916d5f2a

implemented      C1 known-partner/known-coefficient exact inversion ·
                 C2 Spectral-DeTuning-style shared-source / low-rank-residual reproduction ·
                 C2B known-varying-alpha rescaling into the same C2 core ·
                 O3 conditional truncated-protected-source recovery ·
                 direction-matrix conditioning · parameter-recovery evaluation ·
                 DRY-D dependency activation + corrected pre-data DRY-D1 calibration
not implemented  hidden-alpha recovery · DARE incomplete-lineage recovery · TIES recovery ·
                 S09 experiment registry · real privacy attacks · real model experiments
authority        00 §11, §12, §22, §24A, §25, §26, §28B, §34B.1, §34B.1A, §34B.2 ·
                 01 §10, §12, §16, §17, §27, §29, §39 · 02 §C5 · 03 · 04
```

S08 is deterministic pre-data recovery engineering on synthetic fixtures: no PubMed record
fetched, no research model downloaded or trained, no privacy outcome inspected, no GPU used.

## 2. C1 — the oracle control

```text
A_hat = [C_delta - (1 - alpha) B_delta] / alpha        METHOD_ROLE = ORACLE_CONTROL
```
Scientific artifact arithmetic is float32; derived scalars accumulate in float64. The
inversion is exact to float32 rounding — e_F ≈ 5.75e-08 at α = 0.35; across the registered
sweep α ∈ {0.25, 0.35, 0.50, 0.65, 0.90, 1.00} it ranges from 7.64e-08 down to exactly 0 at
α = 1, every value satisfying the frozen criterion e_F ≤ 1e-5. α is read off the issued
descendant, never accepted from a caller. Public C1 accepts only an issued S08 observation
originating from a genuine S07 object; a wrong partner, coefficient, operator, surface or
execution context each reject.

## 3. C2 — shared-source reproduction
```text
C2_METHOD            = SPECTRAL_DETUNING_CORE_FIXED_RANK_REPRODUCTION
C2_N_ITERS           = 1000        C2_RANK_SCHEDULER = DISABLED_FIXED_STRUCTURAL_RANK
LINEAR_RESIDUAL_RANK = 32

S(0) = mean_i X_i
repeat exactly 1000 times:   R_i = X_i - S      L_i = T_r(R_i)      S = mean_i(X_i - L_i)
```

For fixed-alpha linear descendants D_i = α·A + (1−α)·B_i the common source is α·A, so the
method divides **once**: A_hat = S_hat / α. No adaptive stopping rule, no truth-based
optimisation, and no claim to a new common-source or low-rank-inverse method — this is an
occupied-regime reproduction implemented independently from the published description. The
raw numerical core is module-private (`_spectral_detuning_core`); every public scientific
entry point requires a factory-issued `ObservedLineage`, so no caller reaches the mathematics
with raw matrices.

## 4. Descendant-count regimes
```text
k = 1 (unknown partner)   STRUCTURAL_NA — refused at observation issuance, not solved badly
k = 2                     VALID_EVALUATED_CONDITION
k = 4                     VALID_EVALUATED_CONDITION

measured on the planted recoverable fixture:
k = 2   spectral e_F ≈ 1.645600     naive shared mean e_F ≈ 1.161387
k = 4   spectral e_F ≈ 0.000000     naive shared mean e_F ≈ 0.833364
```

The k = 2 result is a **fixture-specific observation**: on this fixture the fixed-rank C2
objective settles on a solution that is not the planted source, and the solve does not beat
the naive mean. S08 asserts no theorem about two-descendant families, and none exists in the
frozen authority. No seed or fixture was retuned during review; both numbers survive the
review repairs unchanged.

## 5. C2B — known varying alpha

After preconditioning on the public coefficients the common source is A itself, so **no
second divide** follows. C2B calls the identical frozen C2 core at the same
`c2_method_version`, introducing no optimiser and estimating no coefficient.

```text
Dtilde_i = D_i / alpha_i   beta_i = (1 - alpha_i)/alpha_i   Dtilde_i = A + beta_i·B_i
k = 2 reference {0.35, 0.65} · k = 4 matched {0.35, 0.45, 0.55, 0.65} ·
k = 4 wide {0.25, 0.40, 0.60, 0.75}
```

## 6. The naive comparator

`NAIVE_UNRESCALED_FIXED_ALPHA` takes `assumed_fixed_alpha` keyword-only, with no default and
no inference — 00 §25 names the comparator without freezing a coefficient-estimation rule, so
none was invented, and the configured value is `REQUIRED_NOT_CALIBRATED` and fails closed on
read. The coefficient is bound into the result's method parameters, serialized provenance and
scientific identity, so α = 0.4 and α = 0.6 are distinguishable before their bytes are
compared.

## 7. Attack-information firewall

Observation factories consume genuine issued S07 objects — `MergeResult`, `ReleaseFamily`,
`TaskVector` — checked by type, never by duck typing. Incomplete-lineage observations carry no
slot for the protected constituent's bytes or identity, the hidden partner's bytes or
identity, or any truth metric: they are never stored, not stripped at read time.

Public `recover_*` functions **independently** require a factory-issued `ObservedLineage`
before reading any scientific field. A duck-typed impostor is rejected even when it declares
the correct regime for the function it attacks, carries the true protected constituent as its
updates, and returns a genuine observation's SHA from `identity()`.

A `ReleaseFamily` is canonically ordered by descendant id before the attacker view is built,
so the same logical lineage in any caller order yields the same observation R and bitwise
identical recovered bytes. Compared methods read one issued observation; none builds its
own.

## 8. Truth binding

`EvaluationTruthBinding` is opaque, factory-issued, immutable and evaluator-only. Its factory
re-observes the genuine S07 source through the same S08 factories, requiring the resulting
observation identity to equal the one the recovery ran on and the supplied protected
constituent's content identity to equal the identity the S07 source recorded.

`RecoveryResult` holds no protected truth. Evaluation requires
`result.observation_identity == binding.observation_identity`. Evaluating an A₁ recovery
against an unrelated A₂, binding an unrelated source family, and pairing an O3 result at rank
s₁ with a binding at s₂ or an unrelated floor family all reject. No solver accepts a binding,
and `src/recovery/solvers.py` imports no truth, evaluation or conditioning module.

## 9. The recovery result

Opaque, non-dataclass, factory-issued, immutable, content-derived. Authoritative tensors are
an immutable tuple of `(name, shape, dtype, bytes)` entries; every mapping a property returns
is freshly built and every array view read-only, so no ordinary write-back can change
scientific content while a stored identity keeps claiming the old bytes. Provenance binds
method and version, observation identity, recovery context identity, parameter surface, dtype,
algorithm parameters, method-specific parameters and the recovered float32 bytes.
`provenance_class = NON_EVIDENTIARY_S08_RECOVERY`, derived from the class and the context
profile; S09 binds run-level provenance and nothing here self-promotes.

## 10. Recovery configuration

The context factory takes **one deep canonical snapshot** of the supplied document; every
parsed setting and `config_sha256` derive from that same snapshot, so a Mapping whose repeated
reads differ cannot make the executed settings and the recorded hash describe different
documents. Frozen scientific production settings:

```text
C2_N_ITERS = 1000 · LINEAR_RESIDUAL_RANK = 32 · CONDITIONING_TOLERANCE = 1e-6
SCIENTIFIC_ARTIFACT_DTYPE = float32 · C2_RANK_SCHEDULER = DISABLED_FIXED_STRUCTURAL_RANK
```

A document claiming `s08.recovery.v1` while changing any of them describes a different method
and is refused rather than hashed under the accepted name. Planted-truth scenarios use
`fixture_recovery_context`, stamping `FIXTURE_ONLY_NOT_SCIENTIFIC` on the context and on every
result issued under it. The C1 criterion (1e-5) and partner norm-match limit (1.25) are
context-bound with no caller-overridable parameter.

## 11. O3 — conditional truncated-source recovery
```text
O3_RECOVERY_IMPLEMENTED = TRUE · O3_RECOVERY_AUTHORIZATION = CONDITIONAL_NOT_RUN
TARGET = TRUNCATED_PROTECTED_SOURCE
```

At fixed α the common source is α·T_s(A) and the residual rank is the public retained rank s,
read from issued merge metadata rather than chosen. At known varying α,
D_i/α_i = T_s(A) + β_i·T_s(B_i), through the same C2 core with no second divide. DARE
incomplete-lineage recovery remains `METHOD_GATED` — the observation factory refuses a DARE
family before any solver sees it.

### O3 error decomposition

```text
e_F                = ||A_hat - A||      / ||A||
e_floor(s)         = the frozen S07 protected-information floor for that truncation
e_solver_origscale = ||A_hat - T_s(A)|| / ||A||
e_solver_retained  = ||A_hat - T_s(A)|| / ||T_s(A)||
```

`e_F − e_floor` is **never** computed or reported as solver error: different quantities on
different denominators. No key of the report carries a difference.

## 12. Conditioning

For known-partner conditions D_i = vec(C_i − B_i) and D = [D_1 … D_k]. For k = 2,
κ = σ₁/σ₂ where numerically non-zero; for k > 2, κ = σ_max/σ_min,nonzero. The numerical-rank
criterion is the project's already-frozen σ_j/σ₁ ≥ 1e-6, not a second invented one.
Rank-deficient families report `UNDEFINED_RANK_DEFICIENT` explicitly and never emit an
ordinary Inf or NaN. Conditioning is explanatory — not a recovery input, not a novelty claim —
and is unreachable from an incomplete-lineage observation, which holds no partner.

## 13. DRY-D — authorized Block-B dependency activation

No unrelated Block-B production changed. DRY-D now runs the complete authoritative 00 §28B.1
ladder:

```text
BLOCK_B_DEPENDENCY_ACTIVATION = TRUE
authorized changes: src/analysis/dryrun.py · configs/p0/dry_run.json · their tests
anchors  32 (oracle) · 0 (zero-delta base)
interior 31, 28, 24, 20, 16, 12, 8, 6, 4, 3, 2, 1                    interior count = 12
```

The generator fixture was enlarged to a 40×36 surface with a rank-32 protected constituent, so
every registered rank through 32 is representable and ranks 31 and 28 are genuine truncations
rather than clamps against a narrower dimension.

## 14. DRY-D1 pre-data recalibration

The historical cutoffs — reference -0.136213, Min-K% -0.135921 — had no reproducible
in-repository calibration provenance for the corrected twelve-rank statistic. Under binding
clarification 02 §C5, DRY-D1 positive-control calibration was regenerated **pre-data**.

**This was not changed because validation failed.** The corrected twelve-rank validation
already passed the historical, looser values; the recalibration was performed because 02 §C5
requires regeneration after an implementation correction and forbids preserving an old
tolerance merely because it passed.

```text
N = 200 · master seeds 810000..810199 · interior ranks (31,28,24,20,16,12,8,6,4,3,2,1)
quantile = 97.5th percentile · method = numpy.percentile(method='linear')
DRY_D1_REFERENCE_THRESHOLD = -0.2946425741305496    empirical power 0.975 (195/200)
DRY_D1_MINK_THRESHOLD      = -0.29732128706527483   empirical power 0.975 (195/200)
```

Both arms are calibrated independently, because the generator's reference and Min-K%
perturbations carry different noise and one shared cutoff would mis-state at least one; both
meet the 02 §C5 target of ≥ 0.95. Historical thresholds status:
`SUPERSEDED_PRE_DATA_BY_02_C5_RECALIBRATION` — recorded in source and in the artifact as
history, consumed by nothing.

### Calibration artifact

`artifacts/p0_pre/P0_PRE_00_DRYRUN_TOLERANCE_CALIBRATION.json` — the canonical 00 §34B.2
name — carries all 200 reference statistics, all 200 Min-K% statistics, every seed, the rank
schedule and anchors, the fixture dimensions and structural rank, the planted-effect
parameters and noise model, the quantile convention, both thresholds, both empirical powers,
and the generating code and config hashes: the full bank is the provenance, not only the two
cutoffs. Validation consumes the frozen values from `configs/p0/dry_run.json`; a test asserts
exact equality with the artifact, and calibration is not regenerated on an ordinary run.

`CALIBRATION_REPRODUCIBLE = TRUE · CALIBRATION_HASH_PROVENANCE = PASS ·
CALIBRATION_INTERNAL_CONSISTENCY = PASS`.

## 15. DRY-D validation
```text
DRY_D1_REFERENCE = -0.2998343651250068   <  -0.2946425741305496
DRY_D1_MINK      = -0.2999171825625034   <  -0.29732128706527483
DRY_D1 = PASS                            label TAIL_EFFECT_CORROBORATED

DRY_D2_REFERENCE = -0.30345240442021365  <  -0.08
DRY_D2_MINK      = -0.0017262022101068354   |·| < 0.04
DRY_D2 = PASS                            label BASE_GEOMETRY_SENSITIVE
```

DRY-D2's thresholds were not recalibrated: 02 §C5 covers positive-control power calibration,
and this pair is a structural spec constant rather than a percentile of a planted bank.

## 16. SVD robustness

Calibration seed 810074 reached a finite 40×36 residual iterate on which NumPy's primary
LAPACK divide-and-conquer path (`gesdd`) failed to converge. The matrix was finite and
ordinary-valued — max |·| ≈ 5.42, Frobenius ≈ 52.2, condition ≈ 1.2e8 — and the QR-iteration
driver decomposed it without difficulty: backend convergence failure, not malformed input.

```text
primary          numpy.linalg.svd(full_matrices=False, dtype=float64)   [gesdd]
fallback         scipy.linalg.svd(lapack_driver="gesvd")  ONLY on convergence LinAlgError
non-finite input refused before either path
```

No rank change, clipping, jitter, seed replacement or trial skipping. The fallback never runs
unless the primary raises, so every iterate that already converged keeps its exact bytes.
`SEED_810074 = PASS · FALLBACK_SCOPE = FAILURE_ONLY · SVD_BACKEND_PROVENANCE = PASS`.

## 17. Final independent verification
```text
S08_FINAL_MECHANICAL_VERIFICATION = PASS   PUBLIC_RECOVERY_GUARD            = PASS
RESULT_ISSUANCE_BOUND             = TRUE   SAME_OBSERVATION_R               = ENFORCED
CALIBRATION_ARTIFACT_COMPLETE     = TRUE   CALIBRATION_REPRODUCIBLE         = TRUE
CALIBRATION_STATISTIC_MATCH       = PASS   POWER_REQUIREMENT                = PASS
SEED_810074                       = PASS   RECOVERY_MATH_REGRESSION         = PASS
CALIBRATION_HASH_PROVENANCE       = PASS   CALIBRATION_INTERNAL_CONSISTENCY = PASS
NEW_BLOCKER = NONE                         NEW_MAJOR = NONE
```

## 18. Test gate and protected surfaces
```text
UNIT = 1249 passed · INTEGRATION = 474 passed · SYNTHETIC = 182 passed
FORMAT = PASS · LINT = PASS · TYPECHECK = PASS · INVARIANTS = 0 violations
BACKEND_CONTRACT = NOT_RUN(BACKEND_NOT_INTEGRATED) · GPU_SMOKE = not rerun

S00_UNCHANGED = TRUE · BLOCK_A_PRODUCTION_UNCHANGED = TRUE
BLOCK_C_PRODUCTION_UNCHANGED = TRUE · S07_PRODUCTION_UNCHANGED = TRUE (9/9 byte-identical)
REAL_DATA = FALSE · REAL_MODEL = FALSE · TRAINING = FALSE · H100 = FALSE
PRIVACY_OUTCOMES = FALSE
```

GPU smoke was not rerun and the existing GPU-smoke evidence artifact remained byte-identical.
`SOFTWARE_SLIP = TRUE · CALENDAR_COMPRESSED_MVRS = ACTIVE`: recorded, not acted on. No test
was weakened, removed or relaxed, and no threshold was moved to save time.

## 19. Acceptance boundary

S08 acceptance does **not** certify real-model recovery, real-model merge correctness,
production HF/PEFT backend integration, real DP execution, a selected DARE p\*, a selected O3
s\*, the P1 primary operator, any privacy outcome, the S09 experiment registry, or production
run manifests. It certifies only the pre-data recovery implementation, its provenance and
information-firewall contracts, evaluation truth binding, conditioning, and the corrected
synthetic DRY-D calibration.

## 20. Next

`NEXT_STAGE = PRODUCTION_BACKEND_INTEGRATION`. Required before `P0_PRE_READY`: real HF/PEFT
backend integration · production adapter extraction · production DP/Opacus integration and
cross-check where applicable · real model revisions and hashes · full integrated synthetic
rerun · cache assertions · H100 GPU smoke · the 1,000-sequence production benchmark.

## 21. Verdict
```text
S08_STATUS=ACCEPTED
UNRESOLVED_BLOCKER=NONE
UNRESOLVED_MAJOR=NONE
C1_EXACT_INVERSION=CLOSED
C2_SHARED_SOURCE=CLOSED
C2B_VARYING_ALPHA=CLOSED
O3_RECOVERY_IMPLEMENTATION=CLOSED
O3_RECOVERY_AUTHORIZATION=CONDITIONAL_NOT_RUN
DARE_INCOMPLETE_RECOVERY=METHOD_GATED
DRY_D_STATUS=COVERED
DRY_D1_CALIBRATION=FROZEN_PRE_DATA
REAL_DATA=FALSE
REAL_MODEL=FALSE
TRAINING=FALSE
H100=FALSE
PRIVACY_OUTCOMES=FALSE
NEXT_STAGE=PRODUCTION_BACKEND_INTEGRATION
```
