# S07 Acceptance — Merge Operators

## 1. Identity and scope

```text
STAGE_ID = S07 · STAGE_NAME = MERGE_OPERATORS · STATUS = ACCEPTED
S07_IMPLEMENTATION_COMMIT = 3ed3a1302f8d81bb4131a64cd25ff8113f8beb3c
                            "feat(s07): freeze merge operators and provenance boundaries"
BRANCH = stage/s07-merge-operators · DATE = 2026-08-28T20:37:54+01:00
BASE_ACCEPTED_STATE    = stage-block-c-v1
BASE_ACCEPTANCE_COMMIT = 1afc76b2b08ef5fdaf990de4f44c35417f9d931c

implemented      O1 = LINEAR / TASK_ARITHMETIC · O2 = DARE · O3 = SVD_TRUNC_MERGE
not implemented  TIES · SLERP · quantization · S08 recovery · experiment registry ·
                 real model execution · privacy evaluation
authority        00 §3, §10, §11, §22, §24, §24A, §25, §26 ·
                 01 §8B, §10, §12, §17, §27, §29, §39, §42 · 02 · 03 · 04
```

S07 is deterministic weight-space engineering on synthetic fixtures: no PubMed record fetched,
no research model downloaded or trained, no privacy outcome inspected, no GPU used.

## 2. Scientific object boundary and execution context

All merging operates on induced LoRA updates ΔW = BA. Raw LoRA factor matrices are never
accepted as verified scientific merge inputs: the factorisation is not unique, so a merge
defined on factors would not be a merge of the objects the study measures [00 §22]. Fixture
and verified updates are **structurally distinct classes**, not two states of one class, so
nothing promotes itself:

```text
FixtureTaskVector    caller-supplied matrices. Permanently NON_EVIDENTIARY_FIXTURE.
VerifiedTaskVector   non-dataclass · factory-issued · immutable · NOT constructible from a
                     caller tensor mapping. Issued only by verified_task_vector_from_adapter,
                     which validates the accepted S06 adapter manifest, re-hashes the adapter
                     artifact bytes, loads the validated adapter, and derives ΔW = BA itself.
```

Authoritative tensor storage is immutable Python `bytes`; every accessor returns a fresh
read-only view. Item assignment and `setflags(write=True)` both raise, and a detached copy
cannot reach the stored bytes, so a recorded `content_identity()` stays truthful and ordinary
NumPy mutation or writeback cannot alter authoritative content. There is no public receipt
object whose hash could be recomputed to manufacture verified standing.

`MergeExecutionContext` is non-dataclass, opaque, factory-issued from the canonical resolved
merge configuration, immutable, and bound by a configuration hash over the exact snapshot
consumed — so a run cannot record one version while executing another.

```text
arithmetic_dtype    = float32   scientific merge artifacts   [01 §10; 00 §25 P0-C1]
diagnostic_dtype    = float64   derived scalar quantities    [01 §10]
svd_workspace_dtype = float64   numerical workspace only
```

The higher-precision SVD workspace is permitted only under the frozen context, and the final
O3 artifact is deterministically cast to float32 before it is hashed or composed. A scientific
float64 execution context fails closed.

## 3. The three operators

```text
O1   ΔW_Ci = α_i · ΔW_A + (1 - α_i) · ΔW_Bi                     [00 §10.1]
     descendant counts k ∈ {1, 2, 4} · k = 8 deferred           [00 §11]
```

Exact task-arithmetic semantics: α is not silently clipped, because 00 §10.1 is task arithmetic
and the registered coefficient schedule belongs to S09's registry, not the operator. Parameter
surfaces must match exactly — no zero filling, no broadcasting, no silent tensor dropping, no
NaN or Inf on input or output.

```text
O2   constituent-wise. Per coordinate at drop probability p:
     dropped -> 0 ;  survivor -> x / (1 - p) ;  then the transformed vectors are merged
     registered candidate drop ratios  p ∈ {0.25, 0.50, 0.75, 0.90}   [00 §24]
     p_min = 0.25 · selected p* = NONE · DARE_GATE_STATUS = NOT_RUN
```

RNG policy, architect-adjudicated: **ONE_MERGE_SEED_WITH_ROLE_SUBSTREAMS**. One merge seed
exists per descendant; protected and partner masks are independently derived from
domain-separated substreams keyed on `merge seed · role · task-vector content identity ·
tensor name · p · operator version · mask scheme`. There are no independently caller-selectable
`protected_seed` / `partner_seed` scientific parameters. Masks are drawn internally and their
hashes derive from the actual mask bytes, so no caller supplies a mask or a mask hash.

```text
O3   per constituent, T_s(W) = rank-s truncated-SVD reconstruction, then
     ΔW_C = α · T_s(ΔW_A) + (1 - α) · T_s(ΔW_B)                 [00 §10.3]
     registered candidate ranks s ∈ {24, 16, 8, 4} · selected s* = NONE
     O3_STATUS = PRE_REGISTERED_CONDITIONAL · O3_GATE_STATUS = NOT_RUN
     numerical effective rank σ_j/σ_1 >= 1e-6 · stable rank ||W||_F²/||W||_2²  [00 §24A.6]
```

Truncate-each-constituent-then-merge, never truncate-after-merge: that makes the discarded
singular directions a property of each constituent rather than of the sum. Identity is taken
over reconstructed matrices, never over U/V, which carry a sign and — at repeated singular
values — a basis ambiguity no backend fixes.

## 4. Bound diagnostics

```text
e_floor(s) = sqrt( Σ_m Σ_{j>s} σ_mj²  /  Σ_m Σ_j σ_mj² )                    [00 §24A.1]
e_DARE,p   = ||ΔW_C_DARE - ΔW_C_LINEAR||_F / ||ΔW_C_LINEAR||_F              [00 §24.1]
theoretical reference  sqrt(p / (1 - p)) ;  p = 0.25 -> sqrt(1/3) ≈ 0.5774
```

`e_floor` is computed from the **protected** constituent only. The authoritative path takes an
issued O3 result plus the actual protected update, reads the retained rank off the result,
recomputes `T_s(ΔW_A)` under the same bound context, and verifies the recomputed truncation
identity against the one the result stored. Caller-supplied `parts`, `truncations` or partner
truncations cannot define the floor — there is no such parameter. A wrong protected constituent
is rejected, and a partner's tail cannot be substituted.

`e_DARE` verifies correspondence before any arithmetic runs: same protected identity, same
partner identity, same alpha, same surface, compatible execution context, and the DARE / LINEAR
operator identities the definition names. Wrong A, B, alpha or context baselines all reject; a
zero denominator is an explicit undefined state, never a silent Inf. The theoretical value is
reported beside the realised one, never gated on.

## 5. MergeResult and ReleaseFamily

`MergeResult` is opaque, non-dataclass, immutable and operator-factory-issued. Scientific
identities derive from the actual inputs and the operation that ran, never from caller-provided
metadata; there is one factory per operator and no generic factory whose optional arguments
choose one. Result identity binds at minimum:

```text
operator · operator version · execution-context hash · protected content identity ·
partner content identity · surface identity · alpha ·
operator-specific parameters and evidence · final FP32 result bytes

provenance class = NON_EVIDENTIARY_S07_MERGE    permanent, derived, non-settable
```

S09 may later bind real execution into run-level evidentiary provenance; an S07 `MergeResult`
cannot self-promote. `ReleaseFamily` derives the protected content identity from the actual
issued results and requires exact equality across all of them, so the same actual protected A
occurs in every descendant [00 §3.4]; a family mixing A1 and A2 is rejected. Partner display
labels exist separately — scientific partner identity derives from partner content bytes.

## 6. Monte-Carlo test note

The synthetic DARE expectation test derives its tolerance from the estimator's own sampling
variance rather than a fixed relative bound:

```text
Y = x/(1-p) with probability 1-p ;  Y = 0 with probability p
E[Y] = x ·  Var(Y) = x² · p/(1-p) ·  for n independent draws Var(mean) = Var(Y)/n
assertion: four standard errors
```

Recorded plainly: this is a synthetic distributional implementation test; it is **not** a
registered scientific gate and changes no registered threshold; it does **not** replace the
exact survivor-rescaling tests, which independently assert `1/(1-p)` scaling and exact zeroing
bitwise at every registered p. The new tolerance is **not** uniformly stricter than the old
one — narrower at p = 0.25, wider at p = 0.50 and 0.75, because the previous bound was scaled
to a quantity unrelated to the estimator's dispersion.

## 7. Final mechanical closure, test gate and protection

```text
S07_FINAL_MECHANICAL_VERIFICATION = PASS

PASS      verified vector factory · MergeResult factory · same-A content · partner content ·
          operator identity · context factory · result-context binding · FP32 artifacts ·
          DARE one-seed policy · DARE diagnostic · O3 protected floor
REJECTED  raw factor to verified · public receipt forgery · NumPy writeback ·
          replace attacks · float64 context · DARE seed replay · caller-supplied O3 parts
TRUE      authoritative storage immutable
VALID     O3 tolerance change · DARE Monte-Carlo change
NEW_BLOCKER = NONE · NEW_MAJOR = NONE

UNIT = 1064 passed · INTEGRATION = 409 passed · SYNTHETIC = 141 passed
FORMAT = PASS · LINT = PASS · TYPECHECK = PASS · INVARIANTS = 0 violations

S00_UNCHANGED = TRUE (1b1b41850ac3417f…) · BLOCK_A_PRODUCTION_UNCHANGED = TRUE (c07b618b…)
BLOCK_B_PRODUCTION_UNCHANGED = TRUE (804fa85d…) · BLOCK_C_PRODUCTION_UNCHANGED = TRUE (76b65d2a…)
REAL_DATA = FALSE · REAL_MODEL = FALSE · RESEARCH_TRAINING = FALSE
H100 = FALSE · PRIVACY_OUTCOMES = FALSE
```

No P0 scientific outcome has been opened.

## 8. Block-B compatibility update

```text
BLOCK_B_PRODUCTION_MODIFIED = FALSE · BLOCK_B_COMPATIBILITY_TEST_UPDATED = TRUE
file = tests/synthetic/test_dry_coverage.py
```

The previous test used the absence of **both** the S07 and S08 surfaces as its proxy for the
DRY-D dependency; S07 now legitimately exists while S08 remains absent, so the proxy went
stale. The replacement preserves the true dependency — Δ_tail_priv needs a recovery R_priv
[00 §28B.4] — by checking that `src/recovery` is empty, that no merge module defines a recovery
quantity, and that `P1_PRIMARY_LOSSY_OPERATOR` fails closed; all three re-fire when S08 lands.
`DRY_D_STATUS = NOT_RUN_DEPENDENCY(S07_S08)` remains correctly deferred, and
`src/analysis/dryrun.py` was NOT modified.

## 9. Acceptance boundary and next stage

Accepting S07 does **not** certify:

```text
a selected DARE p* · DARE utility realism · a selected O3 rank s* · the O3 utility gate ·
primary lossy operator selection · real model merging · real privacy outcomes ·
S08 recovery correctness · S09 experiment registry · production run manifests
```

It certifies the **pre-result merge-operator implementation and its provenance contracts** only.

```text
NEXT_STAGE = S08 · NEXT_STAGE_NAME = RECOVERY
expected bounded scope   C1 exact inversion · C2 spectral / common-source reproduction ·
                         C2B known-varying-alpha rescaling · conditional O3 recovery ·
                         conditioning metrics
```

## 10. Verdict

```text
S07_STATUS=ACCEPTED
UNRESOLVED_BLOCKER=NONE
UNRESOLVED_MAJOR=NONE
O1_LINEAR=CLOSED
O2_DARE=CLOSED
O3_SVD_TRUNC=CLOSED
DARE_GATE_STATUS=NOT_RUN
O3_GATE_STATUS=NOT_RUN
TIES=NOT_IMPLEMENTED
S08_IMPLEMENTED=FALSE
REAL_DATA=FALSE
REAL_MODEL=FALSE
H100=FALSE
PRIVACY_OUTCOMES=FALSE
NEXT_STAGE=S08
NEXT_STAGE_NAME=RECOVERY
```
