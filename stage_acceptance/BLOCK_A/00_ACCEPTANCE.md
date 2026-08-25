# Block A Acceptance

## 1. Identity

```text
BLOCK                  = A
TECHNICAL COVERAGE     = S01 provenance/config + S02 deterministic offline fixtures
BASE_COMMIT            = db9ed078702804b6aa581d74910c6f0e902b5d29   (S00 closure)
IMPLEMENTATION_COMMIT  = dcee3c7df4ee5508c0735100c20540bf2e742ce1
                         "feat(block-a): freeze provenance and deterministic fixtures"
BRANCH                 = stage/predata-accelerated
DATE                   = 2026-08-25
```

## 2. Bounded objective

Block A established the provenance floor every later block writes results through:

- deterministic configuration resolution with a stable resolved-config identity;
- one canonical hashing path (canonical JSON, files, artifacts), experiment and run identity;
- model, data and run provenance manifests;
- immutable run attempts;
- artifact integrity recording and verification;
- deterministic offline rank-32 LoRA / linear-merge / oracle-recovery fixtures.

No real scientific measurement was performed.

## 3. Authority

```text
specs/00_MEASUREMENT_SPEC_v1.9_FINAL_CLOSED.md      scientific authority
specs/01_EXECUTION_STACK_LOCK_v2.md                 engineering authority
specs/02_PRE_EXECUTION_IMPLEMENTATION_CORRECTIONS.md binding clarifications
specs/03_REVIEW_GOVERNANCE_LOCK.md                  binding review process
specs/04_EXECUTION_ACCELERATION_AMENDMENT.md        engineering/review cadence
```

`04` changes execution and review **cadence only**. It changes no scientific definition,
threshold, estimator, split, attack, operator or gate [AUTH: 04 §2, §4].

## 4. Implementation surface

```text
src/provenance/hashing.py         canonical JSON, SHA256, artifact record/verify
src/provenance/config.py          deterministic resolution, extends layering, schema
src/provenance/identity.py        provenance inputs, EXPERIMENT_ID, RUN_ID, attempts
src/provenance/repository.py      derived git/config/environment facts, S00 selection rule
src/provenance/model_manifest.py  01 §13 + §8G contract
src/provenance/data_manifest.py   01 §14 contract, disjoint partitions
src/provenance/run_manifest.py    01 §16 fields, §36 terminal states, §30 seed families
src/provenance/runs.py            immutable run attempts
src/models/fixtures.py            S02 tiny fixture and its known-truth algebra
configs/models/tiny_fixture{,_defaults}.json        all material fixture constants
tests/unit/test_s01_{hashing,config,identity,repository,manifests,run_manifest}.py
tests/unit/test_s02_fixtures.py
tests/integration/test_block_a_run_lifecycle.py
tests/fixtures/s02_fixture_golden.json              frozen numeric expectations
```

## 5. Final provenance invariants

```text
EXPERIMENT_ID deterministically binds the eight frozen 01 §15 provenance inputs
RUN_ID distinguishes immutable attempts of one experiment
a surviving manifest OR run directory permanently reserves an attempt index
a failed attempt is never overwritten; a rerun receives a new RUN_ID
evidentiary git commit and dirty state are derived from the repository, not asserted
resolved-config identity is recomputed from the resolved config actually stored
the claimed environment must equal the environment selected under frozen S00 semantics
hardware runtime/GPU attribution must match the accepted environment manifest
the master seed alone does not satisfy 01 §30 RNG-family provenance
model/data manifests are immutable except for byte-identical idempotent rewrites
the research-model contract is derived from the manifest's role, not a caller default
every metrics path entering SUCCESS is hash-bound
SUCCESS re-verifies every registered artifact before the terminal record is written
manifest verification rechecks identity relationships, not only field presence
```

## 6. Deterministic fixture invariants

```text
offline; no network and no model download
tiny Llama-shaped causal-LM fixture, tokenizer, corpus and canary-like records
rank 32 is structurally genuine: every targeted induced update has SVD rank exactly 32
the induced update uses the frozen configured LoRA scaling convention (lora_alpha/rank)
known linear merge with analytic ground truth
known oracle inversion recovering the hidden constituent
deterministic generation pinned by frozen golden numeric expectations
fixture corruption is detectable
```

The fixture makes identities exact on CPU. It does **not** validate real-model behaviour,
real numerics or any research checkpoint.

## 7. Review history

```text
ROUND 1        independent reviews identified real provenance, immutability, metric and
               run-lifecycle defects
FIX ROUND      validated BLOCKER/MAJOR findings repaired
ROUND 2        changed-surface review narrowed the remaining issues
FINAL PATCH    environment selection; identity-field type failure;
               GPU/environment binding; seed-family guard
```

```text
NO THIRD GENERAL REVIEW WAS PERFORMED
```

`04 §8` permits a second review only on a changed scientific mechanism or an unresolved
BLOCKER/MAJOR, and caps the loop at two rounds [AUTH: 04 §8; 03 §12]. The final patch was
closed by planted counterexamples plus the complete gate.

## 8. Final test evidence

```text
Focused final guard tests   45 passed / 0 failed
Unit                        462 passed / 0 failed
Integration                 156 passed / 0 failed
Format                      PASS - 62 files already formatted
Lint                        PASS
Typecheck                   PASS - 51 source files, mypy strict
Repository invariants       PASS - 0 violations
Synthetic                   NOT_RUN(EMPTY_AT_S00)
```

The synthetic/golden science lane is still intentionally empty: it carries the planted-truth
DRY-A…H and DRY-CACHE scenarios, which belong to Block B [AUTH: 00 §34B; 02 §C6]. An empty
lane reports `NOT_RUN(<reason>)`, never `PASS` [AUTH: 01 §22]. This is not a Block A failure.

## 9. S00 preservation

```text
S00 evidence before  1b1b41850ac3417f088dd62c2a73db3e795b7a94239ddf5b8b3fbe08cd8d15e0
S00 evidence after   1b1b41850ac3417f088dd62c2a73db3e795b7a94239ddf5b8b3fbe08cd8d15e0
44/44 files byte-identical
```

Covering `artifacts/p0_pre/`, `manifests/environments/`, `stage_acceptance/S00/`, `reviews/S00/`.

```text
S00 remains CLOSED.
```

## 10. Scientific exposure

```text
REAL_SCIENTIFIC_DATA_INSPECTED  = FALSE
REAL_PRIVACY_OUTCOMES_INSPECTED = FALSE
H100_USED                       = FALSE
```

Block A is therefore entirely pre-data; the `01 §29` / `04 §12` peeking boundary is intact.

## 11. Scope boundaries

Block A does **not** validate:

```text
scorer correctness            reference-loss attack        Min-K%
fixed-FPR estimation          cross-fitting                statistical inference
real dataset                  real model compatibility     LoRA research training
DP-LoRA                       merge operator implementation
recovery implementation       privacy leakage              RQ1            RQ2
```

These belong to later blocks.

## 12. Deferred non-blocking items

```text
build_bundle code_drift whitespace parsing
JSON duplicate-key diagnostic hardening
generic oracle target-set widening
derived input-artifact graph
generic artifact ownership hardening
fixture cosmetic dimensions
env_probe test hermeticisation
attempt orphan cleanup
attempt listing limit alignment
GPU UUID cosmetic shape validation
```

```text
NONE OF THESE IS A BLOCKER OR MAJOR FOR BLOCK A.
```
Each may be addressed only if it becomes necessary in the block that owns it.

## 13. Final verdict

```text
BLOCK_A_STATUS                 = ACCEPTED
UNRESOLVED_BLOCKER             = NONE
UNRESOLVED_MAJOR               = NONE
REAL_SCIENTIFIC_DATA_INSPECTED = FALSE
H100_USED                      = FALSE

NEXT_BLOCK       = BLOCK_B
NEXT_BLOCK_SCOPE = S03 scorer/cache/cross-fit + S04 statistical engine
```
