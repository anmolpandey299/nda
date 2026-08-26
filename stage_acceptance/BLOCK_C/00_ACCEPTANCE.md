# Block C Acceptance

## 1. Identity

```text
BLOCK   = BLOCK_C
STAGES  = S05 (data/canary) + S06 (training/DP)
STATUS  = ACCEPTED
BLOCK_C_IMPLEMENTATION_COMMIT = 37f5aaf82e1607791a24f0b0652b17689371456a
                                "feat(block-c): freeze data and training contracts"
BRANCH  = stage/predata-accelerated
HEAD    = 37f5aaf82e1607791a24f0b0652b17689371456a
DATE    = 2026-08-26T18:03:40+01:00
BLOCK_B_IMPLEMENTATION_COMMIT = 22e6747cef87556a1d7b6c3e2ac36d7994c44586
BLOCK_B_ACCEPTANCE_COMMIT     = 8b186cff6920fff8308f78157d4468b392c6fad9
```

## 2. Bounded objective and scope

Block C implements and freezes the pre-real-data data and training contracts. No PubMed record
was fetched, no research model downloaded or trained, no privacy outcome inspected, no GPU
used; every test runs on deterministic generated fixtures.

```text
S05  research acquisition contract · publication-date eligibility · deterministic
     normalization · exact dedup · near dedup · immutable master splits · natural-member
     subsets · canary pool · Bernoulli canary inclusion · matched no-canary plan ·
     blind split/dedup control · corpus and data manifests
S06  research model contract · LoRA target policy · trainable-parameter audit · material
     training configuration · explicit RNG seed families · reference LoRA trainer contract ·
     immutable TrainingExecutionContract · matched treated/control schedule · DP
     mechanism/accounting reference · DP smoke and inferential guards · adapter manifests
     and artifact verification
Authority  specs/00 scientific · 01 engineering · 02 clarifications · 03 review process ·
           04 engineering/review cadence only
```

## 3. Frozen natural-data policy and the natural-member adjudication

```text
PUBLICATION_DATE_RULE = publication_date >= 2026-04-01                    [01 §8F]
                        indexing date is NEVER a substitute; it is parsed so a
                        malformed one is still refused, then discarded
MASTER_SPLIT_SEED     = 20260820                                         [00 §6.2]
MASTER_SPLITS         = TRAIN_CANDIDATES 30000 · CALIBRATION_NONMEMBERS 5000
                        EVAL_NONMEMBERS 5000 · RESERVE 5000
```

Deduplication runs **before** split assignment; partitions are disjoint by record id and by
normalised text hash and do not move across seeds. The 00 §6.3 / §27.1 conflict was
adjudicated by the architect before any real id existed:

```text
NATURAL_MEMBER_CALIBRATION = 2000 · NATURAL_MEMBER_EVAL = 5000
NATURAL_MEMBER_EVAL_ACROSS_SEEDS        = one fixed set, reused across 101, 202, 303
NATURAL_MEMBER_CALIBRATION_ACROSS_SEEDS = per-training-seed deterministic draw
CANARY_INCLUSION_ACROSS_SEEDS           = independent Bernoulli(0.5) per canary per seed
CANARY_POOL                             = 4096 candidates
```

The evaluation draw is keyed on the fixed master study/split seed and the domain separator
`natural_member_eval`, calibration on the training seed and `natural_member_calibration`; no
post-hoc numeric seed was introduced, and both subset identities are hashed and persisted.
Each included canary exists exactly once in the corpus; epoch iteration revisits that corpus
and does not duplicate it.

## 4. Blind split / deduplication control

```text
TF-IDF word 1-2 grams · L2 logistic regression · balanced labels enforced at the controller
boundary · 5-fold stratified CV · primary seed 6104 · confirmatory seed 6105 · AUROC ·
95% CI · independent member/non-member record bootstrap, invariant to class row ordering
```

Investigation triggers if **either** the point estimate leaves `[0.47, 0.53]` **or** the 95% CI
excludes 0.5, so a tight interval away from 0.5 inside the band is never CLEAN.
STOP_NATURAL_PRIVACY_ARM only if a concrete duplicate/preprocessing leakage path is identified,
**or** both the 6104 and 6105 AUROCs fall outside `[0.47, 0.53]` in the same direction; anything
else retains the arm and reports the investigation. 00 §6.4 fixes no CI estimator, so its
algorithm, replicates, seed and alpha are frozen in `configs/data/corpus.json` first.

```text
C3_INDEPENDENT_BLIND_BOOTSTRAP = CLOSED
```

## 5. Training and the matched control

```text
LORA_RANK = 32 · PRIMARY_LORA_POLICY = LANGUAGE_TRUNK_MLP_ONLY (fallback declared, unused)
```

Execution is controlled by one immutable `TrainingExecutionContract` binding the TrainingPlan,
ModelContract, actual target modules, trainable-parameter audit, rank/scaling/dropout, resolved
training config, seed families, precision, sequence length, update budget, privacy regime and
backend status. The audit is derived from the surface execution will use, so an audit that
disagrees with execution is unrepresentable.

Matched no-canary rule: the **complete canary-containing treated schedule** defines the
optimizer-update budget; the control matches that count by deterministically cycling its own
natural data order. The treated run is never truncated to the smaller control corpus, and the
control never gains a canary, loses a natural record or receives filler. Final mechanical
counterexample, executed on fixtures:

```text
20 natural · 13 included canaries · batch size 4 · epochs 1
treated corpus size 33 · treated updates 9 · control updates 9
included canaries 13 · consumed canaries 13 · unconsumed canaries 0 · control canaries 0

C4_COMPLETE_TREATED_SCHEDULE = CLOSED
```

## 6. Review closure

```text
BC1_RESEARCH_CORPUS_PROVENANCE   = CLOSED    BC2_MEMBERSHIP_INTEGRITY       = CLOSED
BC3_BLIND_CONTROL                = CLOSED    BC4_TRAINING_EXECUTION_BINDING = CLOSED
BC5_EVIDENTIARY_ADAPTER_MANIFEST = CLOSED    BC6_DP_INFERENTIAL_SET         = CLOSED
NATURAL_POLICY                   = CLOSED
BLOCK_C_FINAL_MECHANICAL_VERIFICATION = PASS
BC1_REGRESSION = PASS · BC2_REGRESSION = PASS · BC5_REGRESSION = PASS ·
BC6_REGRESSION = PASS · NEW_BLOCKER = NONE · NEW_MAJOR = NONE
```

## 7. Differential privacy

```text
DP_SMOKE_SEED = 101 · DP_INFERENTIAL_SEEDS = 101, 202, 303
```

A single `DPRun` cannot claim inferential status: the field does not exist and the property is
derived as NON_INFERENTIAL. Inferential DP status is established only by a validated
`InferentialDPSet` carrying the required seeds, one consistent scientific condition across them,
and an evidentiary-ready backend and accountant for every member.

```text
REFERENCE_DP_ACCOUNTANT = s06.rdp-sgm-integer-orders.v1
reference validation    = sigma 1 · q 0.01 · steps 1000 · delta 1e-5 -> epsilon ~ 2.107753
ACCOUNTANT_REFERENCE_VALID = TRUE · PRODUCTION_DP_ACCOUNTING_READY = FALSE
```

No real achieved epsilon is claimed for the study.

## 8. Configuration status

```text
REQUIRED_NOT_CALIBRATED_COUNT = 12
corpus.source_query · lora.optimizer · lora.learning_rate · lora.epochs · lora.batch_size
lora.gradient_accumulation_steps · lora.sequence_length · lora.gradient_clipping_norm
lora.precision · dp.delta · dp.clipping_norm · dp.noise_multiplier
```

These fail closed for research/evidentiary execution: reading one raises rather than returning a
default, so no silent default can become a scientific choice. Several
`PROVISIONAL_FIXTURE_ONLY` settings still require freezing before real acquisition or execution,
notably `near_duplicate_shingle_size`, `near_duplicate_threshold` and
`blind_control_regularisation`. No final value is invented here.

## 9. Backend boundary

```text
NON_DP_PRODUCTION_BACKEND = NOT_RUN_DEPENDENCY(TORCH_PEFT_BACKEND)
DP_PRODUCTION_BACKEND     = NOT_RUN_DEPENDENCY(OPACUS_BACKEND)
BACKEND_INTEGRATED        = FALSE
REAL_EXECUTION_FAILS_CLOSED_WHILE_DEFERRED = TRUE
```

These dependencies are **deferred, not waived**. Production validation must occur before any
evidentiary model training. Evidentiary adapter validation checks the backend status directly,
so a deferred backend cannot satisfy it under any combination of other fields.

## 10. Test gate, protection and no-peek

```text
UNIT = 868 passed · INTEGRATION = 395 passed · SYNTHETIC = 112 passed
FORMAT = PASS · LINT = PASS · TYPECHECK = PASS · INVARIANTS = 0 violations
S00_EVIDENCE_PRESERVED = TRUE · BLOCK_A_PRODUCTION_PRESERVED = TRUE
BLOCK_B_PRODUCTION_PRESERVED = TRUE
REAL_PUBMED_FETCHED = FALSE · RESEARCH_MODEL_DOWNLOADED = FALSE
RESEARCH_MODEL_TRAINED = FALSE · REAL_PRIVACY_OUTCOMES_INSPECTED = FALSE
H100_USED = FALSE · P0_PRE_READY = FALSE
```

No P0-PRE readiness artifact was created by Block C.

## 11. Acceptance scope boundary and next stage

Accepting Block C does **not** certify:

```text
the final PubMed corpus · the final source query · real model revisions or downloads
production Torch/PEFT execution · production Opacus integration
the actual training hyperparameters · the actual DP epsilon · real H100 training
scientific privacy findings · S07 merge · S08 recovery
```

It certifies the **pre-real-data S05/S06 contracts and their deterministic fixture
implementation** only.

```text
NEXT_BLOCK       = BLOCK_D
NEXT_BLOCK_SCOPE = S07 merge implementation + S08 recovery implementation
```

## 12. Verdict

```text
BLOCK_C_STATUS=ACCEPTED
UNRESOLVED_BLOCKER=NONE
UNRESOLVED_MAJOR=NONE
REAL_PUBMED_FETCHED=FALSE
RESEARCH_MODEL_DOWNLOADED=FALSE
RESEARCH_MODEL_TRAINED=FALSE
REAL_PRIVACY_OUTCOMES_INSPECTED=FALSE
H100_USED=FALSE
BACKEND_INTEGRATED=FALSE
PRODUCTION_DP_ACCOUNTING_READY=FALSE
P0_PRE_READY=FALSE
NEXT_BLOCK=BLOCK_D
NEXT_BLOCK_SCOPE=S07_merge_plus_S08_recovery
```
