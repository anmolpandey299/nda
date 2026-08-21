# 00_MEASUREMENT_SPEC
## Measurement Constitution for Model-Merge Lineage, Parameter Recovery, and Privacy Recovery

**Version:** `v1.9 FINAL CLOSED`  
**Status:** `FINAL CLOSED PRE-DATA EXECUTION AUTHORITY — NO FURTHER SPEC REVIEW`  
**Date:** 20 August 2026  
**Upstream decision document:** `RESEARCH_DIRECTION_LOCK.md`  
**Next execution stage:** `P0 — Measurement / Validity / Recovery Plumbing Pilot`

---

# 0. Purpose and authority

This document converts the locked research direction into an **operational measurement specification**.

It defines:

- the exact research questions to be measured;
- notation and observation models;
- experimental views;
- primary and secondary metrics;
- privacy-endpoint eligibility;
- recovery baselines;
- attack baselines;
- seed structure;
- canary and natural-record design;
- continuous inferential analysis;
- display-only thresholded frontiers;
- pilot gates;
- DARE utility gate;
- deployment-realism scan;
- conditions for expansion, redesign, or stopping.

This document is authoritative for P0 and P1.

Any implementation that deviates from it must record:

```text
SPEC_DEVIATION
```

with:

1. the exact deviation;
2. why it was necessary;
3. whether it occurred before or after viewing the affected result;
4. whether it changes interpretation.

No silent post-hoc changes are allowed.

## 0.1 Pre-data revision record

All revisions listed here were made **before any P0 data were generated or inspected**. They are legitimate specification corrections, not `SPEC_DEVIATION` entries.

### v1.1 changes from v1.0

1. increased the candidate-canary pool from 512 to 4,096 and separated canary/non-canary threshold calibration;
2. added a mandatory text-only blind baseline for corpus-shift detection;
3. added `P0-F`, a synthetic parameter-fidelity ladder that supplies controlled error magnitude without additional training runs;
4. made paired raw audit differences primary within cells and normalized audit gaps secondary for cross-cell comparison;
5. changed the primary inference from a small clustered P1 regression to comparison of operator-induced points against a synthetic magnitude calibration curve;
6. changed seed from a random effect to a fixed/blocking factor for three-seed P0/P1 analyses;
7. corrected the deployment-reuse estimator to perform parent-centered reverse lookup outside the sampled merge set;
8. added an open `Z-INCOMPLETE-VARYING-KNOWN` coefficient regime to P1;
9. marked `k=1` unknown-partner cells as structurally underdetermined / `N/A`;
10. added distance-to-base diagnostics;
11. clarified that the single P0 DP point is a confirmatory measurement-floor check, not a boundary locator;
12. made the best-single-descendant comparator explicitly conservative for estimating lineage benefit.

### v1.2 changes from v1.1

1. made a **LoRA-rank-matched perturbation ladder** the primary fidelity control;
2. retained the full-rank isotropic ladder as a mandatory secondary control;
3. added direct rank-matched-vs-isotropic comparison at equal Frobenius error;
4. added a controlled functional-recovery-versus-privacy-recovery decay analysis;
5. made pooled-lineage normalization arm-matched for canaries versus natural records;
6. nested descendant and pooled-method selection inside the canary cross-fitting loop;
7. required realised FPR beside every operational TPR and fixed-FPR interpolation for cross-view comparison;
8. added a matched no-canary contamination-control training run;
9. partitioned deployment parents into foundation / derivative / unknown and moved the reuse statistic to derivative parents;
10. explicitly stated that public derivative-parent reuse is only a proxy for unobservable withheld-parent reuse;
11. added denominator-stability reporting for normalized audit gaps;
12. declared TPR@0.1%FPR unpowered for P0;
13. made natural-member sampling uniformly random and explicitly excluded canaries;
14. added perplexity interpretation to the DARE utility gate;
15. renamed protected objects `PO1/PO2/PO3` to avoid collision with execution-phase names;
16. recorded the verified Llama 3.2 December 2023 pretraining-data cutoff assumption for the 2025 PubMed corpus.

### v1.3 changes from v1.2

All changes below were made before any P0 result was generated or inspected.

1. extended the coarse fidelity ladders through \(e=2.0\) and prohibited extrapolation beyond calibrated support;
2. added a separate zero-delta/base failure anchor at exactly \(e=1\);
3. added a pre-registered second-stage ladder-refinement rule for accurate 50% crossing estimates;
4. added a functional-normalization eligibility floor mirroring the privacy-denominator protection;
5. added distance-to-base as a structural covariate in operator-residual analysis;
6. made the complete operator-residual analysis run in parallel for the reference-calibrated score and Min-K%;
7. added an explicit P0 compute-budget/throughput gate and reference-score caching plan;
8. reframed the blind text-only control as a split/deduplication leakage check and removed a one-shot false-stop path;
9. made representative-DP seed allocation explicit;
10. fully enumerated the 28 nominal P1 registry cells and marked method-gated / structurally N/A cells before execution;
11. documented the different across-seed pairing structures of canaries and natural records;
12. added normalized-gap denominator-stability interpretation rules;
13. declared TPR@0.1%FPR unpowered for P0;
14. confirmed natural-member sampling is uniform random from trained natural records and excludes canaries.

### v1.4 changes from v1.3

All changes below were made before P0 result inspection.

1. corrected the lineage taxonomy: **known varying coefficients are not an open inverse problem**; they reduce to the shared-source low-rank form after per-descendant rescaling;
2. changed C08/C09 and D06/D07 from open-recovery cells to a rescaled-Spectral reproduction/control regime;
3. explicitly deferred **hidden coefficients with unknown partners** from P1 because the unrestricted problem has scale/non-identifiability unless additional side information is supplied;
4. changed Stage-2 crossing refinement to use a **seed-balanced pooled isotonic fit** for one common refinement bracket per ladder/endpoint;
5. added the rule that `Δe50` is `COMPUTE_LIMITED_NOT_ESTIMATED` if both rank-matched crossing refinements are not affordable;
6. shared `HELDOUT_FUNCTION_RECORDS` with `EVAL_NONMEMBERS`, reducing the per-artifact scoring envelope from 24,608 to 19,608 sequences;
7. updated the P0 scoring envelope from ~10.3M to ~8.2M sequence forwards;
8. added a P1 inference/forward-pass budget and benchmark-derived time projection;
9. required Min-K% to be reduced inline from per-token log-probabilities without persisting full token-level tensors;
10. clarified that the two text-only CV seeds are correlated robustness partitions, not independent replications.

### v1.5 changes from v1.4

All changes below were made before P0 result inspection.

1. pre-registered `O3 = SVD-TRUNC-MERGE` as a fallback information-removal merge transformation if DARE fails its utility gate;
2. gave O3 its own utility and non-triviality gate and prohibited presenting it as a novel or prevalence-backed deployment operator;
3. added a DARE realised perturbation metric and the theoretical \(\sqrt{p/(1-p)}\) reference;
4. split the known-varying-\(lpha\), \(k=4\) stress test into a beta-spread-matched schedule and a wide-spread schedule;
5. added explicit heteroscedasticity metrics: analytic beta spread, realised residual-norm spread, and rescaling amplification;
6. added a matched-partner-norm guard for the \(k\)-versus-heteroscedasticity comparison;
7. updated the P1 registry from 28 to 29 rows and updated the P1 scoring budget accordingly;
8. clarified that `EVAL_NONMEMBERS` does not calibrate the FPR threshold and that shared-record covariance with the function metric is preserved by paired bootstrap;
9. added the contingency that if both DARE and O3 fail their gates, operator-specific RQ1 claims are narrowed rather than rescued post hoc.

### v1.6 changes from v1.5

All changes below were made before P0 result inspection.

1. added the exact O3 constituent-information floor \(e_{\mathrm{floor}}(s)\) from discarded singular-value energy;
2. separated O3's operator-imposed floor, solver error to the released truncated constituent, and absolute error to the original constituent;
3. corrected threshold interpretation: original-constituent L3 is analytically unreachable only when \(e_{\mathrm{floor}}(s^*)>	au\), not merely because the merged-delta O3 distortion passes its non-triviality gate;
4. added a conditional deterministic spectral-truncation ladder if O3 becomes the primary lossy operator;
5. made the truncation ladder the primary O3 calibration and retained the rank-matched additive ladder as a controlled spectral-tail comparison;
6. added the named `SPECTRAL_TAIL_PRIVACY` mechanism analysis;
7. added numerical effective-rank and stable-rank diagnostics for O3 descendants and corresponding linear merges;
8. corrected operator × lineage scope: if O3 is selected, incomplete-lineage recovery remains a shared-source low-rank problem and becomes runnable after a sanity reproduction; if DARE is selected, those cells remain method-gated unless a DARE-compatible method is separately validated;
9. documented why O3 has a separate non-triviality gate while DARE's minimum registered \(p=0.25\) already implies a large reference RMS perturbation;
10. added the conditional O3-ladder compute envelope.

### v1.7 changes from v1.6

All changes below were made before P0 result inspection.

1. made the Min-K% parallel readout **explicitly mandatory for `SPECTRAL_TAIL_PRIVACY`**, where the fixed-base reference score is maximally exposed to distance-to-base geometry;
2. added a pre-P0 **single parameterised scoring-engine requirement** so canary/natural, reference/Min-K%, oracle/descendant/pooled/recovered, and artifact variants do not fork into separate analysis pipelines;
3. added a pre-P0 **synthetic analysis dry-run requirement** with planted truths for §28A, §28B, and §29;
4. upgraded the §42 statistical-freeze requirement: scripts are frozen only after they pass the synthetic dry-run assertions;
5. added an execution-risk rule prohibiting duplicated scoring logic across analysis arms unless documented as a `SCORER_EXCEPTION`;
6. added the exact required pre-P0 implementation artifacts and dry-run outputs.

### v1.8 FINAL changes from v1.7

All changes below were made before P0 result inspection.

1. added `DRY-G`, a full null scenario at approximately the real P1 clustered cell count, and required the analysis stack to **not manufacture** monotone, operator, lineage-gap, or reconstruction-gap effects;
2. added `DRY-H`, an end-to-end outer cross-fitting controller test covering fold construction, nested candidate selection, calibration-only thresholding, realised FPR, fixed-FPR ROC interpolation, and a deliberate leakage negative control;
3. converted all dry-run expectations into numeric mechanical assertions with frozen tolerances;
4. added cache correctness tests: identical inputs must produce a stable key/hit with bitwise-identical cached scores, while changing the scorer-code version must invalidate the cache;
5. added a dated internal execution calendar anchored to 20 August 2026;
6. set the hard internal **P1 launch drop-dead at 3 September 2026, 18:00 BST**; this is a project-management deadline, not a claimed University deadline;
7. pre-registered `MINIMUM_VIABLE_RESULT_SET` as the protected dissertation fallback if time/compute prevents P1;
8. added automatic calendar-compression rules that stop optional operator, DP, incomplete-lineage, isotropic-ladder, and deployment expansions before they threaten the minimum viable result;
9. changed the statistical freeze gate so the cross-fitting and cache negative controls must pass before any real evaluation result is opened.

### v1.9 FINAL CLOSED changes from v1.8 FINAL

All changes below were made before P0 result inspection.

1. replaced hand-set DRY-G null tolerances with empirical tolerances calibrated from 200 fixed synthetic null generators;
2. froze the exact 200 calibration seeds and the resulting 97.5th-percentile null thresholds in a machine-readable calibration artifact;
3. calibrated DRY-B and DRY-D1 positive-control thresholds against 200 fixed planted-effect simulations so the dry run is demonstrably powered;
4. changed DRY-G to three fixed validation trials, with majority-pass semantics; one failed trial triggers investigation rather than an immediate software-stop decision;
5. changed DRY-G uncertainty resampling to **registry-cell cluster bootstrap**, preserving all seed observations belonging to the sampled cell;
6. added `SOFTWARE_SLIP`: P0-PRE is non-compressible, no assertion may be waived/weakened to preserve dates, and incomplete P0-PRE by 27 August 2026 automatically activates `CALENDAR_COMPRESSED_MVRS`;
7. re-baselined P0-PRE from a one-day task to 21–27 August 2026;
8. reconciled the calendar with §35 by classifying P0-C2, P0-D/O3, and P0-E as post-MVRS expansion gates rather than minimum-viable critical-path work;
9. retained the hard internal P1 launch drop-dead of 3 September 2026, 18:00 BST; missing it is an acceptable pre-registered outcome, not a reason to bypass software validity;
10. declared this document closed: the next revision may be driven only by a measured implementation impossibility, a mathematical error, or actual empirical evidence.

---

# 0.2 P0 compute budget and throughput gate

The fidelity ladders require no additional training, but they require substantial inference.

P0 must benchmark the real accelerator **before** the first full ladder is scored.

## 0.2.1 Fixed scoring sets per synthetic artifact

A synthetic reconstruction is scored on, at most:

```text
CANARY_CANDIDATES          = 4,096
CALIBRATION_NONMEMBERS     = 5,000
NATURAL_MEMBER_EVAL        = 5,000
EVAL_NONMEMBERS            = 5,000
HELDOUT_FUNCTION_RECORDS   = EVAL_NONMEMBERS
FIXED_LOGIT_PROMPTS        = 512
```

`NATURAL_MEMBER_CALIBRATION` is not required for a single synthetic ladder artifact because no descendant/pooling method is being selected.

`HELDOUT_FUNCTION_RECORDS` intentionally reuses `EVAL_NONMEMBERS`.

This is valid because:

- both require untrained natural records from the same target distribution;
- **FPR thresholds are calibrated on `CALIBRATION_NONMEMBERS`, not on `EVAL_NONMEMBERS`**;
- `EVAL_NONMEMBERS` is used only for held-out realised FPR / ROC evaluation and descriptive held-out function measurement;
- the function metric is not used to tune the attack;
- no model, attack, threshold, pooled method, or recovery method is selected using either held-out metric;
- when comparing \(R_{\mathrm{func}}\) and \(R_{\mathrm{priv}}\), bootstrap resampling preserves the shared-record dependence rather than treating the two statistics as independent;
- all evaluation outputs are computed only after the relevant attack/recovery code is frozen.

Conservative per-artifact planning count:

\[
N_{\mathrm{seq/artifact}}=19{,}608.
\]

Where one forward pass supports multiple metrics, outputs must be cached and reused.

The fixed base/reference quantities:

\[
\ell_{\mathrm{ref}}(x)
\]

and base-model logit outputs are computed **once per unique record/prompt and cached**.

## 0.2.2 Coarse-ladder artifact count

Stage 1 contains:

```text
13 error levels
× 3 perturbation draws
× 3 training seeds
× 2 perturbation structures
= 234 synthetic artifacts
```

plus:

```text
3 zero-delta/base anchors
```

one per training seed.

Conservative Stage-1 target-model sequence-forward envelope:

\[
(234+3)\times19{,}608=4{,}647{,}096.
\]

## 0.2.3 Refinement upper bound

The refinement rule in §28A.8 can add at most:

```text
8 new error levels per ladder type
× 3 draws
× 3 seeds
× 2 ladder types
= 144 additional synthetic artifacts.
```

Maximum refinement envelope:

\[
144\times19{,}608=2{,}823{,}552
\]

additional target-model sequence forwards.

Maximum ladder envelope:

\[
7{,}470{,}648
\]

target-model sequence forwards.

Add a 10% planning reserve for P0-C2, DARE, and repeated diagnostics:

\[
N_{\mathrm{P0,plan}}\approx8.2\text{ million sequence forwards}.
\]

## 0.2.4 Hardware benchmark

Before full ladder execution, benchmark:

```text
1,000 representative sequences
```

using the actual:

- accelerator;
- precision;
- batch size;
- tokenizer;
- sequence-length distribution;
- required loss/logit outputs.

Let measured throughput be:

\[
q=\text{sequences/second}.
\]

Projected P0 inference time is:

\[
H_{\mathrm{P0}}
=
\frac{N_{\mathrm{P0,plan}}}{3600q}.
\]

Planning examples only:

| Measured throughput \(q\) | Approx. 8.2M-forward envelope |
|---:|---:|
| 10 seq/s | 228 GPU-hours |
| 20 seq/s | 114 GPU-hours |
| 40 seq/s | 57 GPU-hours |
| 80 seq/s | 29 GPU-hours |

These are not assumed runtimes. The measured benchmark replaces them.

Record:

```text
P0_00_COMPUTE_BUDGET.json
```

with:

- accelerator type;
- precision;
- batch size;
- sequence-length distribution;
- measured sequences/s;
- measured tokens/s;
- projected Stage-1 hours;
- projected refinement hours;
- available P0 compute budget.

If maximum refinement exceeds available compute, Stage 1 still runs unchanged. Refinement priority is:

1. rank-matched privacy crossing;
2. rank-matched functional crossing;
3. isotropic privacy crossing;
4. isotropic functional crossing.

`Δe50` requires **both priorities 1 and 2**.

If only priority 1 is affordable:

```text
Δe50 = COMPUTE_LIMITED_NOT_ESTIMATED
```

This is distinct from `RIGHT_CENSORED`, which is reserved for an eligible curve that remains above the target level through calibrated support.

Any compute-driven reduction must be frozen before the affected refined results are inspected.

---

## 0.2.5 Conditional O3 spectral-truncation ladder budget

This budget is incurred only if:

```text
P1_PRIMARY_LOSSY_OPERATOR = SVD_TRUNC_MERGE
```

The conditional truncation ladder in §28B adds the retained ranks:

```text
s ∈ {31, 28, 24, 20, 16, 12, 8, 6, 4, 3, 2, 1}
```

for each of the 3 training seeds.

The existing anchors are reused:

```text
s = 32 -> oracle A
s = 0  -> zero-delta/base anchor
```

so they require no new scoring artifacts.

New scoring artifacts:

```text
12 ranks × 3 seeds = 36 artifacts
```

At:

\[
19{,}608
\]

sequences per artifact:

\[
36\times19{,}608
=
705{,}888
\]

additional sequence forwards.

With a 10% reserve:

\[
N_{\mathrm{O3\ ladder,plan}}
\approx
0.78\text{ million sequence forwards}.
\]

Thus the maximum P0 inference plan becomes approximately:

```text
8.2M + 0.78M ≈ 9.0M sequence forwards
```

only if O3 is activated.

| Measured throughput \(q\) | Conditional O3 addition |
|---:|---:|
| 10 seq/s | 22 GPU-hours |
| 20 seq/s | 11 GPU-hours |
| 40 seq/s | 5.4 GPU-hours |
| 80 seq/s | 2.7 GPU-hours |

Add this conditional cost to `P0_00_COMPUTE_BUDGET.json` immediately after O3 selection and before §28B is scored.

---

# 0.3 P1 compute budget

P0 feasibility does not automatically imply that P1 fits the available compute budget.

P1 therefore receives a separate forward-pass projection using the same measured throughput \(q\) from §0.2.

## 0.3.1 Upper-bound artifact accounting

The 29-row registry in §36 contains:

```text
19 calibration rows
10 authorized representative-DP rows
```

with structural N/A rows never executed.

For a conservative upper bound, assume every non-N/A operator/method gate eventually passes.

A cell with \(k\) descendants requires target-model scoring for:

```text
k descendant artifacts
+ 1 recovered artifact
```

The pooled-lineage view reuses descendant scores and requires no extra model forward pass.

Across all non-N/A calibration registry rows:

```text
57 scored model artifacts per seed
```

Across the authorized DP registry:

```text
36 scored model artifacts per seed
```

Total:

```text
93 artifacts per seed
× 3 seeds
= 279 scored cell artifacts
```

Add six cached oracle views:

```text
3 calibration A_s
+ 3 DP A_s
```

for:

```text
285 P1 scored model artifacts
```

as a conservative upper bound.

The O3 fallback is mutually substitutive for O2 in the main P1 operator axis if DARE fails; it does not automatically add a second full operator grid.

## 0.3.2 P1 scoring envelope

At:

\[
19{,}608
\]

sequences per scored artifact:

\[
285\times19{,}608=5{,}588{,}280
\]

sequence forwards.

Add a 10% inference reserve:

\[
N_{\mathrm{P1,plan}}\approx6.15\text{ million sequence forwards}.
\]

This upper bound falls if method/operator-gated cells remain closed.

It excludes:

- model/adaptor training;
- recovery-optimizer iterations;
- one-time merge construction;
- SVD construction cost for O3;
- deployment-scan API work.

Those costs are budgeted separately after P0 provides measured values.

## 0.3.3 Benchmark-derived P1 time

| Measured throughput \(q\) | Approx. 6.15M-forward envelope |
|---:|---:|
| 10 seq/s | 171 GPU-hours |
| 20 seq/s | 85 GPU-hours |
| 40 seq/s | 43 GPU-hours |
| 80 seq/s | 21 GPU-hours |

Record the P1 projection in:

```text
P0_00_COMPUTE_BUDGET.json
```

before authorizing P1.

P1 is not launched unless:

```text
projected_P0_remaining_hours
+ projected_P1_hours
<= available_compute_budget
```

or a pre-result compute-reduction plan has been frozen.

---

# 1. Frozen research questions

## RQ1 — Parameter recovery versus privacy recovery

> **Under ordinary model-merge release families, how does privacy recovery vary with parameter-recovery quality across merge transformations and lineage knowledge?**

The primary inferential object is the **continuous relationship** between parameter recovery error and normalized privacy recovery.

The project does not assume that parameter recovery and privacy recovery coincide.

---

## RQ2 — Value of joint lineage

> **Separately, when does access to the joint public release lineage improve privacy auditing over artifact-by-artifact evaluation?**

This is not assumed to follow from RQ1.

The study will separately measure:

1. the value of pooling the release family; and
2. the value of routing that same information through a reconstruction.

---

# 2. Non-goals

This study does **not** attempt to establish:

- the first unseen-source recovery attack;
- the first multi-descendant recovery attack;
- a new generic low-rank inverse theorem;
- universal identifiability of TIES or DARE;
- a new coefficient-recovery method as a headline;
- the first generic recovery-to-privacy attack;
- the first demonstration that function fidelity can differ from privacy fidelity;
- the first passive lineage attack;
- a violation of differential privacy;
- information creation by reconstruction;
- universal privacy rankings of merge algorithms.

The paper is a **measurement and security-characterization study**.

---

# 3. Core objects and notation

## 3.1 Base model

Let:

\[
\theta_0
\]

denote the public base model.

### Primary model

```text
Meta Llama 3.2 1B
```

The primary study uses the 1B base to remain feasible under MSc compute while supporting full white-box LoRA analysis.

### Replication model

```text
Pythia ~1B class
```

A second architecture is **not part of P0**.

It is added only under the expansion rules in §30.

---

## 3.2 Hidden constituent

Let the private task-specific constituent be represented by the induced LoRA update:

\[
\Delta W_A.
\]

The released constituent checkpoint itself is not available to the attacker.

When notation is lighter, the constituent is written:

\[
A.
\]

All parameter-recovery metrics are computed on **induced updates** \(\Delta W=BA\), not raw LoRA factor matrices, because the LoRA factorization is not unique.

---

## 3.3 Partners

Let:

\[
B_1,\ldots,B_k
\]

denote partner constituents or partner task updates used to create public descendants.

Partners are trained independently of the protected fine-tuning records of \(A\).

---

## 3.4 Public descendants

The public release family is:

\[
C_{1:k}=(C_1,\ldots,C_k).
\]

Each descendant contains the **same trained draw of \(A\)**.

The base training of \(A\) is not re-run for each descendant.

---

## 3.5 Lineage information

Let:

\[
Z
\]

denote attacker-visible side information.

Depending on the condition, \(Z\) may contain:

- \(\theta_0\);
- merge family;
- partner identities/checkpoints;
- coefficients;
- random masks;
- other released metadata.

---

## 3.6 Complete attacker observation

The full lineage observation is:

\[
R=(C_{1:k},Z).
\]

All L2 and L3 empirical capability definitions are relative to \(R\).

---

## 3.7 Recovered constituent

A recovery method:

\[
h\in\mathcal H
\]

maps:

\[
R \rightarrow \hat A=h(R).
\]

The induced recovered update is:

\[
\widehat{\Delta W}_A.
\]

---

# 4. Protected objects

Three objects must be kept separate in all code, plots, and writing.

## PO1 — Constituent confidentiality

Can the hidden update/function be reconstructed?

---

## PO2 — Training-record privacy

Can an attack distinguish protected fine-tuning members from non-members?

---

## PO3 — Contributor presence

Can the attacker infer that a contributor participated?

PO3 is secondary and not part of P0/P1 unless explicitly activated by the expansion rules.

---

# 5. Formal DP boundary

Let:

\[
A=\mathcal M(D_A)
\]

be one output of an \((\varepsilon,\delta)\)-DP training mechanism.

If:

\[
R=g(A,Z)
\]

and \(Z\) is independent of \(D_A\) conditional on \(A\), then \(R\) is joint post-processing of \(A\).

Likewise:

\[
\hat A=h(R)
\]

is post-processing.

Therefore:

> **Exact or approximate constituent recovery does not, by itself, violate the original DP guarantee.**

This study measures:

- constituent concealment;
- empirical audit power;
- practical attack sensitivity.

It does not redefine DP.

If merge selection, coefficient selection, partner selection, or release selection reads protected private data again, that condition is outside the core post-processing threat model and must be separately accounted for.

---

# 6. Primary data regime

## 6.1 Primary natural corpus

Use a **recent PubMed abstract corpus** whose records post-date the verified pretraining-data cutoff of the selected primary base model.

### Target sampling period

```text
2025 publication year
```

For Meta Llama 3.2 1B, Meta's official model card reports:

```text
pretraining data cutoff = December 2023
```

Therefore 2025 PubMed publications are post-cutoff for the primary base-model choice.

Before corpus acquisition, save the primary-source model-card evidence in:

```text
P0_PRE_00_SCORER_INTERFACE.md
P0_PRE_00_DRYRUN_TOLERANCE_CALIBRATION.json
P0_PRE_01_SYNTHETIC_ANALYSIS_INPUT.csv
P0_PRE_02_SYNTHETIC_ANALYSIS_RESULTS.json
P0_PRE_03_SYNTHETIC_ANALYSIS_ASSERTIONS.txt
P0_PRE_04_CROSSFIT_CONTROLLER_RESULTS.json
P0_PRE_05_CACHE_ASSERTIONS.txt
P0_PRE_06_CALENDAR_GATE.md
P0_00_COMPUTE_BUDGET.json
P0_00_BASE_MODEL_DATA_CUTOFF.md
```

If the primary base model changes, this assumption must be re-verified before the corpus is frozen.

Reason:

- reduces ambiguity from base-model pretraining membership;
- retains a realistic domain-adaptation setting;
- supports clean member/non-member construction;
- avoids relying on the original Pile membership assumption.

The exact PubMed query and acquisition script must be version-controlled.

---

## 6.2 Fixed corpus partitions

Create one immutable master corpus before any training.

Target sizes:

```text
TRAIN_CANDIDATES        = 30,000 natural records
CALIBRATION_NONMEMBERS = 5,000 natural records
EVAL_NONMEMBERS        = 5,000 natural records
RESERVE                 = 5,000 natural records
```

All records must be deduplicated by normalized exact text hash.

Near-duplicate filtering must be performed before split assignment.

The split seed is:

```text
20260820
```

Once created, these partitions do not change across training seeds.

---

## 6.3 Natural-member calibration and evaluation subsets

For each training run, select **uniformly at random without replacement** from trained **natural records only**:

```text
NATURAL_MEMBER_CALIBRATION = 2,000
NATURAL_MEMBER_EVAL        = 5,000
```

Canaries are excluded from both subsets.

`NATURAL_MEMBER_CALIBRATION` is used only for calibration-only model/view selection where member labels are required.

`NATURAL_MEMBER_EVAL` is held out for final natural-record attack evaluation.

The RNG seed for these selections is derived deterministically from the training seed.

The same candidate identities must be reused across all audit views within a seed.

---

## 6.4 Mandatory blind split / deduplication leakage check

Before any model-based natural-record privacy result is interpreted, test whether the fixed member/non-member split is distinguishable **from text alone**.

Because the natural corpus is drawn from one master corpus and assigned randomly, this control primarily checks for:

- exact/near-duplicate leakage;
- preprocessing artifacts;
- accidental split construction bias;
- unexpected topic/length imbalance.

Use:

```text
TF-IDF word 1–2 grams
+ L2-regularized logistic regression
```

Input:

- natural member-evaluation records;
- natural evaluation non-members.

The classifier receives no model-based features.

Evaluation:

- balanced labels;
- 5-fold stratified CV;
- primary metric: AUROC;
- first CV seed: `6104`;
- confirmatory CV seed: `6105`.

### Initial clean criterion

\[
|\mathrm{AUROC}-0.5|\le0.03.
\]

A 95% CI excluding \(0.5\) alone does **not** immediately stop the arm.

### Investigation trigger

If either:

- the first point estimate is outside \([0.47,0.53]\); or
- its 95% CI excludes \(0.5\),

run:

```text
BLIND_SPLIT_INVESTIGATION
```

consisting of:

1. repeat CV with seed `6105`;
2. audit exact duplicate hashes across partitions;
3. rerun the frozen near-duplicate check;
4. compare record length and publication-month distributions.

### Stop rule

Use:

```text
STOP_NATURAL_PRIVACY_ARM
```

only if:

1. a concrete duplicate/preprocessing leakage path is found; or
2. both fixed CV partitions have AUROC outside \([0.47,0.53]\) in the same direction.

Otherwise retain the natural arm and report the control with uncertainty.

The two CV seeds are **correlated robustness partitions of the same records**, not independent replications. Their purpose is to ensure that one arbitrary fold assignment does not determine the stop decision.

The text-only AUROC is a mandatory control row in every natural-record privacy table.

---

# 7. Canary design

## 7.1 Candidate canary pool

Construct:

```text
N_CANARY_CANDIDATES = 4,096
```

unique synthetic canaries.

Each canary must:

- be unique;
- appear at most once in the training set;
- contain a high-entropy secret component;
- have a fixed template family;
- avoid real PII;
- be generated before training;
- be fixed across views.

---

## 7.2 Inclusion mechanism

For each training seed \(s\):

\[
I_{j,s}\sim\operatorname{Bernoulli}(0.5)
\]

independently for each candidate canary \(j\).

If \(I_{j,s}=1\), the canary appears **exactly once** in the training corpus.

If \(I_{j,s}=0\), it does not appear.

Expected canary members per seed, and approximately the same number of excluded canaries:

\[
2{,}048.
\]

This synthetic load is intentionally high for audit power and is therefore tested for contamination of the natural-data arm in §7.5.

---

## 7.3 Repetition rule

The same canary must **not** be inserted multiple times in the sample-level DP arm.

Repeated canaries are allowed only in an explicitly separated:

```text
NON_DP_MEMORIZATION_CALIBRATION
```

or:

```text
GROUP_PRIVACY_EXPLORATION
```

arm.

Repeated-canary results cannot support sample-level DP claims.

---

## 7.4 Canary-arm cross-fitting and threshold calibration

Canary and natural-record FPR calibration are **separate statistical arms**.

For canaries:

- \(I_{j,s}=0\) canaries are the non-member controls;
- natural PubMed non-members are never used to set a canary threshold;
- canary IDs receive a fixed 5-fold assignment generated before training.

For each artifact/view and each seed:

1. hold out one canary fold as the **outer evaluation fold**;
2. use only the other four folds for all calibration-only choices, including:
   - descendant selection;
   - pooled-aggregator selection;
   - hyperparameter selection;
   - operational 1% FPR threshold estimation;
3. estimate the operational 1% FPR threshold using only **excluded canaries** from the four calibration folds;
4. apply the frozen selection and threshold to included and excluded canaries in the held-out fold;
5. repeat for all five folds;
6. aggregate out-of-fold scores and predictions.

Thus no held-out canary participates in choosing the descendant, pooled method, hyperparameters, or threshold used to score itself.

For natural records:

- descendant / pooled-method selection uses `NATURAL_MEMBER_CALIBRATION` plus `CALIBRATION_NONMEMBERS`;
- thresholds are calibrated on `CALIBRATION_NONMEMBERS`;
- final evaluation uses `NATURAL_MEMBER_EVAL` plus `EVAL_NONMEMBERS`.

The two arm-specific threshold distributions must never be mixed.

For every view report both:

1. the **realised operational FPR** and corresponding TPR from the calibration-only threshold; and
2. the ROC-interpolated TPR at exactly 1% FPR computed as an evaluation statistic from held-out scores.

Cross-view headline comparisons use the common-FPR interpolated `TPR@1%FPR`; operational TPR/FPR pairs are reported alongside it.

---

## 7.5 Canary-load contamination control

Train one additional **matched no-canary control** for the non-DP calibration regime.

Use:

```text
control seed = 101
```

with:

- the same natural training corpus;
- the same optimizer / LoRA hyperparameters;
- the same training and data-order seeds where technically possible;
- the same number of optimizer updates as the corresponding canary-containing run;
- **zero synthetic canaries**.

Compare the canary-containing and no-canary models on:

1. natural held-out NLL/perplexity;
2. natural-record attack-score distributions;
3. natural-record TPR@1%FPR/AUC;
4. distance-to-base.

The blind baseline in §6.4 is a property of the corpus split, not the trained model, so it is run once rather than redundantly per model.

Flag canary-load contamination if either holds:

1. the bootstrap 95% CI for relative natural held-out NLL change excludes 0 and the absolute relative change is at least 1%; or
2. the paired 95% CI for natural-record TPR@1%FPR change excludes 0 and the absolute change is at least 0.02.

If contamination is flagged, the confirmatory natural-data arm must either:

- reduce the canary fraction; or
- train separate models for high-powered canary auditing and natural-record privacy analysis.

No natural-data headline claim may silently rely on a contaminated training regime.

---

# 8. Training conditions

P0 uses two privacy regimes.

## 8.1 Calibration regime

Purpose:

> guarantee measurable privacy dynamic range before attempting recovery/privacy coupling.

Default:

```text
NON_DP LoRA
```

If non-DP memorization is still below the measurement-power gate, the data/training protocol must be redesigned before continuing.

---

## 8.2 Representative DP regime

P0 includes one representative DP-LoRA condition.

Initial target:

```text
epsilon target ≈ 8
```

with the exact achieved accountant output recorded.

This is not used to claim a universal epsilon frontier.

Its P0 purpose is a **representative confirmatory measurement-floor check**. A single \(\varepsilon\approx8\) point cannot locate the transition at which the privacy instrument goes dark.

If a DP transition is later mapped, at least one weaker-privacy point must be added and its target epsilon frozen **before** the confirmatory DP sweep is opened.

Broader epsilon sweeps are deferred until P1/P2 expansion.

### P0 DP seed allocation

P0 begins with:

```text
DP_SMOKE seed = 101
```

for implementation/accounting validation only.

A one-seed DP smoke result:

- cannot satisfy §18's across-seed privacy-eligibility rule;
- cannot enter inferential \(R_{\mathrm{priv}}\), \(G_L\), or \(G_R\);
- cannot support a confirmatory privacy-floor claim.

If the DP arm is retained after smoke validation, train:

```text
DP seeds = {101, 202, 303}
```

before any inferential DP-arm statement.

Thus:

```text
1 seed = smoke/plumbing only
3 seeds = minimum inferential DP set
```

---

## 8.3 LoRA configuration

Primary rank for P0:

```text
rank = 32
```

All LoRA implementation details must be fixed across seeds:

- target modules;
- scaling;
- dropout;
- optimizer;
- learning rate;
- number of epochs/steps;
- batch size;
- clipping;
- DP noise multiplier where applicable.

These values must be written into the experiment configuration file and hashed.

---

# 9. Training-seed design

## P0 calibration

Use:

```text
3 independent training seeds
```

for the non-DP calibration arm.

The representative DP arm follows the smoke-then-3-seed rule in §8.2.

Seed IDs:

```text
101
202
303
```

Each seed changes:

- parameter initialization where applicable;
- mini-batch order;
- DP noise;
- canary inclusion decisions.

The natural corpus partitions remain fixed.

---

## Confirmatory expansion

If P0 and P1 pass all gates, final confirmatory primary-model cells use:

```text
5 total training seeds
```

by adding:

```text
404
505
```

The extra seeds are added only to the confirmatory cells, not retroactively to every exploratory grid cell.

---

# 10. Merge/release families

## 10.1 Operator O1 — linear / task-arithmetic baseline

This is the information-preserving baseline.

It is not a novelty contribution.

---

## 10.2 Operator O2 — DARE

DARE is the primary information-removal merge operator.

It enters P1 only after passing the utility-realism gate in §24.

---

## 10.3 Operator O3 — SVD-TRUNC-MERGE fallback

`O3` is pre-registered **before P0-D is run**, so the project does not choose a replacement operator after learning whether DARE succeeds.

For each constituent task-vector matrix \(\tau\), define the rank-\(s\) truncated-SVD approximation:

\[
T_s(\tau)
=
U_{1:s}\Sigma_{1:s}V_{1:s}^{\top}.
\]

The fallback merge is:

\[
C^{\mathrm{SVD}}_i
=
\theta_0
+
\alpha_iT_s(\tau_A)
+
(1-\alpha_i)T_s(\tau_{B_i}).
\]

Because the primary LoRA rank is:

```text
r = 32
```

candidate retained ranks are:

```text
s ∈ {24, 16, 8, 4}
```

O3 is a **controlled lossy merge transformation** with:

- one explicit rank knob;
- known discarded singular directions;
- direct interaction with the low-rank structure used by the recovery baseline.

It is **not** claimed to be:

- novel;
- the dominant deployed merge operator;
- evidence that this exact truncation procedure is prevalent in public releases.

O3 is activated in the main P1 operator axis **only if DARE fails §24**.

Its gate is §24A.

If DARE passes, O3 is optional and does not automatically enter the main grid.

---

## 10.4 TIES

TIES is not part of initial P0/P1 implementation.

It enters P2 only if:

- P0/P1 pipelines are valid;
- the primary/fallback operator results are interpretable;
- implementation can separate operator effects from pipeline failure.

TIES is treated empirically, not as solved sparse-plus-low-rank theory.

---

## 10.5 Optional O4 — post-training quantization

Quantization may be added in P2 as a deployment-oriented release-channel robustness check if compute permits.

It is not used to rescue P1 and is not part of the frozen P1 registry.

---

# 11. Descendant count

P0/P1 use:

\[
k\in\{1,2,4\}.
\]

The value:

```text
k = 8
```

is deferred.

It is added only if the \(k=4\) curve has not saturated and additional lineage is scientifically informative.

---

# 12. Lineage-knowledge conditions

P0/P1 distinguish three authorized lineage regimes.

## Z-HIGH — high lineage information

Attacker knows:

- base model;
- merge operator;
- partner checkpoint(s);
- coefficient(s);
- mask where the operator publicly exposes it.

This is an oracle/control regime.

---

## Z-INCOMPLETE-FIXED — unknown partner, fixed known coefficient

Primary condition:

```text
partner unknown
merge family known
coefficient α fixed across descendants and known
```

This is the Spectral-DeTuning-compatible reproduction regime.

For unknown partners:

```text
k = 1  -> N/A / structurally underdetermined
k >= 2 -> evaluated
```

---

## Z-INCOMPLETE-VARYING-KNOWN — unknown partner, varying known coefficients

Activated only after `P0-C2` reproduces the fixed-coefficient baseline.

For linear/task-arithmetic descendants:

\[
D_i=C_i-\theta_0
=
\alpha_i\tau_A+(1-\alpha_i)\tau_{B_i}.
\]

When each nonzero \(\alpha_i\) is **known to the attacker**, divide by \(\alpha_i\):

\[
\widetilde D_i
=
\frac{D_i}{\alpha_i}
=
\tau_A
+
\frac{1-\alpha_i}{\alpha_i}\tau_{B_i}.
\]

Scalar multiplication preserves rank, so the partner term remains low rank.

Therefore known varying coefficients reduce to the same:

> shared source + per-descendant low-rank residual

family after deterministic rescaling.

Consequently:

> **known varying coefficients are not treated as a new inverse problem.**

They are a reproduction/stress-test of the common-source baseline under heterogeneous residual scales.

For unknown partners:

```text
k = 1 -> N/A
k >= 2 -> evaluated
```

---

## Z-INCOMPLETE-ALPHA-HIDDEN — deferred, not a P1 cell

If partner identities and coefficients are both unknown:

\[
D_i=\alpha_i\tau_A+U_i,
\qquad
\operatorname{rank}(U_i)\le r,
\]

the unrestricted problem contains scale ambiguity and may contain deeper non-identifiability.

Without an external scale anchor:

\[
\tau_A\rightarrow c\tau_A,
\qquad
\alpha_i\rightarrow\alpha_i/c
\]

can preserve the shared product wherever coefficient bounds permit.

Simply enforcing:

\[
\|\tau_A\|_F=1
\]

would choose an arbitrary representative and would not recover the true constituent scale required by the L3 parameter-error metric.

Therefore hidden-\(\alpha\) recovery is **not authorized in P1**.

It may be added later only if a new pre-result spec supplies non-oracle information sufficient to fix scale/ambiguity, such as:

- one known coefficient anchor;
- a finite public coefficient candidate set;
- a justified external norm prior;
- another independently defensible side constraint.

No hidden-\(\alpha\) method may use the true coefficient or true constituent norm during attack-time estimation.

---

# 13. Four audit views

Every eligible experimental cell must support as many of the following views as the condition permits.

## V0 — Oracle constituent

\[
V_A=A.
\]

Used only for evaluation and normalization.

---

## V1 — Strong single-descendant comparator

For each \(C_i\), compute the same attack score.

Define the single-descendant comparator by **calibration-only selection**, not by evaluation labels.

### Canary arm

Inside each outer cross-fitting fold:

1. score all descendants on the four calibration folds;
2. select the best descendant using only those calibration folds;
3. freeze that descendant;
4. evaluate it on the held-out fold.

### Natural arm

Select the best descendant using only:

- `NATURAL_MEMBER_CALIBRATION`;
- `CALIBRATION_NONMEMBERS`.

Then freeze it before final evaluation.

Denote the resulting evaluation statistic:

\[
M_{\mathrm{desc}}.
\]

This is intentionally a **strong single-artifact comparator**. Therefore the lineage gap is conservative: it lower-bounds the advantage over an auditor who possesses an arbitrary single released artifact.

---

## V2 — Pooled lineage

\[
V_R=R=(C_{1:k},Z).
\]

The pooled attack uses the same base per-artifact attack scores as V1 plus one pre-registered aggregation layer.

---

## V3 — Recovered constituent

\[
V_{\hat A}=\hat A=h(R).
\]

The same single-model attack code used for \(A\) and \(C_i\) is pointed at \(\hat A\).

---

# 14. Base privacy attack family

The study must avoid a naive-vs-engineered comparison.

All single-model views use the same attack code.

## 14.1 Primary score — reference-calibrated loss score

For record \(x\) and model \(V\):

\[
s_{\mathrm{ref}}(x;V)
=
\ell_{\mathrm{ref}}(x)
-
\ell_V(x).
\]

Where:

- \(\ell_V(x)\) is token-level average negative log likelihood under \(V\);
- \(\ell_{\mathrm{ref}}(x)\) is the same quantity under a fixed reference model.

Primary reference:

```text
the un-fine-tuned base model θ0
```

Any stronger reference ensemble added later must be applied identically to all single-model views.

---

## 14.2 Secondary score — Min-K%

Use one fixed \(K\) selected before evaluation.

Initial:

```text
K = 20%
```

A small calibration-only sensitivity check may compare:

```text
K ∈ {10%, 20%, 30%}
```

The selected value is then frozen.

### Implementation/storage rule

Min-K% requires per-token log-probabilities, but full token-level arrays are not persisted for the full ladder.

Within each scoring batch:

1. compute token-level log-probabilities;
2. apply the frozen token-validity mask;
3. compute the bottom-\(K\%\) scalar inline;
4. persist only:
   - record ID;
   - scalar Min-K% score;
   - token count;
   - minimal diagnostic metadata.

Full per-token arrays may be retained only for a small predeclared diagnostic sample.

This adds no model forward passes and prevents avoidable storage growth.

---

## 14.3 Raw loss

Raw loss is reported only as a weak baseline.

It is not the principal attack.

---

## 14.4 Distance-to-base diagnostic

Because the primary reference score uses the public base \(\theta_0\), every single-model view must also report its distance from the base:

\[
d_0(V)
=
\|\theta_V-\theta_0\|_F
\]

and, where the model is represented as an update:

\[
d_0(V)=\|\Delta W_V\|_F.
\]

Report:

- \(d_0(A)\);
- \(d_0(C_i)\);
- \(d_0(\hat A)\).

This is a diagnostic for the possibility that merging/DARE mechanically moves a model toward the fixed reference and shrinks a reference-loss score for geometric reasons unrelated to memorization.

The reference-independent Min-K% attack is the mandatory robustness check for any headline effect seen only with the fixed-base reference score.

---

# 15. Pooled-lineage attack

The pooled view must not be deliberately weak.

## 15.1 Arm-matched per-descendant normalization

Normalize each descendant's attack score using the **same statistical arm** as the records being evaluated.

### Canary arm

Within each outer cross-fitting fold, use only excluded canaries from the four calibration folds:

\[
z_i^{\mathrm{can}}(x)
=
\frac{
 s(x;C_i)-\mu^{\mathrm{can}}_{i,\mathrm{cal}}
}{
 \sigma^{\mathrm{can}}_{i,\mathrm{cal}}
}.
\]

### Natural-record arm

Use only `CALIBRATION_NONMEMBERS`:

\[
z_i^{\mathrm{nat}}(x)
=
\frac{
 s(x;C_i)-\mu^{\mathrm{nat}}_{i,\mathrm{cal}}
}{
 \sigma^{\mathrm{nat}}_{i,\mathrm{cal}}
}.
\]

Canary scores are never standardized using natural-record moments, and natural-record scores are never standardized using canary moments.

---

## 15.2 Primary fixed aggregator

Use a Stouffer-style mean:

\[
z_{\mathrm{mean}}(x)
=
\frac{1}{\sqrt{k}}
\sum_{i=1}^{k}z_i(x).
\]

This is deterministic and pre-registered.

---

## 15.3 Strong pooled comparator

Additionally fit one L2-regularized logistic pooling model:

\[
\phi_{\mathrm{ridge}}(z_1,\ldots,z_k).
\]

Requirements:

- L2 regularization;
- no held-out evaluation labels used for fitting or model selection;
- same feature construction across cells;
- no architecture-specific hand engineering after evaluation.

### Canary arm

Inside each outer fold, fit/tune/select using only the four calibration folds.

### Natural arm

Fit/tune/select using only:

- `NATURAL_MEMBER_CALIBRATION`;
- `CALIBRATION_NONMEMBERS`.

The stronger of the fixed Stouffer and ridge pooled methods is selected using **calibration performance only**, then applied to the corresponding held-out evaluation records.

This protects against a weak pooled baseline while preventing selection leakage.

---

# 16. Primary privacy endpoint

The principal cross-view privacy metric is:

\[
\boxed{\mathrm{TPR}@1\%\mathrm{FPR}}
\]

computed from held-out scores at a common 1% FPR operating point.

Arm-specific operational thresholds are separately calibrated as follows:

- canary arm: excluded canaries under the cross-fitting rule in §7.4;
- natural-record arm: `CALIBRATION_NONMEMBERS`, with realised FPR checked on `EVAL_NONMEMBERS`.

Synthetic canaries and natural non-members are never pooled to set a threshold.

For every view, report:

- common-FPR `TPR@1%FPR`;
- calibration-threshold operational TPR;
- realised operational FPR;
- AUC as secondary.

---

## 16.1 0.1% FPR

At the P0 canary scale of roughly 2,048 excluded canaries, 0.1% FPR is determined by roughly two observations and is **pre-declared unpowered**.

Therefore:

```text
TPR@0.1%FPR = DEFERRED IN P0
```

It may be added only in a later confirmatory stage with a substantially larger held-out non-member pool.

---

# 17. Privacy null and normalization

Let:

\[
M(V)
\]

denote TPR@1%FPR for view \(V\).

The null level is:

\[
M_0=0.01.
\]

Define the oracle privacy signal:

\[
\Delta_A=M(A)-M_0.
\]

---

# 18. Privacy-endpoint eligibility

A training condition is eligible for normalized privacy-recovery analysis only if the oracle constituent has measurable signal.

A condition is:

```text
PRIVACY_ENDPOINT_ELIGIBLE
```

iff both are true:

1. the bootstrap 95% CI for:
   \[
   M(A)-M_0
   \]
   excludes \(0\) on the positive side; and
2. median:
   \[
   M(A)\ge 0.05
   \]
   across the required training seeds.

The 5% TPR floor is a pre-registered dynamic-range requirement.

---

## 18.1 Ineligible condition

Otherwise label:

```text
PRIVACY_ENDPOINT_INELIGIBLE
```

and report:

- \(M(A)\);
- confidence interval;
- attack score distributions;
- one-run audit sanity check where applicable.

Do **not** assign:

```text
L2 = FALSE
```

to these cells.

---

# 19. Normalized privacy recovery

For an eligible condition:

\[
\boxed{
R_{\mathrm{priv}}(V)
=
\frac{M(V)-M_0}
{M(A)-M_0}
}
\]

Do not clamp \(R_{\mathrm{priv}}\) to \([0,1]\).

Endpoint checks:

\[
M(V)=M(A)
\Rightarrow
R_{\mathrm{priv}}=1
\]

and:

\[
M(V)=M_0
\Rightarrow
R_{\mathrm{priv}}=0.
\]

---

# 20. Audit-gap metrics

All audit-gap analyses report **paired raw differences first** and normalized gaps second.

The paired raw difference is the primary within-cell statistic because all views share the same oracle denominator while the numerator can be estimated directly with paired uncertainty.

## 20.1 Descendant recovery

\[
R_{\mathrm{desc}}
=
R_{\mathrm{priv}}(V1).
\]

## 20.2 Pooled-lineage recovery

\[
R_{\mathrm{pool}}
=
R_{\mathrm{priv}}(V2).
\]

## 20.3 Reconstructed-model recovery

\[
R_{\mathrm{rec}}
=
R_{\mathrm{priv}}(V3).
\]

---

## 20.4 Raw lineage advantage — primary within-cell statistic

\[
\boxed{
D_L
=
M_{\mathrm{pool}}-M_{\mathrm{desc}}
}
\]

Report \(D_L\) with a paired bootstrap 95% CI.

---

## 20.5 Normalized lineage value — secondary cross-cell statistic

\[
\boxed{
G_L
=
R_{\mathrm{pool}}-R_{\mathrm{desc}}
=
\frac{D_L}{M(A)-M_0}
}
\]

Its CI must propagate uncertainty in both numerator and denominator.

---

## 20.6 Raw reconstruction-routing advantage — primary within-cell statistic

\[
\boxed{
D_R
=
M_{\mathrm{rec}}-M_{\mathrm{pool}}
}
\]

Report \(D_R\) with a paired bootstrap 95% CI.

---

## 20.7 Normalized reconstruction-routing value

\[
\boxed{
G_R
=
R_{\mathrm{rec}}-R_{\mathrm{pool}}
=
\frac{D_R}{M(A)-M_0}
}
\]

Because \(\hat A=h(R)\), \(G_R>0\) must not be described as information creation. It is an attack-representation/adversary-instantiation effect.

---

## 20.8 Denominator stability

For every privacy-eligible training condition report:

\[
\mathrm{RSE}_{\Delta_A}
=
\frac{
\mathrm{SE}[M(A)-M_0]
}{
M(A)-M_0
}.
\]

Interpretation rule:

- if \(\mathrm{RSE}_{\Delta_A}\le0.20\), normalized cross-cell comparisons may be interpreted normally;
- if \(0.20<\mathrm{RSE}_{\Delta_A}\le0.30\), label normalized comparisons `DENOMINATOR_UNSTABLE`;
- if \(\mathrm{RSE}_{\Delta_A}>0.30\), do not use \(G_L/G_R\) for inferential cross-cell ranking; rely on raw paired \(D_L/D_R\).

This rule does not affect the validity of the raw paired differences.

---

# 21. One-run empirical DP audit

Use Steinke-style one-run auditing over the independent canary inclusion decisions as a **secondary readout**.

## Primary use

Estimate one empirical privacy lower bound for the training/release mechanism as a sanity check.

---

## Channel-specific optional readout

The same canary decisions may be scored through:

- \(A\);
- \(C_i\);
- pooled lineage where the auditing method supports a scalar score;
- \(\hat A\).

If empirical lower bounds are computed through these channels, label them:

```text
CHANNEL_SPECIFIC_EMPIRICAL_LOWER_BOUND
```

not:

```text
epsilon_of_artifact
```

The TPR@1%FPR comparison remains primary because the epsilon lower bound will be noisier at MSc-scale canary counts.

---

# 22. Parameter-recovery metric

The primary parameter metric is induced-update relative Frobenius error:

\[
\boxed{
e_F
=
\frac{
\|\widehat{\Delta W}_A-\Delta W_A\|_F
}{
\|\Delta W_A\|_F
}
}
\]

computed:

- globally;
- per layer.

---

## Secondary parameter metrics

Report:

1. cosine similarity of flattened induced updates;
2. layerwise relative error;
3. spectral error:
   \[
   \|\sigma(\widehat{\Delta W})-\sigma(\Delta W_A)\|;
   \]
4. maximum layer error;
5. error concentration across layers.

Raw LoRA factors are never used as the headline recovery metric.

---

# 23. Functional-recovery metrics

For \(A\) and every reconstructed/synthetic checkpoint, report:

1. held-out PubMed negative log likelihood;
2. perplexity;
3. per-record loss-rank correlation;
4. output/logit agreement on a fixed held-out prompt set.

Functional metrics are explanatory/secondary and are not assumed to be privacy proxies.

---

## 23A. Normalized functional recovery

For the fidelity-ladder analysis let:

\[
L(V)=\mathrm{NLL}_{\mathrm{heldout}}(V).
\]

Define:

\[
\Delta_L=L(\theta_0)-L(A).
\]

The NLL-based functional normalization is eligible only if all are true:

1. the paired bootstrap 95% CI for \(\Delta_L\) excludes \(0\) on the positive side;
2. the relative improvement satisfies:
   \[
   \frac{\Delta_L}{L(\theta_0)}\ge0.02;
   \]
3. denominator relative SE satisfies:
   \[
   \mathrm{RSE}_{\Delta_L}
   =
   \frac{\mathrm{SE}(\Delta_L)}{\Delta_L}
   \le0.20.
   \]

If eligible:

\[
\boxed{
R_{\mathrm{func}}(V)
=
\frac{
L(\theta_0)-L(V)
}{
L(\theta_0)-L(A)
}
}
\]

so:

\[
R_{\mathrm{func}}(A)=1,
\qquad
R_{\mathrm{func}}(\theta_0)=0.
\]

Do not clamp.

### Fallback logit metric

If NLL normalization is ineligible, let:

\[
S(V,A)
\]

be the predeclared mean logit-agreement score with oracle \(A\) on the fixed prompt set, with \(S(A,A)=1\).

Define:

\[
\Delta_S=1-S(\theta_0,A).
\]

The fallback is eligible only if:

\[
\Delta_S\ge0.02
\]

and the bootstrap 95% CI excludes \(0\).

Then:

\[
R_{\mathrm{func,logit}}(V)
=
\frac{
S(V,A)-S(\theta_0,A)
}{
1-S(\theta_0,A)
}.
\]

If neither function normalization is eligible:

```text
FUNCTION_ENDPOINT_INELIGIBLE
```

and no finite function/privacy \(\Delta e_{50}\) claim is made.

---

# 24. DARE utility-realism gate

Before DARE enters the main frontier experiment, run a utility sweep.

Candidate drop ratios:

```text
p ∈ {0.25, 0.50, 0.75, 0.90}
```

For each \(p\), compare the DARE descendant to the corresponding non-DARE merge.

Primary utility quantity:

\[
\Delta\mathrm{NLL}_{p}
=
\frac{
\mathrm{NLL}(C_p)-\mathrm{NLL}(C_{\mathrm{baseline}})
}{
\mathrm{NLL}(C_{\mathrm{baseline}})
}.
\]

Also report:

\[
R_{\mathrm{PPL},p}
=
\frac{
\mathrm{PPL}(C_p)
}{
\mathrm{PPL}(C_{\mathrm{baseline}})
}.
\]

A DARE setting is release-realistic iff:

\[
\Delta\mathrm{NLL}_{p}\le0.05.
\]

Define:

\[
p^*
=
\max\{p:\Delta\mathrm{NLL}_{p}\le0.05\}.
\]

The main DARE frontier uses:

```text
p = p*
```

and optionally one lower drop rate.

The candidate grid has a hard minimum:

```text
p_min = 0.25
```

Therefore:

```text
if no p >= 0.25 passes:
    DARE = REMOVE_FROM_MAIN_P1
```

## 24.1 DARE perturbation-strength diagnostic

For each \(p\), report the realised merged-delta distortion:

\[
e_{\mathrm{DARE},p}
=
\frac{
\|
\Delta W_{C_p}
-
\Delta W_{C_{\mathrm{baseline}}}
\|_F
}{
\|
\Delta W_{C_{\mathrm{baseline}}}
\|_F
}.
\]

For independent coordinate dropping/rescaling on a fixed delta, the reference RMS error factor is:

\[
e_{\mathrm{DARE,theory}}
=
\sqrt{\frac{p}{1-p}}.
\]

At:

\[
p=0.25,
\]

this equals:

\[
\sqrt{\frac{1}{3}}
\approx0.577.
\]

If:

```text
p* = 0.25
```

label:

```text
DARE_MINIMUM_GATE
```

and display both realised and theoretical perturbation strengths.

### Deliberate non-triviality-gate asymmetry

DARE does not receive a separate O3-style \(e\ge0.25\) non-triviality gate.

This is deliberate because the minimum registered DARE setting already has a large reference RMS perturbation, whereas mild SVD truncation can be arbitrarily close to identity.

The realised DARE distortion is still reported.

No privacy interpretation is allowed for a DARE artifact that fails the utility gate.

---

# 24A. O3 SVD-TRUNC-MERGE fallback gate

This gate is specified before the DARE outcome is known.

Execute it only if:

```text
DARE = REMOVE_FROM_MAIN_P1
```

unless O3 was separately budgeted as a secondary pre-result control.

Candidate retained ranks:

```text
s ∈ {24, 16, 8, 4}
```

For each \(s\), build the O3 merge from §10.3.

## 24A.1 Analytic protected-constituent information floor

For every LoRA-target matrix \(m\), let:

\[
\sigma_{m,1}\ge\sigma_{m,2}\ge\cdots
\]

be singular values of \(\Delta W_{A,m}\).

Define:

\[
\boxed{
e_{\mathrm{floor}}(s)
=
\sqrt{
\frac{
\sum_m\sum_{j>s}\sigma_{m,j}^2
}{
\sum_m\sum_j\sigma_{m,j}^2
}
}
}
\]

which equals:

\[
\frac{
\|T_s(\Delta W_A)-\Delta W_A\|_F
}{
\|\Delta W_A\|_F
}.
\]

This is the best achievable original-constituent error if only \(T_s(\Delta W_A)\) is released and no external information restores the discarded singular tail.

It is an operator-imposed information floor, not a solver error.

## 24A.2 Utility gate

Define:

\[
\Delta\mathrm{NLL}^{\mathrm{SVD}}_{s}
=
\frac{
\mathrm{NLL}(C^{\mathrm{SVD}}_s)
-
\mathrm{NLL}(C_{\mathrm{linear}})
}{
\mathrm{NLL}(C_{\mathrm{linear}})
}.
\]

Require:

\[
\Delta\mathrm{NLL}^{\mathrm{SVD}}_{s}\le0.05.
\]

Report the corresponding perplexity ratio.

## 24A.3 Non-triviality gate

Define:

\[
e_{\mathrm{O3},s}
=
\frac{
\|
\Delta W_{C^{\mathrm{SVD}}_s}
-
\Delta W_{C_{\mathrm{linear}}}
\|_F
}{
\|
\Delta W_{C_{\mathrm{linear}}}
\|_F
}.
\]

Require:

\[
e_{\mathrm{O3},s}\ge0.25.
\]

This prevents an almost-identity truncation from serving as the only information-removal operator.

Important:

\[
e_{\mathrm{O3},s}\ge0.25
\]

is measured on the complete merged delta and does **not** imply:

\[
e_{\mathrm{floor}}(s)\ge0.25
\]

for the protected constituent alone.

## 24A.4 Selected fallback rank

Choose the **smallest retained rank** \(s\) satisfying both:

\[
\Delta\mathrm{NLL}^{\mathrm{SVD}}_{s}\le0.05
\]

and:

\[
e_{\mathrm{O3},s}\ge0.25.
\]

Call it:

\[
s^*.
\]

If none passes:

```text
O3 = REMOVE_FROM_MAIN_P1
```

## 24A.5 O3 floor / solver reporting rule

For selected \(s^*\), report all of the following.

### Absolute error to the original constituent

\[
e_F
=
\frac{
\|\widehat{\Delta W}_A-\Delta W_A\|_F
}{
\|\Delta W_A\|_F
}.
\]

### Analytic operator floor

\[
e_{\mathrm{floor}}(s^*).
\]

### Solver error to the released truncated constituent

\[
\boxed{
e_{\mathrm{solver,origscale}}
=
\frac{
\|
\widehat{\Delta W}_A-T_{s^*}(\Delta W_A)
\|_F
}{
\|\Delta W_A\|_F
}
}
\]

and:

\[
e_{\mathrm{solver,retained}}
=
\frac{
\|
\widehat{\Delta W}_A-T_{s^*}(\Delta W_A)
\|_F
}{
\|T_{s^*}(\Delta W_A)\|_F
}.
\]

Do **not** define solver excess as:

\[
e_F-e_{\mathrm{floor}},
\]

because that subtraction assumes a geometric decomposition that an imperfect reconstruction need not satisfy.

### Threshold reachability

For any display threshold \(\tau\):

```text
L3_ORIGINAL_REACHABLE(s*) = [e_floor(s*) <= τ]
```

If:

\[
e_{\mathrm{floor}}(s^*)>\tau,
\]

annotate:

```text
OPERATOR_FLOOR_EXCEEDS_TAU
```

rather than calling the thresholded L3 failure a solver failure.

## 24A.6 Effective-rank diagnostic

For every O3 descendant and corresponding linear merge, report per target matrix:

1. theoretical rank upper bound;
2. numerical effective rank;
3. stable rank.

Numerical rank uses:

\[
\sigma_j/\sigma_1\ge10^{-6}.
\]

Stable rank:

\[
r_{\mathrm{stable}}(W)
=
\frac{\|W\|_F^2}{\|W\|_2^2}.
\]

For a two-parent merge:

```text
linear theoretical rank <= 2r = 64
O3 theoretical rank     <= 2s*
```

per target matrix before accidental subspace overlap.

O3 may simultaneously destroy parameter energy and simplify the inverse problem. Effective/stable rank is therefore the first pre-registered explanation for unexpectedly easy O3 recovery.

## 24A.7 Operator-axis contingency

If DARE fails and O3 passes:

```text
P1_PRIMARY_LOSSY_OPERATOR = SVD_TRUNC_MERGE
```

If DARE passes:

```text
P1_PRIMARY_LOSSY_OPERATOR = DARE
```

If both fail:

```text
OPERATOR_AXIS_INSUFFICIENT
```

Then:

- no unregistered operator is added after seeing results;
- the study does not claim an empirical comparison “across merge transformations” using real operator artifacts;
- the synthetic ladders, linear controls, audit-gap analysis, and other valid measurements remain;
- operator-specific RQ1 conclusions are narrowed before interpretation.

---

# 25. Recovery baselines

Recovery is divided into distinct regimes.

## P0-C1 — Known partner + known coefficient

For:

\[
C=\alpha A+(1-\alpha)B,
\]

recover:

\[
\hat A
=
\frac{C-(1-\alpha)B}{\alpha}.
\]

This is an implementation oracle.

### Pass criterion

Compute in float32 and require:

\[
e_F\le10^{-5}.
\]

Failure means merge/parameter accounting is incorrect.

---

## P0-C2 — Unknown partners + fixed known coefficient

Use a Spectral-DeTuning-style shared-source / low-rank-residual baseline.

Purpose:

> reproduce the occupied common-source regime.

This is not a novelty result.

---

## P0-C2B — Unknown partners + varying known coefficients

For each descendant:

\[
\widetilde D_i
=
\frac{C_i-\theta_0}{\alpha_i}
=
\tau_A+\beta_i\tau_{B_i},
\]

where:

\[
\beta_i=\frac{1-\alpha_i}{\alpha_i}.
\]

Apply the same frozen P0-C2 shared-source recovery implementation.

No new optimizer is introduced.

### Sanity control

Before privacy evaluation, verify numerically that rescaling a synthetic known-varying-\(\alpha\) family produces the expected common-source form to float tolerance.

### Naive comparator

Also run:

```text
NAIVE_UNRESCALED_FIXED_ALPHA
```

Failure of the rescaled baseline to improve on this comparator under material coefficient heterogeneity triggers implementation review, not an identifiability claim.

## P0-C2B.1 Coefficient schedules

### k = 2 reference schedule

```text
α = {0.35, 0.65}
```

Then:

\[
\beta
=
\{1.857,\ 0.538\},
\qquad
\rho_\beta\approx3.45.
\]

Maximum coefficient-rescaling amplification:

\[
a_\alpha
=
\max_i\frac{1}{|\alpha_i|}
\approx2.86.
\]

### k = 4 spread-matched schedule

```text
α = {0.35, 0.45, 0.55, 0.65}
```

Then:

\[
\beta
=
\{1.857,\ 1.222,\ 0.818,\ 0.538\},
\qquad
\rho_\beta\approx3.45.
\]

Thus the analytic beta spread is matched to the \(k=2\) condition while descendant count changes.

### k = 4 wide-spread schedule

```text
α = {0.25, 0.40, 0.60, 0.75}
```

Then:

\[
\beta
=
\{3.000,\ 1.500,\ 0.667,\ 0.333\},
\qquad
\rho_\beta=9.0.
\]

and:

\[
a_\alpha=4.0.
\]

This is the explicit heteroscedastic stress condition.

## P0-C2B.2 Partner-norm matching guard

Select partner sets using parameter information only, before privacy outcomes are viewed.

For C08/C09M/C09W require:

\[
\frac{
\max_i\|\tau_{B_i}\|_F
}{
\min_i\|\tau_{B_i}\|_F
}
\le1.25.
\]

No partner is selected using membership/privacy outcomes.

If this cannot be satisfied:

```text
PARTNER_NORM_MATCH_FAILED
```

and \(k\)-versus-heteroscedasticity comparisons are interpreted with realised residual spread explicitly rather than as cleanly isolated factors.

## P0-C2B.3 Stress-test metrics

Report:

### Analytic beta spread

\[
\rho_\beta
=
\frac{
\max_i|\beta_i|
}{
\min_i|\beta_i|
}.
\]

### Realised residual-norm spread

\[
\rho_{\mathrm{resid}}
=
\frac{
\max_i
\|\beta_i\tau_{B_i}\|_F
}{
\min_i
\|\beta_i\tau_{B_i}\|_F
}.
\]

### Maximum rescaling amplification

\[
a_\alpha
=
\max_i\frac{1}{|\alpha_i|}.
\]

Also report:

- recovery error \(e_F\);
- conditioning \(\kappa\);
- functional recovery;
- privacy recovery where eligible.

## P0-C2B.4 Intended contrasts

Primary \(k\) contrast:

```text
C08  : k=2, beta spread ≈ 3.45
C09M : k=4, beta spread ≈ 3.45
```

Primary heteroscedasticity contrast:

```text
C09M : k=4, beta spread ≈ 3.45
C09W : k=4, beta spread = 9.0
```

If realised \(\rho_{\mathrm{resid}}\) differs materially despite analytic matching, include it as an explanatory covariate and do not attribute the full difference to \(k\).

---

## P0-C3 — Open operator regimes

After C1/C2/C2B are validated, the project may study:

- DARE;
- later TIES;
- incomplete masks;
- more complex multi-parent conditions.

Hidden coefficients with unknown partners remain deferred under §12 unless an additional scale-fixing information source is separately specified.

The contribution of P1 does **not** depend on inventing a new inverse solver. Its primary scientific object remains the measured relationship among:

- recovery error;
- privacy recovery;
- operator structure;
- lineage-aware audit power.

---

## P0-C4 — O3 incomplete-lineage recovery, conditional

Activate only if:

```text
P1_PRIMARY_LOSSY_OPERATOR = SVD_TRUNC_MERGE
```

For fixed known \(\alpha\):

\[
C_i^{\mathrm{SVD}}-\theta_0
=
\alpha T_s(\tau_A)
+
(1-\alpha)T_s(\tau_{B_i}).
\]

The common source is:

\[
T_s(\tau_A),
\]

and every partner residual has rank at most \(s\).

Therefore the same shared-source low-rank recovery family used in P0-C2 applies.

For varying known \(\alpha_i\):

\[
\frac{
C_i^{\mathrm{SVD}}-\theta_0
}{
\alpha_i
}
=
T_s(\tau_A)
+
\frac{1-\alpha_i}{\alpha_i}T_s(\tau_{B_i}),
\]

so the same P0-C2B rescaling applies.

If this sanity reproduction passes:

```text
C14/C15/C17/C18 = RUN_REPRODUCTION
```

The solver target is \(T_s(\tau_A)\), while absolute \(e_F\) remains measured against \(\tau_A\).

If DARE is selected, those incomplete-lineage lossy cells remain `METHOD_GATED` unless a DARE-compatible recovery method is frozen before privacy outcomes.

---

# 26. Conditioning measurement

For descendant-direction vectors:

\[
D_i=\operatorname{vec}(C_i-B_i)
\]

in conditions where \(B_i\) is known, form the direction matrix:

\[
\mathcal D
=
[D_1,\ldots,D_k].
\]

Report relevant singular values.

For \(k=2\):

\[
\boxed{
\kappa
=
\frac{\sigma_1(\mathcal D)}
{\sigma_2(\mathcal D)}
}
\]

when \(\sigma_2>0\).

For \(k>2\), use:

\[
\kappa
=
\frac{
\sigma_{\max}
}{
\sigma_{\min,\mathrm{nonzero}}
}.
\]

Also report the effective numerical rank under a fixed tolerance.

Conditioning is an explanatory variable for recovery error.

It is not itself a novelty claim.

---

# 27. Canary-versus-natural construct validity

P0 separately measures privacy persistence for:

- canaries;
- natural records.

For each group, compute score changes across:

\[
A
\rightarrow
C_i
\rightarrow
\hat A.
\]

---

## 27.1 Primary validity test

Fit a pre-specified mixed-effects model on standardized privacy scores:

```text
score ~ record_type * view + C(seed)
```

where:

```text
record_type ∈ {canary, natural}
view ∈ {A, descendant, recovered}
C(seed) = seed entered as a fixed/blocking factor
```

The interaction:

```text
record_type × view
```

tests whether canaries and natural records respond differently to transformation.

### Pairing asymmetry

The two record types differ across seeds:

- natural evaluation identities are fixed/reused across seeds;
- canary inclusion membership is independently re-drawn by Bernoulli(0.5) per seed.

Therefore analyses are computed within seed first, preserving record identity where it exists, and seed-specific effects are then combined under the fixed/blocking-factor analysis.

The model must not imply the same across-seed member pairing for canaries and natural records.

---

## 27.2 Divergence rule

Treat canary behavior as materially divergent if:

1. the interaction is statistically significant at:
   \[
   \alpha=0.05;
   \]
   and
2. the standardized interaction effect size is:
   \[
   |d|\ge0.3.
   \]

---

## 27.3 Consequence branch

If canaries and natural records do **not** materially diverge:

- canaries may be used as a high-powered audit instrument;
- natural-record results must still be reported.

If they **do** diverge:

> canaries support only canary/audit-power claims.

All claims about natural training-record privacy must be supported by the natural-record arm.

P0-A must therefore establish measurement power separately for canaries and natural records.

---

# 28. P0 measurement-power gate

For each of the 3 calibration seeds, evaluate the oracle \(A_s\).

Measure:

- canary TPR@1%FPR;
- natural-record TPR@1%FPR;
- bootstrap CIs.

---

## 28.1 Canary gate

Canary privacy measurement is considered powered if the aggregate across seeds satisfies the eligibility rule in §18.

---

## 28.2 Natural-record gate

Natural-record privacy measurement is considered powered if the aggregate across seeds satisfies the same eligibility rule.

---

## 28.3 Kill/redesign rule

If **neither** canary nor natural oracle signal is eligible in the non-DP calibration arm:

```text
STOP_PRIVACY_GRID
```

Do not add operators, epsilons, datasets, or models.

Redesign the privacy instrument/training calibration first.

---

## 28.4 Limited continuation

If canaries are eligible but natural records are not:

- the parameter/privacy measurement may continue for canary behavior;
- no natural-record privacy claim is allowed;
- natural-record instrumentation must be improved before a natural-data headline claim.

---

# 28A. P0-F — Synthetic fidelity ladders

P0 includes two controlled fidelity ladders generated from the exact/oracle constituent after `P0-C1`, plus a separate zero-delta failure anchor.

No additional model training is required.

## 28A.1 Stage-1 coarse target levels

Use:

```text
e_target ∈ {
  0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30,
  0.40, 0.60, 0.80, 1.00, 1.50, 2.00
}
```

The oracle provides the exact:

```text
e = 0
```

anchor.

For each seed, level, and ladder type:

```text
N_PERTURBATION_DRAWS = 3
```

Stage-1 synthetic count:

```text
13 × 3 × 3 × 2 = 234 artifacts
```

Synthetic draws are calibration replicates, not independent training seeds.

## 28A.2 Primary LoRA-rank-matched ladder

For each LoRA-target matrix:

\[
d_{\mathrm{out}}\times d_{\mathrm{in}},
\]

sample:

\[
B'_q\in\mathbb R^{d_{\mathrm{out}}\times r},
\qquad
A'_q\in\mathbb R^{r\times d_{\mathrm{in}}},
\]

with:

```text
r = 32
```

and independent Gaussian entries.

Let:

\[
\Xi_q^{\mathrm{rank}}=B'_qA'_q.
\]

Globally normalize across all target matrices:

\[
\widetilde\Xi_q^{\mathrm{rank}}
=
\frac{\Xi_q^{\mathrm{rank}}}
{\|\Xi_q^{\mathrm{rank}}\|_F}.
\]

Construct:

\[
\widehat{\Delta W}^{\mathrm{rank}}_{e,q}
=
\Delta W_A
+
e_{\mathrm{target}}
\|\Delta W_A\|_F
\widetilde\Xi_q^{\mathrm{rank}}.
\]

Require realised relative error:

\[
e_{\mathrm{target}}\pm10^{-4}.
\]

This is the primary magnitude control.

## 28A.3 Secondary full-rank isotropic ladder

Sample entrywise:

\[
\Xi_q^{\mathrm{iso}}\sim\mathcal N(0,1)
\]

over the same target tensors, globally normalize, and construct:

\[
\widehat{\Delta W}^{\mathrm{iso}}_{e,q}
=
\Delta W_A
+
e_{\mathrm{target}}
\|\Delta W_A\|_F
\widetilde\Xi_q^{\mathrm{iso}}.
\]

Require the same realised-error tolerance.

This is the secondary structural control.

## 28A.4 Zero-delta / base failure anchor

A recovery that returns no constituent update has:

\[
\widehat{\Delta W}_{\mathrm{zero}}=0
\]

and therefore:

\[
\frac{\|0-\Delta W_A\|_F}{\|\Delta W_A\|_F}=1.
\]

Score this anchor once per seed.

Do **not** equate it with a synthetic \(e=1\) perturbation.

At equal \(e=1\):

- zero-delta anchor has \(\widehat{\Delta W}=0\);
- synthetic ladders have \(\widehat{\Delta W}=\Delta W_A+\Xi\).

The same error magnitude can therefore correspond to very different parameter-space locations.

## 28A.5 Measurements at every rung

Compute:

- parameter metrics;
- held-out NLL/perplexity;
- eligible normalized functional recovery;
- per-record loss-rank agreement;
- reference-calibrated privacy score;
- mandatory Min-K% privacy score;
- TPR@1%FPR;
- eligible \(R_{\mathrm{priv}}\);
- distance-to-base.

## 28A.6 Controlled perturbation-structure effect

At matched \(e\):

\[
\Delta_{\mathrm{structure}}(e)
=
R_{\mathrm{priv}}^{\mathrm{rank}}(e)
-
R_{\mathrm{priv}}^{\mathrm{iso}}(e).
\]

Report the same comparison for functional recovery.

At \(e=1\), compare both synthetic ladders with the zero-delta anchor.

## 28A.7 Calibration support / no extrapolation

Stage-1 support is:

\[
e\in[0,2.0].
\]

For any operator point outside calibrated support:

```text
u_j = UNDEFINED_BEYOND_CALIBRATION
```

Do not extrapolate isotonic, spline, or other calibration fits.

Such points are reported raw but excluded from matched-calibration residual inference.

## 28A.8 Stage-2 crossing refinement

Stage 2 estimates 50% decay locations without allowing coarse-grid spacing or one noisy seed to determine the bracket.

This is calibration refinement, not adaptive hypothesis searching.

### 28A.8.1 Seed-balanced rung values

For each ladder type, eligible endpoint, and Stage-1 error level \(e\):

1. average the three perturbation draws within each seed:
   \[
   \bar R_s(e);
   \]
2. compute the equally weighted seed-balanced rung value:
   \[
   \bar R_{\mathrm{pool}}(e)
   =
   \frac{1}{3}
   \sum_{s\in\{101,202,303\}}
   \bar R_s(e).
   \]

### 28A.8.2 Pooled monotone bracket

Fit one isotonic curve to the seed-balanced Stage-1 rung values:

\[
\hat f_{\mathrm{pool}}(e).
\]

Define the refinement bracket **only from this pooled isotonic fit**, never from raw noisy rung values.

Find the first adjacent Stage-1 pair:

\[
[e_L,e_U]
\]

with:

\[
\hat f_{\mathrm{pool}}(e_L)>0.5,
\qquad
\hat f_{\mathrm{pool}}(e_U)\le0.5.
\]

This creates one common bracket per:

```text
ladder type × endpoint
```

rather than a different grid for every seed.

### 28A.8.3 Added levels

Add:

```text
4 interior geometrically spaced levels
```

inside each unique bracket.

If privacy and function share a bracket, share the levels.

Deduplicate all new levels.

Maximum:

```text
≤ 8 added levels per ladder type
```

Each added level uses:

```text
3 perturbation draws × 3 seeds
```

and the **same refined levels across every seed**.

### 28A.8.4 Crossing estimation after refinement

After refinement:

1. refit seed-specific isotonic curves on the augmented grid;
2. report each seed's crossing;
3. report the seed-balanced pooled crossing;
4. estimate uncertainty using bootstrap resampling that preserves seed blocks and perturbation-draw structure.

### Edge cases

If the pooled isotonic curve is already below 0.5 at \(e=0.01\), add four geometrically spaced levels in:

\[
[0.001,0.01].
\]

If it remains above 0.5 at \(e=2.0\):

```text
RIGHT_CENSORED_AT_2.0
```

Do not extend farther in P0.

If an endpoint is ineligible:

```text
NO_REFINEMENT_FOR_INELIGIBLE_ENDPOINT
```

If required rank-matched refinement is omitted under the frozen compute-priority rule:

```text
COMPUTE_LIMITED_NOT_ESTIMATED
```

rather than a censoring label.

---

## 28A.9 Role

The rank-matched ladder is the primary magnitude calibration.

The isotropic ladder quantifies dependence on perturbation geometry.

Real operator-induced reconstructions are compared against the rank-matched curve at matched error only within calibrated support.

---

# 28B. Conditional O3 spectral-truncation ladder

Run only if:

```text
P1_PRIMARY_LOSSY_OPERATOR = SVD_TRUNC_MERGE
```

The §28A ladders add perturbation energy. O3 removes singular directions. Therefore §28A is not the primary calibration null for O3.

## 28B.1 Retained-rank schedule

Use:

```text
s ∈ {32, 31, 28, 24, 20, 16, 12, 8, 6, 4, 3, 2, 1, 0}
```

where:

- `s=32` reuses oracle \(A\);
- `s=0` reuses the zero-delta/base anchor;
- the 12 interior ranks require new scoring artifacts.

For each seed:

\[
\Delta W_s^{\mathrm{trunc}}
=
T_s(\Delta W_A).
\]

No perturbation draws are required.

## 28B.2 Exact error coordinate

Each point has exact:

\[
e_{\mathrm{floor}}(s)
=
\frac{
\|T_s(\Delta W_A)-\Delta W_A\|_F
}{
\|\Delta W_A\|_F
}.
\]

Score:

\[
\left(
e_{\mathrm{floor}}(s),
R_{\mathrm{priv}}^{\mathrm{trunc}}(s)
\right)
\]

and, where eligible:

\[
\left(
e_{\mathrm{floor}}(s),
R_{\mathrm{func}}^{\mathrm{trunc}}(s)
\right).
\]

## 28B.3 Primary O3 solver decomposition

For selected \(s^*\), use:

\[
T_{s^*}(\Delta W_A)
\]

as the operator-only reference.

Privacy solver effect:

\[
\boxed{
\Delta_{\mathrm{solver,priv}}
=
R_{\mathrm{priv}}(\widehat A)
-
R_{\mathrm{priv}}(T_{s^*}(A))
}
\]

Function solver effect:

\[
\Delta_{\mathrm{solver,func}}
=
R_{\mathrm{func}}(\widehat A)
-
R_{\mathrm{func}}(T_{s^*}(A)).
\]

These isolate recovery-pipeline effects **after** accounting for information already removed by O3.

For O3, the additive-ladder residual is not the primary operator residual.

## 28B.4 Named spectral-tail mechanism analysis

Use the rank-matched additive ladder as a same-magnitude comparator.

Define:

\[
\boxed{
\Delta_{\mathrm{tail,priv}}(s)
=
R_{\mathrm{priv}}^{\mathrm{trunc}}(s)
-
\hat f_{\mathrm{priv}}^{\mathrm{rank}}
\left(
e_{\mathrm{floor}}(s)
\right)
}
\]

and:

\[
\Delta_{\mathrm{tail,func}}(s)
=
R_{\mathrm{func}}^{\mathrm{trunc}}(s)
-
\hat f_{\mathrm{func}}^{\mathrm{rank}}
\left(
e_{\mathrm{floor}}(s)
\right).
\]

Name this analysis:

```text
SPECTRAL_TAIL_PRIVACY
```

Interpretation:

- \(\Delta_{\mathrm{tail,priv}}<0\): truncating the bottom singular directions destroys more privacy signal than generic rank-matched error of equal magnitude; this is consistent with privacy-relevant information being disproportionately represented in discarded tail directions;
- \(\Delta_{\mathrm{tail,priv}}>0\): spectral-tail removal preserves more privacy signal than generic matched error;
- near zero: total low-rank error magnitude explains the effect without a strong tail-specific deviation.

This is a controlled mechanism measurement, not a universal causal claim about singular directions.

### Mandatory reference-free corroboration

Because truncation lowers distance-to-base while the additive rank-matched ladder generally increases it, the fixed-\(\theta_0\) reference-calibrated score can confound spectral-tail location with base-distance geometry.

Therefore `SPECTRAL_TAIL_PRIVACY` must be computed **twice**:

1. using the primary reference-calibrated loss score; and
2. using the predeclared Min-K% score.

Define:

\[
\Delta_{\mathrm{tail,priv}}^{\mathrm{ref}}(s)
\]

and:

\[
\Delta_{\mathrm{tail,priv}}^{\mathrm{MinK}}(s).
\]

Interpretation rule:

```text
same sign + compatible magnitude:
    tail interpretation is corroborated

reference score only:
    label BASE_GEOMETRY_SENSITIVE
    do not make a spectral-tail privacy claim

Min-K% only:
    report score-family disagreement
    investigate attack-specific sensitivity

opposite signs:
    NO_STABLE_TAIL_INTERPRETATION
```

A headline `SPECTRAL_TAIL_PRIVACY` claim requires that the reference-calibrated and Min-K% analyses agree in direction.

## 28B.5 Effective-rank companion analysis

For each \(s\), report:

- theoretical rank bound;
- numerical effective rank;
- stable rank;
- \(e_{\mathrm{floor}}(s)\);
- privacy recovery;
- function recovery.

Lower rank may make common-source recovery easier even while parameter energy is removed.

---

# 29. Primary inferential analysis

The thresholded 2×2 table is **not** the primary inferential object.

The primary inference combines:

1. the controlled rank-matched fidelity ladder;
2. the controlled isotropic fidelity ladder; and
3. real operator-induced reconstruction points.

For every privacy-eligible observation \(j\), record:

\[
(e_j,r_j)
\]

where:

\[
e_j=e_{F,j},
\qquad
r_j=R_{\mathrm{priv},j}.
\]

Metadata include:

- operator;
- \(k\);
- lineage regime;
- seed;
- conditioning;
- DARE rate where applicable;
- functional fidelity;
- distance-to-base ratio;
- error-location features.

---

## 29.1 Primary analysis A — controlled magnitude calibration

Estimate both seed-specific curves and the seed-balanced pooled curve defined in §28A.8.

For each seed \(s\), estimate the **primary rank-matched** monotone curve:

\[
\hat f^{\mathrm{rank}}_s(e)
\approx
E[R_{\mathrm{priv}}\mid e,\text{rank-matched synthetic error},s].
\]

Estimate the isotropic curve separately:

\[
\hat f^{\mathrm{iso}}_s(e).
\]

Report:

1. Spearman rank correlation between \(e\) and \(R_{\mathrm{priv}}\);
2. rank-matched isotonic-regression curve;
3. isotropic isotonic-regression curve;
4. bootstrap confidence bands;
5. absolute prediction error within each ladder;
6. \(\Delta_{\mathrm{structure}}(e)\) at matched error;
7. corresponding normalized functional-recovery curves.

Synthetic perturbation draws are not treated as independent training replicates. Seed remains the unit of training replication.

---

## 29.2 Primary analysis B — operator deviation at matched error

For every real recovery point \(j\) from seed \(s(j)\), compute:

\[
u_j
=
r_j
-
\hat f^{\mathrm{rank}}_{s(j)}(e_j).
\]

Interpretation:

- \(u_j\approx0\): privacy recovery is consistent with rank-matched error magnitude alone;
- \(u_j>0\): more privacy recovery survives than rank-matched synthetic error of the same magnitude predicts;
- \(u_j<0\): less privacy recovery survives than the rank-matched calibration predicts.

This is the primary operator-level test for DARE and other operator points for which the rank-matched additive ladder is geometrically appropriate.

### O3 calibration exception

If O3 is selected, its primary calibration is §28B's deterministic truncation reference.

The rank-matched additive comparison is retained only as the named `SPECTRAL_TAIL_PRIVACY` mechanism analysis.

### Mandatory Min-K% parallel readout

Repeat the complete matched-calibration residual analysis using the predeclared Min-K% score:

\[
u^{\mathrm{MinK}}_j
=
R^{\mathrm{MinK}}_{\mathrm{priv},j}
-
\hat f^{\mathrm{rank,MinK}}_{s(j)}(e_j).
\]

This is mandatory, not conditional.

Any headline operator residual must be reported together with the Min-K% direction/effect.

---

## 29.3 Structural explanation of operator residuals

For DARE/additive-calibrated points, analyze \(u_j\) using:

```text
residual_privacy ~ operator + lineage + log_conditioning + log_distance_to_base_ratio + C(seed)
```

Do not use this regression to claim distance-to-base separates O3 from operator identity: O3 truncation deterministically lowers \(d_0\), making the covariate potentially collinear with the O3 indicator.

O3 is handled by §28B's matched truncation control.

where seed is a fixed/blocking factor.

Primary uncertainty is obtained by paired/clustered bootstrap respecting seed and release-family structure.

With only three training seeds, do not estimate a random seed variance component.

---

## 29.4 Predictive comparison

As a secondary summary compare:

### Magnitude-only model

```text
privacy_recovery ~ parameter_error + C(seed)
```

### Structural model

```text
privacy_recovery ~ parameter_error + operator + lineage + log_conditioning + log_distance_to_base_ratio + C(seed)
```

Use grouped cross-validation that never splits descendants/reconstructions from the same training seed across train and validation folds.

With three seeds this is descriptive sensitivity analysis, not the sole inferential basis.

In the five-seed confirmatory stage, use leave-one-seed-out prediction.

---

## 29.5 Functional-versus-privacy decay analysis

Using the synthetic ladders, plot on the same error axis:

\[
R_{\mathrm{func}}(e)
\]

and:

\[
R_{\mathrm{priv}}(e).
\]

For the primary rank-matched ladder estimate:

\[
e^{\mathrm{func}}_{50}
=
\inf\{e:R_{\mathrm{func}}(e)\le0.5\}
\]

and:

\[
e^{\mathrm{priv}}_{50}
=
\inf\{e:R_{\mathrm{priv}}(e)\le0.5\}.
\]

Define:

\[
\boxed{
\Delta e_{50}
=
e^{\mathrm{priv}}_{50}
-
e^{\mathrm{func}}_{50}
}
\]

and estimate a paired bootstrap 95% CI.

Interpretation:

- \(\Delta e_{50}>0\): privacy capability survives to larger parameter error than task-function fidelity;
- \(\Delta e_{50}<0\): privacy capability collapses earlier than task-function fidelity;
- \(\Delta e_{50}\approx0\): the two decay on similar scales.

Use the Stage-2 refinement rule in §28A.8 before estimating a finite crossing.

If either curve remains above 0.5 through \(e=2.0\), report that crossing as **right-censored at 2.0**.

Never extrapolate beyond calibrated support.

Repeat as a secondary analysis for the isotropic ladder.

This controlled metric-decoupling analysis is scientifically interpretable even if no merge operator produces a large residual.

---

# 30. Error-location mechanism analysis

This analysis is activated if real operator-induced points show meaningful deviation from the synthetic magnitude curve.

Do not run it as a fishing expedition if operator residuals are consistently near zero.

The synthetic fidelity ladder already supplies the magnitude control; §30 asks whether the **location/direction** of real reconstruction error explains the remaining residual.

Candidate pre-specified location features:

1. layerwise error vector;
2. fraction of total error concentrated in top 10% of layers;
3. residual projection onto leading singular directions of \(\Delta W_A\);
4. residual alignment with member-example gradient directions on a fixed diagnostic subset;
5. attention/MLP block localization.

The mechanism question is:

> At comparable total parameter error, does where the error lies predict privacy recovery?

If O3 is activated, `SPECTRAL_TAIL_PRIVACY` in §28B.4 is the first mechanism analysis because it manipulates a known error location — the discarded spectral tail — under full experimental control.

The more exploratory layer/gradient-location analyses in this section are attempted only after the controlled spectral-tail result is reported.

This is secondary to the primary continuous \(e\leftrightarrow r\) relationship.

---

# 31. Thresholded 2×2 presentation

For readability only, define display thresholds after they are frozen in the analysis plan.

Initial proposed values:

```text
τ_parameter = 0.10 relative Frobenius error
ρ_privacy   = 0.80 normalized privacy recovery
```

Thus:

```text
L3✓ if e_F ≤ 0.10
L2✓ if R_priv ≥ 0.80
```

For O3 cells also show:

```text
e_floor(s*)
L3_ORIGINAL_REACHABLE(s*)
e_solver_origscale
e_solver_retained
```

If:

```text
e_floor(s*) > τ_parameter
```

annotate:

```text
OPERATOR_FLOOR_EXCEEDS_TAU
```

so operator-imposed information loss is not conflated with solver failure.

These thresholds do not define the primary inferential conclusion.

---

## 31.1 Sensitivity panel

Always show sensitivity over:

```text
τ ∈ {0.05, 0.10, 0.20}
ρ ∈ {0.50, 0.80, 0.90}
```

If the qualitative 2×2 interpretation changes substantially across this grid, state that explicitly.

---

# 32. Statistical uncertainty

## 32.1 TPR confidence intervals

Use bootstrap confidence intervals with resampling at the record level within seed.

Minimum:

```text
2,000 bootstrap replicates
```

---

## 32.2 Seed-level reporting

Never pool seeds without also showing seed-specific estimates.

For three-seed P0/P1 analyses, seed is treated as a **fixed/blocking factor**, not a random effect.

Report:

- each seed;
- mean/median across seeds;
- between-seed variation.

A random-effects variance component may be considered only in later replication with materially more independent training seeds.

Normalized \(G_L/G_R\) values from `DENOMINATOR_UNSTABLE` cells are shown descriptively but are not used for inferential cross-cell ranking.

---

## 32.3 Paired comparisons

When comparing views derived from the same trained \(A_s\), use paired bootstrap/resampling where possible.

This includes:

- \(A_s\) vs \(C_{i,s}\);
- pooled lineage vs recovered model;
- canary vs natural persistence.

---

## 32.4 Multiple comparisons

The main hypotheses are limited.

Secondary exploratory comparisons must be labeled exploratory.

If multiple operator/knowledge pairwise tests are reported, use Benjamini-Hochberg FDR correction.

---

# 33. Datasets and attack calibration split discipline

No natural record may appear simultaneously as:

- training member;
- calibration non-member;
- evaluation non-member.

## 33.1 Natural-record arm

Reference/FPR calibration is performed only on:

```text
CALIBRATION_NONMEMBERS
```

Final natural-record FPR evaluation uses:

```text
EVAL_NONMEMBERS
```

Natural members come only from the predeclared member-evaluation subset.

---

## 33.2 Canary arm

Canary thresholds are calibrated only from:

```text
I_j = 0 excluded canaries
```

using the 5-fold cross-fitting protocol in §7.4.

Natural PubMed non-members are never used to calibrate a synthetic-canary threshold.

---

## 33.3 Freeze discipline

The final evaluation labels remain unopened until:

- model training is complete;
- attack code is frozen;
- pooled aggregation is frozen;
- recovery pipeline is frozen for that experimental phase.

---

# 34. Deployment-realism scan

This is a bounded empirical side study performed during P0 week.

The purpose is to test whether **fine-tuned/derivative parents**, rather than merely common foundation checkpoints, are reused across multiple public merged descendants.

The scan can observe only public parents. It therefore measures public derivative-parent reuse as a **proxy for the plausibility of repeated reuse of a withheld constituent**; it cannot directly observe withheld-parent reuse by construction.

## 34.1 Sampling frame

Use publicly available Hugging Face metadata, model cards, and merge configurations.

Retrieve a candidate pool of up to:

```text
1,000
```

models discoverable under predeclared model-merging / mergekit-related queries/tags.

---

## 34.2 Eligibility

A sampled model is eligible if:

- it is publicly downloadable;
- it is explicitly presented as a merged model;
- at least one parent/source model can be identified from declared metadata, model card, or merge configuration.

---

## 34.3 Sample

From eligible models, take:

```text
N = 200
```

using fixed deterministic ordering or RNG seed:

```text
20260820
```

If fewer than 200 eligible models are available, use all eligible models and report the shortfall.

---

## 34.4 Parent classification

Before coding the 200-model sample, freeze:

```text
P0_13_FOUNDATION_MODEL_ALLOWLIST.json
```

Classify every declared parent as:

```text
FOUNDATION
DERIVATIVE
UNKNOWN
```

### FOUNDATION

Exact repository ID appears in the frozen allowlist of canonical pretrained/base checkpoints.

### DERIVATIVE

Public metadata/model cards provide positive evidence that the parent is derived from another model, such as:

- fine-tuned model;
- instruction-tuned derivative;
- adapter-applied model;
- merged model;
- task/domain specialist derivative.

A parent not on the foundation allowlist is **not automatically** called derivative; positive derivative evidence is required.

### UNKNOWN

Use where lineage metadata is insufficient.

Do not silently map `UNKNOWN` to `DERIVATIVE`.

---

## 34.5 Parent-centered reverse lookup

For every unique declared `DERIVATIVE` parent:

1. normalize the exact repository identifier;
2. search public Hugging Face metadata/model cards/merge configurations for other merged models declaring that exact parent;
3. use exact repository-ID matching where available;
4. deduplicate descendant repositories;
5. record the number of publicly discoverable merged descendants declaring that parent.

Foundation parents are reverse-counted only as descriptive background.

If the Hub API lacks a direct reverse-parent relation, use exact parent-ID search across model cards/configs and document discoverability limitations.

---

## 34.6 Primary deployment statistics

Among sampled merges containing at least one identifiable derivative parent, define:

\[
\boxed{
P_{\mathrm{reuse}}^{\mathrm{deriv}}
=
\frac{
\#\{\text{sampled merges with a DERIVATIVE parent having }\ge2\text{ public merged descendants}\}
}{
\#\{\text{sampled merges with at least one DERIVATIVE parent}\}
}
}
\]

Also report the overall sample proportion:

\[
P_{\mathrm{reuse,all}}^{\mathrm{deriv}}
=
\frac{
\#\{\text{sampled merges with a reused DERIVATIVE parent}\}
}{N}.
\]

Foundation-parent reuse is reported separately and **never used as evidence for the withheld-expert threat model**.

Also report:

- distribution of public descendant counts per reused derivative parent;
- maximum observed derivative-parent reuse count;
- proportion of sampled merges with no identifiable derivative parent;
- proportion of `UNKNOWN` parents;
- reverse-search coverage limitations.

This is a bounded ecosystem measurement, not an estimate of invisible withheld-parent reuse itself.

---

## 34.7 Stopping rule

Stop once:

1. the predefined eligible merge sample is coded;
2. every declared parent is classified;
3. reverse lookup is completed for every unique `DERIVATIVE` parent.

Do not expand the original merge sample because the derivative reuse rate is high or low.

A low derivative-parent reuse rate narrows the threat-model motivation; it does not trigger ad hoc resampling.

---

# 34A. Execution architecture — single parameterised scorer

The largest remaining threat to validity is implementation divergence across many legitimate analysis paths.

P0 therefore uses **one scoring engine**, not separate scripts for each arm/view/attack.

## 34A.1 Required public interface

The core scoring call must be parameterised by:

```text
arm       ∈ {CANARY, NATURAL}
score     ∈ {REFERENCE_LOSS, MIN_K}
view      ∈ {ORACLE, DESCENDANT, POOLED, RECOVERED, SYNTHETIC}
artifact  = immutable artifact identifier
seed      = training seed
fold      = outer cross-fit fold or null where not applicable
```

Conceptually:

```python
score_records(
    *,
    arm,
    score,
    view,
    artifact,
    seed,
    fold,
    record_ids,
    context
) -> ScoreTable
```

Every returned row contains at minimum:

```text
record_id
seed
fold
arm
score_family
view
artifact_id
membership_label
scalar_score
token_count
```

Additional view-specific metadata may be attached, but the core schema does not change.

## 34A.2 One implementation of cross-fitting

The five-fold canary protocol in §7.4 is implemented once.

The same cross-fitting controller performs:

1. calibration-fold construction;
2. descendant selection where applicable;
3. pooled-method selection where applicable;
4. hyperparameter selection;
5. threshold calibration;
6. held-out fold scoring;
7. realised-FPR calculation;
8. fixed-FPR ROC interpolation.

No analysis-specific script may reproduce this logic independently.

Natural-arm calibration uses the same controller with the natural calibration/evaluation partitions substituted for the canary folds.

## 34A.3 Pooled view

The pooled-lineage view is implemented as a **score transformation over already cached descendant scores**.

It does not invoke the model again.

Both:

```text
STOUFFER_MEAN
RIDGE_POOL
```

consume the same canonical `ScoreTable`.

## 34A.4 Cache keys

Model forward outputs are cached under a content-addressed key containing:

```text
model/artifact hash
record-set hash
tokenizer hash
score-code version
precision
max sequence length
```

Changing any of these invalidates the cache.

## 34A.5 Exception rule

Duplicated scoring logic is prohibited unless recorded as:

```text
SCORER_EXCEPTION
```

with:

- why the shared interface cannot represent the case;
- code path;
- tests establishing numerical equivalence where equivalence is expected;
- affected claims.

No `SCORER_EXCEPTION` may be introduced silently after viewing the affected result.

---

# 34B. Synthetic statistical-analysis dry run

Before P0-A0, run the complete analysis stack on synthetic tables where the truth is known.

This dry run is an **implementation test**, not a scientific result.

## 34B.1 Required planted scenarios

At minimum generate and test:

### DRY-A — monotone magnitude relationship

Plant:

\[
R_{\mathrm{priv}}
=
1-c e+\epsilon
\]

with no operator residual.

Expected:

```text
Spearman < 0
median operator residual ≈ 0
structural model provides little extra predictive value
```

### DRY-B — operator deviation

Plant a positive residual for one operator after conditioning on \(e\).

Expected:

```text
u_operator > 0
paired/bootstrap CI detects the planted direction
```

### DRY-C — function/privacy decoupling

Plant:

```text
e50_priv != e50_func
```

Expected:

```text
sign(estimated Δe50) = sign(planted Δe50)
```

### DRY-D — spectral-tail effect

Plant truncation points below the rank-matched privacy curve.

Expected:

```text
Δ_tail_priv_ref < 0
Δ_tail_priv_MinK < 0
tail direction corroborated
```

Also plant a geometry-confounded scenario where only the reference score moves.

Expected:

```text
BASE_GEOMETRY_SENSITIVE
```

### DRY-E — denominator instability

Plant:

\[
M(A)-M_0
\]

near zero.

Expected:

```text
DENOMINATOR_UNSTABLE
normalized cross-cell ranking disabled
```

### DRY-F — beyond-calibration point

Plant an operator observation with:

\[
e>2.0.
\]

Mechanical assertion:

```text
result_label == UNDEFINED_BEYOND_CALIBRATION
```

and no extrapolated residual is created.

---

### DRY-G — complete null / false-positive control

DRY-G checks that the full analysis stack can report a null without manufacturing a result.

The tolerance values are **not chosen by hand**.

They are calibrated once from:

```text
N_NULL_CALIBRATION_TRIALS = 200
calibration master seeds  = 810000 … 810199
```

using the same:

```text
29 registry-like cells
× 3 training seeds
```

clustered design used by the validation trials.

The frozen calibration output is:

```text
P0_PRE_00_DRYRUN_TOLERANCE_CALIBRATION.json
```

No real model result is used in this calibration.

#### Planted null

For every calibration/validation trial plant:

- \(R_{\mathrm{priv}}\) independent of \(e\);
- no operator effect;
- no lineage effect;
- \(D_L=0\) in expectation;
- \(D_R=0\) in expectation;
- cell-level shared noise plus within-seed residual noise.

The cell-level component deliberately makes the effective independent unit closer to the registry cell than to 87 independent rows.

#### Frozen validation trials

After calibration, run exactly:

```text
DRY_G_VALIDATION_SEEDS = {810200, 810201, 810202}
```

Each validation seed creates one independent synthetic null trial.

A single failed trial does **not** immediately fail P0-PRE.

Use:

```text
if 3/3 pass:
    DRY_G = PASS

if 2/3 pass:
    DRY_G = PASS_WITH_INVESTIGATION
    write DRY_G_INVESTIGATION.md

if <=1/3 pass:
    DRY_G = FAIL
```

`PASS_WITH_INVESTIGATION` still requires confirming that the failed statistic lies inside the empirical calibration distribution and that no implementation bug is identified.

#### Null tolerances

From the frozen 200-trial calibration:

\[
|\rho_{\mathrm{Spearman}}|
\le
0.355249.
\]

For operator residuals:

\[
\max_o|\operatorname{median}(u_o)|
\le
0.015369.
\]

For raw audit gaps:

\[
|\overline{D_L}|
\le
0.003274,
\]

and:

\[
|\overline{D_R}|
\le
0.003068.
\]

For grouped structural-model improvement:

\[
\Delta_{\mathrm{CV}}
=
\mathrm{MAE}_{\mathrm{magnitude}}
-
\mathrm{MAE}_{\mathrm{structural}}
\le
0.003423.
\]

These are the empirical 97.5th-percentile null tolerances generated before real data.

#### Null confidence intervals

For DRY-G only, uncertainty resampling is by **registry cell cluster**:

1. sample registry cells with replacement;
2. retain all three seed observations belonging to every sampled cell;
3. recompute the statistic.

This preserves the actual dependence structure.

For every validation trial additionally require:

```text
cell-cluster 95% CI for Spearman contains 0
cell-cluster 95% CI for each operator mean-u contains 0
cell-cluster paired 95% CI for D_L contains 0
cell-cluster paired 95% CI for D_R contains 0
```

The fixed empirical point tolerances and the CI criteria must both be satisfied for that trial to count as PASS.

#### Structural model null

Using the same leave-one-training-seed-out grouped CV intended for the real analysis, require:

```text
Δ_CV <= 0.003423
```

under the planted null.

The analysis stack must therefore demonstrate both:

```text
TRUE EFFECT -> DETECTABLE
NULL EFFECT -> NOT MANUFACTURED
```

before real outcomes are opened.

---

### DRY-H — cross-fitting controller + deliberate leakage control

This scenario operates on **per-record synthetic scores**, not downstream \((e,r)\) tables.

Use approximately the real canary scale:

```text
4,096 candidate records
~50% members / ~50% non-members
5 outer folds
```

Generate at least two candidate descendant/attack-score channels:

```text
STRONG
WEAK
```

with known Gaussian member/non-member separation.

The canonical controller must perform, inside every outer fold:

1. fold construction;
2. calibration-only candidate selection;
3. calibration-only threshold estimation;
4. held-out fold scoring;
5. realised FPR calculation;
6. ROC-interpolated TPR at exactly 1% FPR.

#### H1 — clean IID controller test

For the planted strong channel, compute the analytic Gaussian TPR implied by the planted separation and the 99th-percentile non-member threshold.

Require:

```text
0.004 <= aggregate realised FPR <= 0.016
|observed fixed-FPR TPR - analytic TPR| <= 0.08
STRONG selected in >= 4 of 5 outer folds
```

#### H2 — deliberate held-out calibration leak

Construct a negative-control dataset in which the held-out non-member distribution is deliberately shifted relative to calibration data.

Run:

- the correct calibration-only threshold; and
- a deliberately leaky threshold estimated from the held-out non-members.

Require:

```text
TPR_leaky - TPR_proper >= 0.05
```

for the planted leakage scenario.

This test exists solely to prove that the test suite detects a threshold leakage path when one is deliberately introduced.

#### H3 — structural overlap guard

Deliberately pass a calibration/evaluation split with at least one overlapping `record_id`.

Require:

```text
LeakageError
```

before metric computation.

---

### DRY-CACHE — content-addressed cache correctness

Run two cache assertions.

#### Stable hit

With identical:

```text
artifact hash
record-set hash
tokenizer hash
score-code version
precision
max sequence length
```

require:

```text
cache_key_1 == cache_key_2
cache_hit == TRUE
cached scalar scores are bitwise-identical
```

#### Version invalidation

Change only:

```text
score-code version
```

Require:

```text
new_cache_key != old_cache_key
old cached entry is not returned under new key
```

A cache implementation that fails either assertion cannot be used for P0.

---

## 34B.1A Frozen assertion tolerances

All dry-run assertions are mechanical.

### Calibration provenance

The fixed tolerance bank is generated from:

```text
P0_PRE_00_DRYRUN_TOLERANCE_CALIBRATION.json
```

with:

```text
null calibration seeds     = 810000 … 810199
positive calibration seeds = 810000 … 810199
N per calibration family   = 200
```

The generator code and calibration artifact are frozen before real results.

### Frozen thresholds

| Scenario | Required assertion |
|---|---|
| DRY-A | Spearman \(\rho<-0.90\) |
| DRY-B | mean planted operator residual \(>0.175143\) |
| DRY-C | estimated \(\Delta e_{50}>0.15\) with planted positive sign |
| DRY-D1 | reference \(\Delta_{\mathrm{tail,priv}}<-0.136213\) **and** Min-K% \(\Delta_{\mathrm{tail,priv}}<-0.135921\) |
| DRY-D2 | reference \(\Delta_{\mathrm{tail,priv}}<-0.08\) and \(|\mathrm{MinK}\ \Delta_{\mathrm{tail,priv}}|<0.04\), yielding `BASE_GEOMETRY_SENSITIVE` |
| DRY-E | denominator relative SE \(>0.30\), yielding `DENOMINATOR_UNSTABLE` |
| DRY-F | result exactly `UNDEFINED_BEYOND_CALIBRATION` |
| DRY-G | majority of 3 fixed validation trials passes calibrated null thresholds + cell-cluster CI conditions |
| DRY-H | all controller/leakage assertions in §34B.1 |
| DRY-CACHE | stable-hit and version-invalidation assertions both pass |

### Positive-control power calibration

For DRY-B, the pass threshold:

\[
0.175143
\]

is the 2.5th percentile of 200 simulations with the planted positive residual.

Thus approximately 97.5% of correctly implemented planted-effect trials should exceed it.

For DRY-D1, the thresholds:

\[
-0.136213
\]

for the reference score and:

\[
-0.135921
\]

for Min-K% are the 97.5th percentiles of their respective 200 planted negative-effect simulations.

Thus approximately 97.5% of correctly implemented planted tail-effect trials should remain more negative than the frozen cutoff.

Changing these tolerances after real P0 results are observed is prohibited.

---

## 34B.2 Dry-run acceptance

The statistical stack is considered frozen only if all planted scenarios pass their assertions.

Required outputs:

```text
P0_PRE_00_DRYRUN_TOLERANCE_CALIBRATION.json
P0_PRE_01_SYNTHETIC_ANALYSIS_INPUT.csv
P0_PRE_02_SYNTHETIC_ANALYSIS_RESULTS.json
P0_PRE_03_SYNTHETIC_ANALYSIS_ASSERTIONS.txt
P0_PRE_04_CROSSFIT_CONTROLLER_RESULTS.json
P0_PRE_05_CACHE_ASSERTIONS.txt
DRY_G_INVESTIGATION.md                 # only when DRY-G = PASS_WITH_INVESTIGATION
```

Any failed assertion must be fixed before real P0 evaluation data are opened.

## 34B.3 Freeze semantics

The §42 checkbox:

```text
[ ] Statistical script frozen
```

means:

```text
code version tagged
AND
synthetic dry run passed
AND
assertion report saved
```

It does not mean merely that a script file exists.

---

# 34C. Calendar gate and minimum viable result set

The experiment is now constrained by calendar time as strongly as by compute.

The dates below are **internal research deadlines**, not claimed University submission dates.

They are anchored to:

```text
current project date = 20 August 2026
```

and exist to protect a complete dissertation result from being displaced by optional expansion.

## 34C.1 Dated execution schedule

The dates below are **internal research deadlines**, not claimed University deadlines.

The critical long pole is software correctness, not GPU inference.

### 20 August 2026

```text
v1.9 FINAL CLOSED specification freeze
tolerance calibration frozen
```

No further speculative methodology review.

### 21–27 August 2026 — NON-COMPRESSIBLE P0-PRE SOFTWARE WINDOW

Complete:

```text
real HF/PEFT scorer backend integration
single parameterised scorer
content-addressed cache
canonical cross-fitting controller
pooled score transformations
analysis engine
DRY-A … DRY-H
DRY-CACHE
all frozen assertions
```

The 1,000-sequence H100 benchmark is executed immediately after the production scoring path is verified.

The software gate is complete only when:

```text
SCORER_ENGINE = VERIFIED
ANALYSIS_DRY_RUN = PASS
CROSSFIT_NEGATIVE_CONTROL = PASS
CACHE_ASSERTIONS = PASS
P0_00_COMPUTE_BUDGET.json = WRITTEN
```

### SOFTWARE_SLIP rule

P0-PRE is **not compressible**.

Under no circumstance may the project:

- skip a dry-run scenario;
- weaken a tolerance;
- remove a negative control;
- bypass cross-fit validation;
- disable cache invalidation;
- open real evaluation outcomes

to preserve a calendar date.

If the complete P0-PRE gate is not true by:

```text
2026-08-27T23:59:59+01:00
```

automatically set:

```text
SOFTWARE_SLIP = TRUE
CALENDAR_COMPRESSED_MVRS = ACTIVE
```

All downstream dates slide as required.

The hard P1 launch drop-dead **does not slide**.

### 28 August 2026 — baseline if P0-PRE finishes within window

Run:

```text
P0-A0 split / deduplication leakage check
reference-output cache completion
```

### 29–31 August 2026

Run:

```text
P0-A oracle privacy-power experiments
3 non-DP seeds
matched no-canary control
```

Decide which privacy endpoints are eligible.

### 1 September 2026

Run:

```text
P0-B canary/natural construct-validity check
P0-C1 exact linear oracle plumbing
```

`P0-C1` must pass before ladder execution.

### 2–5 September 2026

Priority execution:

```text
primary rank-matched Stage-1 fidelity ladder
privacy + function scoring
reference + Min-K%
```

If the project reaches 3 September without a complete MVRS:

```text
P1 = NOT_STARTED_CALENDAR
```

This is an acceptable pre-registered outcome.

### 6–7 September 2026

Run:

```text
rank-matched Stage-2 crossing refinement
C01/C02/C03 four-view audit
k ∈ {1,2,4}
```

### 8 September 2026

`MINIMUM_VIABLE_RESULT_SET` review and first complete results package.

### 3 September 2026, 18:00 BST — HARD P1 DROP-DEAD

This deadline remains:

```text
P1_LAUNCH_DEADLINE = 2026-09-03T18:00:00+01:00
```

P1 may launch only if the MVRS and all required launch gates happen to complete **ahead of the baseline schedule**.

Otherwise:

```text
P1 = NOT_STARTED_CALENDAR
```

No software-validity gate is weakened to make P1 fit.

### 9 September 2026 — empirical-results freeze target

If MVRS completion requires the full baseline window, finish authorized MVRS reruns/figures and freeze the empirical result set.

No new experiment family begins after this point.

### 10 September 2026 onward

Priority:

```text
analysis consolidation
threats-to-validity
dissertation writing
figure/table verification
reproducibility package
supervisor feedback
```

---

## 34C.1A Reconciliation with §35 — expansion gates

The §35 execution sequence contains:

```text
P0-C2
P0-D / P0-D2 / conditional O3
P0-E
```

These are scientifically valid but **not part of the MVRS critical path**.

Their calendar status is:

```text
P0-C2 = POST_MVRS_EXPANSION
P0-D  = POST_MVRS_EXPANSION
P0-E  = POST_MVRS_EXPANSION
```

Rules:

1. Do not run them before the MVRS if they compete for engineering, GPU, or analysis attention.
2. If MVRS finishes early enough to permit P1 before the hard drop-dead:
   - run the relevant P0-C2/P0-D prerequisite before launching the corresponding incomplete-lineage/operator P1 cells;
   - P0-E may run in parallel if it does not threaten P1/MVRS evidence.
3. If the hard P1 drop-dead is missed:
   - P0-C2/P0-D/P0-E are optional paper-extension work only;
   - none is required for the dissertation's minimum empirical story.
4. `CALENDAR_COMPRESSED_MVRS` automatically defers all three unless needed to repair an MVRS validity problem.

Thus an undated expansion gate can never silently block or displace the minimum viable result.

---

## 34C.2 MINIMUM_VIABLE_RESULT_SET

The minimum viable result is pre-registered now so calendar pressure cannot redefine success later.

It is the cheapest complete empirical story supported by the design.

### MVRS-1 — measurement validity

Require:

```text
P0-A0 complete
P0-A complete
at least one privacy arm PRIVACY_ENDPOINT_ELIGIBLE
P0-C1 exact recovery pass
```

If only the canary arm is eligible, claims are explicitly canary/audit-instrument claims.

Natural-record claims require the natural arm to be eligible and its blind/dedup checks to pass.

### MVRS-2 — controlled error-decay result

Complete the **primary rank-matched ladder** for the eligible privacy arm(s).

Required outputs:

\[
e
\rightarrow
R_{\mathrm{priv}}
\]

and, where function endpoint is eligible:

\[
e
\rightarrow
R_{\mathrm{func}}.
\]

Run Stage-2 refinement for both rank-matched crossings if affordable.

Primary result:

\[
\Delta e_{50}
=
e^{\mathrm{priv}}_{50}
-
e^{\mathrm{func}}_{50}
\]

when both crossings are estimable.

If only one refinement is affordable:

```text
Δe50 = COMPUTE_LIMITED_NOT_ESTIMATED
```

and the coarse curves remain reportable.

### MVRS-3 — lineage/audit comparison at k = 1, 2, 4

Run only the high-information linear controls:

```text
C01
C02
C03
```

with the four audit views:

```text
A
best single descendant
pooled lineage
recovered constituent
```

and both score families.

Required primary statistics:

\[
D_L=M_{\mathrm{pool}}-M_{\mathrm{desc}}
\]

and:

\[
D_R=M_{\mathrm{rec}}-M_{\mathrm{pool}}.
\]

Report across:

\[
k\in\{1,2,4\}.
\]

This is sufficient to answer whether joint-lineage auditing improves on artifact-by-artifact auditing under a controlled recoverable release family.

### MVRS-4 — minimum uncertainty/reporting package

Require:

```text
3 training seeds
paired/raw gap CIs
seed-specific results
realised FPR beside TPR
reference + Min-K% readouts
distance-to-base diagnostics
all dry-run and cache assertions PASS
```

### MVRS scientific story

If only the MVRS is completed, the dissertation may claim, subject to actual results:

> We establish a controlled measurement curve linking constituent reconstruction error to recoverable membership signal, test whether privacy and task function decay on the same parameter-error scale, and measure whether jointly observing a public merge lineage changes audit power relative to evaluating one descendant at a time.

This is a complete dissertation story.

It does **not** require:

- DARE;
- O3;
- TIES;
- incomplete-lineage recovery;
- a DP epsilon sweep;
- a second architecture;
- deployment-prevalence evidence.

---

## 34C.3 Calendar compression rule

Immediately enter:

```text
CALENDAR_COMPRESSED_MVRS
```

if:

- `SOFTWARE_SLIP = TRUE`; or
- the measured benchmark; or
- implementation progress

shows that the dated schedule cannot protect the MVRS while also supporting expansion.

Under `CALENDAR_COMPRESSED_MVRS`, priority is:

1. scorer/cross-fit correctness;
2. P0-A measurement power;
3. P0-C1 oracle;
4. primary rank-matched ladder;
5. rank-matched crossing refinement;
6. C01/C02/C03 four-view audit;
7. minimum figures/tables;
8. dissertation writing.

Automatically defer, in this order:

```text
O4 quantization
second architecture
TIES
DP epsilon sweep
DARE/O3 operator expansion
incomplete-lineage expansion
deployment-realism scan
isotropic ladder beyond already-computed points
```

No deferred component is needed to declare the MVRS complete.

---

# 35. P0 execution sequence

Run P0 in this exact order.

## P0-PRE — Scoring-engine and analysis dry run

Before any empirical P0 result is opened:

1. implement §34A's parameterised scoring engine;
2. run §34B's synthetic analysis dry run;
3. require all dry-run assertions to pass.

Required status:

```text
SCORER_ENGINE = VERIFIED
ANALYSIS_DRY_RUN = PASS
CROSSFIT_NEGATIVE_CONTROL = PASS
CACHE_ASSERTIONS = PASS
```

Only then continue.

---

## P0-0 — Compute benchmark and reference caching

Run §0.2 and freeze:

```text
P0_00_COMPUTE_BUDGET.json
```

Compute/cache all fixed \(	heta_0\) reference outputs once.

---

## P0-A0 — Blind split / deduplication leakage check

Run §6.4 before interpreting the natural-record arm.

An initial statistical flag triggers `BLIND_SPLIT_INVESTIGATION`, not an immediate stop.

Use `STOP_NATURAL_PRIVACY_ARM` only after the confirmed §6.4 failure rule.

---

## P0-A — Oracle privacy power + canary-load control

Train three canary-containing calibration seeds and the matched no-canary control from §7.5.

Measure oracle \(A\) separately for:

- unique canaries;
- natural records.

Apply §18 eligibility separately by arm.

### Gate

If both are ineligible:

```text
STOP_PRIVACY_GRID
```

Redesign measurement.

---

## P0-B — Canary construct validity

Compare canary and natural persistence under a minimal linear merge and recovery.

Apply §27 consequence branch.

---

## P0-C1 — Linear oracle plumbing

Known partner + known coefficient.

Require:

\[
e_F\le10^{-5}.
\]

Failure:

```text
STOP_RECOVERY_PIPELINE
```

Debug before continuing.

---

## P0-F — Synthetic fidelity ladders

Immediately after P0-C1 passes, run both ladders in §28A:

1. primary LoRA-rank-matched ladder;
2. secondary full-rank isotropic ladder.

This creates controlled error-magnitude and perturbation-structure coverage before the operator grid.

---

## P0-C2 — Occupied common-source baseline

**Calendar class:** `POST_MVRS_EXPANSION` unless required for an already-authorized early P1 launch.

Unknown partners + fixed coefficient.

Reproduce/adapt Spectral-DeTuning-style common-source recovery.

Use only:

```text
k ∈ {2,4}
```

for the unknown-partner baseline.

Record:

- parameter error;
- functional recovery;
- privacy recovery where eligible;
- conditioning;
- comparison to naive shared mean/base subtraction.

If the published-style baseline cannot outperform the naive baseline after implementation verification, stop the novel recovery arm and debug.

---

## P0-D — DARE utility gate

**Calendar class:** `POST_MVRS_EXPANSION` unless required for an already-authorized early P1 launch.

Run §24.

If DARE passes:

```text
P1_PRIMARY_LOSSY_OPERATOR = DARE
```

If DARE fails, execute P0-D2.

---

## P0-D2 — O3 SVD-TRUNC-MERGE fallback gate

Run §24A only if DARE fails, unless O3 was separately budgeted as a pre-result secondary control.

If O3 passes:

```text
P1_PRIMARY_LOSSY_OPERATOR = SVD_TRUNC_MERGE
```

then, before P1 authorization, run:

```text
P0-D3  Conditional O3 spectral-truncation ladder (§28B)
P0-C4  O3 incomplete-lineage recovery validation (§25)
```

If O3 also fails:

```text
OPERATOR_AXIS_INSUFFICIENT
```

and apply §24A.7.

---

## P0-E — Derivative-parent deployment scan

**Calendar class:** `POST_MVRS_EXPANSION`.

Run §34.

Record the derivative-parent reuse statistic and narrow threat-model wording if needed.

---

# 36. P1 minimal experimental grid and cell registry

P1 begins only if P0-A, P0-C1, P0-F, and relevant operator/method gates pass.

Primary operator selection:

```text
if DARE passes §24:
    primary lossy operator = DARE
elif O3 passes §24A:
    primary lossy operator = SVD_TRUNC_MERGE
else:
    OPERATOR_AXIS_INSUFFICIENT
```

The selected lossy operator is frozen before operator privacy outcomes are opened.

Other factors:

```text
k ∈ {1,2,4}
privacy ∈ {CALIBRATION, REPRESENTATIVE_DP}
lineage ∈ {
  Z-HIGH,
  Z-INCOMPLETE-FIXED,
  Z-INCOMPLETE-VARYING-KNOWN
}
```

Use only:

```text
RUN_CONTROL
RUN_REPRODUCTION
RUN_MEASUREMENT
CONDITIONAL
METHOD_GATED
OPERATOR_GATED
STRUCTURAL_NA
DP_SMOKE_ONLY
```

## 36.1 Calibration-regime registry

| ID | Operator | k | Lineage | Schedule | Status | Interpretation |
|---|---|---:|---|---|---|---|
| C01 | LINEAR | 1 | Z-HIGH | fixed | RUN_CONTROL | closed-form oracle |
| C02 | LINEAR | 2 | Z-HIGH | fixed | RUN_CONTROL | algebraic control |
| C03 | LINEAR | 4 | Z-HIGH | fixed | RUN_CONTROL | algebraic control |
| C04 | LINEAR | 1 | Z-INCOMPLETE-FIXED | fixed | STRUCTURAL_NA | unknown partner, one mixture |
| C05 | LINEAR | 2 | Z-INCOMPLETE-FIXED | fixed | RUN_REPRODUCTION | Spectral/common-source |
| C06 | LINEAR | 4 | Z-INCOMPLETE-FIXED | fixed | RUN_REPRODUCTION | Spectral/common-source |
| C07 | LINEAR | 1 | Z-INCOMPLETE-VARYING-KNOWN | — | STRUCTURAL_NA | unknown partner, one mixture |
| C08 | LINEAR | 2 | Z-INCOMPLETE-VARYING-KNOWN | k2-reference | RUN_REPRODUCTION | rescaled shared-source stress control |
| C09M | LINEAR | 4 | Z-INCOMPLETE-VARYING-KNOWN | spread-matched | RUN_REPRODUCTION | k contrast at matched beta spread |
| C09W | LINEAR | 4 | Z-INCOMPLETE-VARYING-KNOWN | wide-spread | RUN_REPRODUCTION | heteroscedastic stress condition |
| C10 | LOSSY | 1 | Z-HIGH | fixed | OPERATOR_GATED | DARE or O3 selected pre-result |
| C11 | LOSSY | 2 | Z-HIGH | fixed | OPERATOR_GATED | DARE or O3 selected pre-result |
| C12 | LOSSY | 4 | Z-HIGH | fixed | OPERATOR_GATED | DARE or O3 selected pre-result |
| C13 | LOSSY | 1 | Z-INCOMPLETE-FIXED | fixed | STRUCTURAL_NA | unknown partner, one mixture |
| C14 | LOSSY | 2 | Z-INCOMPLETE-FIXED | fixed | CONDITIONAL | O3: RUN_REPRODUCTION; DARE: METHOD_GATED |
| C15 | LOSSY | 4 | Z-INCOMPLETE-FIXED | fixed | CONDITIONAL | O3: RUN_REPRODUCTION; DARE: METHOD_GATED |
| C16 | LOSSY | 1 | Z-INCOMPLETE-VARYING-KNOWN | — | STRUCTURAL_NA | unknown partner, one mixture |
| C17 | LOSSY | 2 | Z-INCOMPLETE-VARYING-KNOWN | k2-reference | CONDITIONAL | O3: RUN_REPRODUCTION; DARE: METHOD_GATED |
| C18 | LOSSY | 4 | Z-INCOMPLETE-VARYING-KNOWN | spread-matched | CONDITIONAL | O3: RUN_REPRODUCTION; DARE: METHOD_GATED |

Calibration registry:

```text
19 explicit rows
```

including four structural N/A rows.

C09W is a linear stress condition and is not automatically duplicated under the lossy operator.

## 36.2 Representative-DP registry

One-seed DP execution is smoke-only. Inferential status requires the 3-seed DP set.

| ID | Operator | k | Lineage | Schedule | Before 3 DP seeds | After 3 DP seeds |
|---|---|---:|---|---|---|---|
| D01 | LINEAR | 1 | Z-HIGH | fixed | DP_SMOKE_ONLY | RUN_CONTROL |
| D02 | LINEAR | 2 | Z-HIGH | fixed | DP_SMOKE_ONLY | RUN_CONTROL |
| D03 | LINEAR | 4 | Z-HIGH | fixed | DP_SMOKE_ONLY | RUN_CONTROL |
| D04 | LINEAR | 2 | Z-INCOMPLETE-FIXED | fixed | DP_SMOKE_ONLY | RUN_REPRODUCTION |
| D05 | LINEAR | 4 | Z-INCOMPLETE-FIXED | fixed | DP_SMOKE_ONLY | RUN_REPRODUCTION |
| D06 | LINEAR | 2 | Z-INCOMPLETE-VARYING-KNOWN | k2-reference | DP_SMOKE_ONLY | RUN_REPRODUCTION |
| D07 | LINEAR | 4 | Z-INCOMPLETE-VARYING-KNOWN | spread-matched | DP_SMOKE_ONLY | RUN_REPRODUCTION |
| D08 | LOSSY | 1 | Z-HIGH | fixed | DP_SMOKE_ONLY | OPERATOR_GATED |
| D09 | LOSSY | 2 | Z-HIGH | fixed | DP_SMOKE_ONLY | OPERATOR_GATED |
| D10 | LOSSY | 4 | Z-HIGH | fixed | DP_SMOKE_ONLY | OPERATOR_GATED |

Incomplete-lineage lossy DP cells follow the calibration-regime recovery status.

If O3 is selected and P0-C4 passes, corresponding incomplete-lineage DP cells may be added only through a pre-result registry revision using the same truncated-source recovery model.

If DARE is selected, they remain unauthorized until a DARE-compatible recovery method is validated.

They are not silently added to the current 10-row DP registry.

Total explicit P1 registry:

```text
19 calibration rows + 10 DP rows = 29 rows
```

No unregistered cell may be added mid-run without a pre-result revision or `SPEC_DEVIATION`.

### What is scientifically open in P1

Primary open measurements are:

- how a utility-valid information-removal merge transformation changes the \(e\leftrightarrow R_{\mathrm{priv}}\) relationship;
- whether operator-induced points depart from the appropriate calibration:
  - rank-matched additive ladder for DARE;
  - deterministic truncation ladder for O3;
- whether joint lineage changes practical privacy-audit power;
- whether reconstruction routing changes attack effectiveness;
- whether privacy and function decay at different error scales;
- if O3 is selected, whether spectral-tail removal behaves differently from generic rank-matched error.

Known-varying-\(\alpha\) linear cells remain controls/stress tests.

### Operator × lineage interaction scope

The phrase “across merge transformations and lineage knowledge” does **not** guarantee a fully crossed factorial interaction.

If DARE is selected and C14/C15/C17/C18 remain method-gated:

- operator effects are measured at `Z-HIGH`;
- lineage effects are measured primarily under `LINEAR`;
- no operator × lineage interaction claim is made.

If O3 is selected and P0-C4 passes:

- C14/C15/C17/C18 become runnable reproductions;
- operator and lineage axes are crossed for those calibration cells;
- an operator × lineage interaction may be analyzed under that validated O3 recovery model.

If `OPERATOR_AXIS_INSUFFICIENT` is triggered, operator-specific claims are removed rather than rescued after the fact.

## 36.3 Known varying-alpha schedules

Use the three schedules in §25:

```text
k=2 reference:
    α = {0.35, 0.65}

k=4 spread-matched:
    α = {0.35, 0.45, 0.55, 0.65}

k=4 wide-spread:
    α = {0.25, 0.40, 0.60, 0.75}
```

The coefficients are known to the attacker and handled through deterministic rescaling.

Hidden coefficients are not part of P1.

---

# 37. DP-arm deliverable

The DP arm is not required to populate the normalized privacy-recovery scatter.

If:

\[
M(A)\approx M_0
\]

and the condition is privacy-endpoint ineligible, its deliverable is:

> **a confirmatory measurement-floor result under a representative strong-DP setting where constituent recovery may still be evaluated.**

P0 does not claim to locate the DP transition with one epsilon value.

The one-seed `DP_SMOKE` result is plumbing/accounting only. The following reporting requirements apply to the 3-seed inferential DP set if executed.

Report:

- parameter recovery;
- functional recovery;
- oracle attack metric;
- confidence interval;
- ineligibility status.

Do not force an \(R_{\mathrm{priv}}\) value.

---

# 38. Primary hypotheses

The hypotheses are deliberately null-safe.

## H1 — Parameter-proxy validity

> Privacy recovery is monotonically associated with parameter recovery quality.

This is evaluated continuously.

---

## H2 — Parameter-error sufficiency

> Real operator-induced privacy recovery is explained by the synthetic magnitude-calibration curve; operator, lineage, conditioning, and error location provide no additional explanatory value at matched total parameter error.

Evidence against H2 is expressed as systematic non-zero operator residuals relative to the synthetic ladder and motivates the location analysis.

---

## H3 — Joint-lineage value

Primary within-cell null:

\[
D_L=0.
\]

Alternative:

\[
D_L\neq0.
\]

The normalized \(G_L\) is a secondary cross-cell effect-size summary.

No positive direction is presupposed.

---

## H4 — Reconstruction-routing value

Primary within-cell null:

\[
D_R=0.
\]

Alternative:

\[
D_R\neq0.
\]

The normalized \(G_R\) is a secondary cross-cell effect-size summary.

Interpretation remains attack-representation, not information creation.

---

# 39. Falsification / null outcomes

The study remains scientifically useful under the following outcomes.

## Outcome A

\[
R_{\mathrm{priv}}
\]

is a tight monotone function of parameter error.

Interpretation:

> conventional parameter-recovery quality is a useful proxy for privacy recovery in this setting.

---

## Outcome B

At comparable parameter error, privacy recovery varies systematically by operator.

Interpretation:

> parameter-error magnitude alone is insufficient.

---

## Outcome C

Parameter recovery succeeds while privacy recovery does not.

Interpretation:

> model confidentiality and training-record privacy separate.

---

## Outcome D

Parameter recovery fails while privacy recovery remains high.

Interpretation:

> privacy-relevant information survives beyond conventional model recoverability.

---

## Outcome E

\[
D_L\approx0.
\]

(and correspondingly \(G_L\) near zero in eligible cells).

Interpretation:

> joint lineage provides little practical audit value beyond the strongest descendant.

This is a valid hostile/null result.

---

## Outcome F

\[
D_R\approx0.
\]

(and correspondingly \(G_R\) near zero in eligible cells).

Interpretation:

> reconstruction is unnecessary for the tested privacy attack family.

---

# 40. Expansion rules

## Add TIES only if

- P0/P1 pipeline is stable;
- DARE/linear comparisons are interpretable;
- privacy measurement is powered in at least one calibration regime;
- there is remaining scientific uncertainty about signal-dependent information removal.

---

## Add \(k=8\) only if

- \(k=4\) has not saturated recovery/audit behavior.

---

## Add epsilon sweep only if

- at least one DP condition is privacy-endpoint eligible or
- the specific objective is to map the measurement-floor transition.

Candidate later grid:

```text
ε ∈ {1, 4, 8}
```

but only after the measurement instrument is validated.

---

## Add second architecture only if

at least one of the following appears in the primary model:

- meaningful parameter/privacy decoupling;
- meaningful \(G_L\);
- meaningful \(G_R\);
- a clear operator-dependent effect.

The replication architecture is not used to rescue a null primary result.

---

# 41. Stop conditions

## Stop privacy expansion if

- calibration oracle remains privacy-ineligible after one predeclared measurement redesign.

---

## Stop recovery expansion if

- exact linear oracle fails after implementation debugging; or
- the occupied common-source baseline cannot be reproduced under a verified implementation.

---

## Remove an operator if

- the operator destroys utility beyond its predeclared realism gate;
- implementation ambiguity makes information-loss versus code failure impossible to separate.

For O3, a large absolute \(e_F\) does not by itself imply solver failure.

First compare:

```text
e_floor(s*)
e_solver_origscale
e_solver_retained
```

and separate operator-imposed information loss from algorithmic recovery error.

---

## Reopen the RQ only if

a verified prior work directly covers:

- ordinary public merge release family;
- passive observer;
- hidden/source-absent constituent;
- joint-lineage observation;
- continuous parameter-recovery versus privacy-recovery analysis;
- descendant-only versus lineage-aware audit comparison.

Adjacent work alone does not reopen the RQ.

---

# 42. Pre-evaluation freeze checklist

Before opening final evaluation labels, confirm:

```text
[ ] Parameterised scorer implemented and tested
[ ] 200-trial dry-run tolerance calibration artifact frozen
[ ] DRY-G majority-of-three null validation passed
[ ] Any single DRY-G failure investigated before acceptance
[ ] DRY-H cross-fitting / deliberate-leak assertions passed
[ ] Cache stable-hit / version-invalidation assertions passed
[ ] Synthetic analysis dry run passed all planted assertions
[ ] Compute benchmark / P0 GPU-hour projection recorded
[ ] P1 upper-bound GPU-hour projection recorded
[ ] SOFTWARE_SLIP state recorded
[ ] Internal P1 drop-dead recorded as 2026-09-03 18:00 BST
[ ] P0-C2/P0-D/P0-E calendar class recorded as POST_MVRS_EXPANSION
[ ] MINIMUM_VIABLE_RESULT_SET completion status recorded
[ ] Base/reference outputs cached
[ ] Dataset splits hashed
[ ] Blind split/dedup check passed, investigated, or natural arm formally stopped
[ ] Canary pool hashed
[ ] Canary fold assignment hashed
[ ] Matched no-canary contamination-control config frozen
[ ] Training configs hashed
[ ] Seed list fixed
[ ] Merge configs fixed
[ ] Recovery code version tagged
[ ] Attack code version tagged
[ ] Reference model fixed
[ ] Min-K setting fixed
[ ] Pooled aggregator fixed
[ ] Canary-vs-natural threshold calibration rule fixed
[ ] Eligibility rule fixed
[ ] DARE gate evaluated / selected rate fixed
[ ] O3 fallback implementation/gate frozen before DARE result inspection
[ ] O3 analytic floor/effective-rank implementation validated
[ ] Conditional O3 truncation-ladder rank schedule frozen
[ ] P1 primary lossy operator frozen before operator privacy outcomes
[ ] Varying-alpha schedules and partner-norm guard frozen
[ ] Rank-matched + isotropic fidelity-ladder generators/version fixed
[ ] Stage-2 ladder-refinement rule frozen
[ ] Zero-delta/base anchor checked
[ ] Foundation-model allowlist frozen before deployment coding
[ ] Statistical script version tagged
[ ] Synthetic dry-run assertion report saved
[ ] Statistical script frozen only after dry-run PASS
[ ] Display thresholds frozen
[ ] Any SPEC_DEVIATION entries reviewed
```

---

# 43. Required artifacts from P0

P0 must produce:

```text
P0_00_BASE_MODEL_DATA_CUTOFF.md
P0_01_DATA_SPLIT_MANIFEST.json
P0_02_CANARY_MANIFEST.json
P0_03_TRAINING_CONFIGS/
P0_04_BLIND_TEXT_BASELINE.csv
P0_05_ORACLE_PRIVACY_POWER.csv
P0_06_CANARY_LOAD_CONTAMINATION.csv
P0_07_CANARY_NATURAL_VALIDITY.csv
P0_08_LINEAR_ORACLE_RECOVERY.csv
P0_09_SYNTHETIC_FIDELITY_LADDERS.csv
P0_10_SPECTRAL_BASELINE_RECOVERY.csv
P0_11_CONDITIONING.csv
P0_12_DARE_UTILITY_GATE.csv
P0_12B_O3_SVD_TRUNC_GATE.csv
P0_12C_VARYING_ALPHA_STRESS.csv
P0_12D_O3_INFORMATION_FLOOR_AND_RANK.csv
P0_12E_O3_TRUNCATION_LADDER.csv
P0_12F_O3_INCOMPLETE_LINEAGE_RECOVERY.csv
P0_13_FOUNDATION_MODEL_ALLOWLIST.json
P0_14_DEPLOYMENT_REALISM.csv
P0_15_GATE_DECISIONS.md
```

---

# 44. Required figures from P1

If P1 executes, plan for these figures.

## Figure 1 — Study schematic

\[
A \rightarrow C_{1:k} \rightarrow \{C_i,\ R,\ \hat A\}
\]

with the four audit views.

---

## Figure 2 — Parameter error vs privacy recovery

Scatter:

\[
e_F
\quad\text{vs}\quad
R_{\mathrm{priv}}
\]

with:

- rank-matched synthetic curve/band as the primary magnitude baseline;
- isotropic synthetic curve/band as the perturbation-structure control;
- operator-induced points overlaid at measured error;
- lineage condition markers;
- seed uncertainty.

Primary visual questions:

1. do rank-matched and isotropic perturbations differ at identical \(e_F\)?
2. do real operator points depart from the rank-matched calibration curve?

---

## Figure 3 — Functional versus privacy decay

Show:

\[
R_{\mathrm{func}}(e)
\]

and:

\[
R_{\mathrm{priv}}(e)
\]

with:

- \(e_{50}^{\mathrm{func}}\);
- \(e_{50}^{\mathrm{priv}}\);
- \(\Delta e_{50}\).

---

## Figure 4 — Residual privacy recovery

\[
u=r-\hat f^{\mathrm{rank}}(e)
\]

shown by operator/lineage.

---

## Figure 4B — O3 spectral-tail decomposition — conditional

Only if O3 is activated.

Show:

- \(e_{\mathrm{floor}}(s)\rightarrow R_{\mathrm{priv}}^{\mathrm{trunc}}(s)\);
- rank-matched additive curve at matched errors;
- selected O3 recovered point;
- \(e_{\mathrm{solver,origscale}}\);
- effective and stable rank.

Annotate:

\[
\Delta_{\mathrm{tail,priv}}(s)
\]

and the corresponding functional comparison.

This separates operator-imposed loss, solver error, and spectral-location effects.

---

## Figure 5 — Audit gaps

Show paired raw:

\[
D_L,\quad D_R
\]

as primary, with normalized:

\[
G_L,\quad G_R
\]

as cross-cell summaries and denominator-stability annotations.

---

## Figure 6 — Display-only L2/L3 map

2×2 classification plus threshold sensitivity.

---

## Figure 7 — Canary vs natural persistence

Construct-validity result.

---

## Figure 8 — DP measurement floor

Parameter recovery and oracle privacy signal across privacy strength if the epsilon expansion is activated.

---

# 45. Interpretation guardrails

Never write:

> “Reconstruction increased the information available to the attacker.”

Write:

> “Reconstruction made information already present in the joint lineage more exploitable by the implemented attack.”

Never write:

> “The reconstructed model has epsilon X.”

Write:

> “Auditing through the reconstructed channel yields an empirical lower bound on the same underlying training/release mechanism.”

Never write:

> “TIES is non-identifiable.”

unless a formal impossibility proof exists.

Write:

> “Under the tested recovery family and attacker knowledge, TIES occupied a non-recovery region.”

Never write:

> “DP prevented recovery.”

unless the measured object is parameter recovery and the causal claim is justified.

DP training noise and privacy attack power are distinct quantities.

---

# 46. Evidence policy

Use:

> **Evidence quality gates factual claims. Public existence gates priority claims.**

Therefore:

- unreviewed work may defeat an unconditional “first” claim;
- unreviewed work does not establish empirical truth merely by existing;
- all numerical prior-work claims require verified primary-source support before appearing in the dissertation.

---

# 47. Final implementation instruction

The team should now implement **P0 only**.

Do not build the full grid.

Do not add TIES.

Do not add \(k=8\).

Do not add a second architecture.

Do not start a broad epsilon sweep.

The execution order is:

```text
P0-PRE Parameterised scorer + DRY-A…H + cache assertions
   ↓
P0-0  Compute benchmark + reference caching
   ↓
P0-A0 Blind split / deduplication leakage check
   ↓
P0-A  Oracle privacy power + matched no-canary contamination control
   ↓
P0-B  Canary ↔ natural construct validity
   ↓
P0-C1 Exact linear recovery plumbing
   ↓
P0-F1 Coarse ladders through e=2.0 + zero-delta anchor
   ↓
P0-F2 Pre-registered crossing refinement where eligible
   ↓
P0-C2 Spectral/common-source baseline
   ↓
P0-D  DARE utility gate
   ↓
P0-D2 O3 SVD-truncation fallback gate only if DARE fails
   ↓
P0-D3 Conditional O3 truncation ladder if O3 is selected
   ↓
P0-C4 Conditional O3 incomplete-lineage recovery validation
   ↓
P0-E  Derivative-parent deployment-realism scan
   ↓
MVRS completion review by 2 Sep
   ↓
P1 launch only if all gates pass before 3 Sep 2026 18:00 BST
   ↓
otherwise CALENDAR_COMPRESSED_MVRS + writing
```

---

# 48. Final status

```text
RESEARCH DIRECTION: LOCKED
MEASUREMENT SPEC: v1.9 FINAL CLOSED — PRE-DATA EXECUTION AUTHORITY
BROAD LITERATURE SEARCH: CLOSED
FULL EXPERIMENT GRID: NOT AUTHORIZED
NEXT ACTION: BUILD/PASS NON-COMPRESSIBLE P0-PRE, RUN BENCHMARK, THEN OPEN P0-A0
```


# 48. Final closure rule

This specification is now:

```text
CLOSED
```

Do not initiate another speculative reviewer cycle.

A v2 measurement specification is permitted only if one of the following is documented:

```text
ACTUAL_IMPLEMENTATION_IMPOSSIBILITY
DIRECT_MATHEMATICAL_ERROR
ACTUAL_EMPIRICAL_MEASUREMENT_REQUIRING_PREDECLARED_BRANCH
FORMAL_SUPERVISOR_REQUIREMENT
```

The following are not sufficient reasons:

```text
another reviewer proposes extra robustness
a new model/library becomes fashionable
an optional experiment sounds interesting
a null result is disappointing
P1 no longer fits the calendar
```

The next normal project document is an execution/result artifact, not another methodology revision.
