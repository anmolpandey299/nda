# CLAUDE.md — operational memory

Short and operational [AUTH: 01 §28]. It does not duplicate the specs.

> Read the relevant spec sections for the current task. Never summarize the entire spec into
> memory and then work from the summary alone.

## 1. Project purpose

Measure privacy leakage and constituent recoverability in ordinary model-merge release
families: does parameter recovery imply privacy recovery, and is joint lineage worth more
than a single descendant? [AUTH: 00 §1]

## 2. Controlling documents

`specs/00_MEASUREMENT_SPEC_v1.9_FINAL_CLOSED.md` scientific authority ·
`specs/01_EXECUTION_STACK_LOCK_v2.md` engineering authority ·
`specs/02_PRE_EXECUTION_IMPLEMENTATION_CORRECTIONS.md` binding clarifications ·
`specs/03_REVIEW_GOVERNANCE_LOCK.md` binding review process.
Hashes: `specs/SPEC_HASHES.json`. Accepted stage plan: `stage_acceptance/<stage>/01_PLAN.md`.

## 3. Non-negotiable invariants — pointers, not restatements

```text
M_PRIMARY = common-FPR TPR@1%FPR; operational TPR/FPR reported beside it   02 §C1
one tie-safe fixed-FPR estimator, owned by src/scoring/                    02 §C2; 00 §34A.2
individual-record ROC + "max TPR at repeated FPR" is PROHIBITED            02 §C2, §C8
one scorer, one cross-fitting controller                                   00 §34A.1, §34A.2
pooled view never invokes the model                                        00 §34A.3
cache key includes SCORING_CODE_HASH and ENVIRONMENT_LOCK_SHA256           02 §C7; 01 §12, §32
no calibration/evaluation overlap; no evaluation IDs in calibration        01 §23; 00 §34B.1
arm-matched thresholding                                                   01 §15.1; 01 §23
Min-K% reduced inline; per-token arrays not persisted                      00 §14.2
DRY-G gate = family-wise max-statistic, N_FWER_CALIBRATION = 2_000         02 §C4
material constants live in configs/**, never in source                     01 §17
no notebook is an execution path                                           01 §18
no result without a valid run manifest -> NON_EVIDENTIARY                  01 §16
no result-peeking before a component is frozen                             01 §29
seeds recorded separately; never one undocumented global seed              01 §30
statistics live in importable modules, not in plotting scripts             01 §34
```

## 4. Repo commands

```text
uv sync --extra cpu-dev     reproduce the CPU/dev lane from the committed uv.lock
make format [CHECK=1]       apply / verify formatting
make lint                   ruff check
make typecheck              mypy over typed core modules
make env-capture            01 §12 steps 2-9 on the real H100 image
make preflight              ordered gate (steps 0-6) + readiness evaluation (steps 7-8)
```

## 5. Test commands

```text
make unit               pure functions and mathematical identities
make integration        multiple modules, tiny fixtures, no network
make synthetic          planted-truth DRY-A..H and DRY-CACHE, plus golden
make backend-contract   NOT_RUN while BACKEND_INTEGRATED = FALSE
make gpu-smoke          NOT_RUN without an H100
```

Order is authority: unit -> integration -> synthetic/golden -> gpu smoke -> scientific gate
[AUTH: 01 §22]. An empty lane reports `NOT_RUN(<reason>)`, never `PASS`.

## 6. Stage workflow

`S00` bootstrap -> `S01` provenance/config -> `S02` tiny fixtures -> `S02B` model
compatibility gate -> `S03` scorer/cache/cross-fit -> `S04` statistical engine -> `S05` data
-> `S06` training -> `S07` merge -> `S08` recovery -> `S09` registry -> `S10` benchmark ->
`P0-A0`/`P0-A` [AUTH: 01 §39, §50]. One bounded stage per branch `stage/<stage-id>`; every
accepted stage is tagged [AUTH: 01 §27, §41].

## 7. Review workflow

Architect -> implementer -> deterministic tests -> blind Claude review -> blind Codex review
-> reconciliation -> architect response (`FIX`/`REJECT`/`DEFER`) -> mechanical adjudication
[AUTH: 03 §2, §10, §11]. Maximum two rounds [AUTH: 03 §12]. Reviews write only to
`reviews/<stage>/`; diagnostics only to `reviews/<stage>/scratch/` [AUTH: 03 §9].
Plans are capped at 1,200 lines [AUTH: 03 §13].

## 8. Forbidden actions

```text
no production edit by a reviewer session                 01 §3.3; 03 §9
no commit to main / experiment-frozen                    01 §27
no drive-by refactor; one bounded stage per branch       01 §27, §41
no scientific-threshold change; no result-driven tuning  01 §3.2; 00 §34B.1A
no pip install -U during a run                           01 §12
no floating `main` model revision                        01 §8G
no OpenRouter anywhere in the engineering path           01 §6.2, §24
no second scoring / cross-fit / fixed-FPR path           00 §34A.2, §34A.5; 02 §C2
no evidentiary run from a dirty production tree          01 §35(3), §36
no overwrite of a failed run directory                   01 §36
no rewrite of published experiment history               01 §27
```

## 9. Artifact locations

```text
manifests/models|data|environments|runs/   provenance                 01 §13-§16
artifacts/runs/<RUN_ID>/                   immutable run directory    01 §36
artifacts/cache/                           content-addressed cache    00 §34A.4
artifacts/p0_pre/                          P0-PRE gate artifacts      00 §34B.2
results/p0|p1/                             result tables              00 §43, §44
logs/<RUN_ID>/                             stdout/stderr              01 §16
reviews/<stage>/                           blind reviews, responses   03 §2-§11
stage_acceptance/<stage>/                  01 §45 bundle, 20 §26 fields
specs/deviations/                          SPEC_DEVIATION, SCORER_EXCEPTION
```
