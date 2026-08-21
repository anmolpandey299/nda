# ARCHITECT_FINDING_RESPONSES — S00

**Stage:** `S00`
**Round:** 2 of 2 — final [AUTH: 03 §12]
**Responds to:** `CLAUDE_BLIND_REVIEW.md` (verdict FAIL), `CODEX_BLIND_REVIEW.md` (verdict FAIL)
**Patched artifact:** `STAGE_PLAN.md`, 1,186 lines (cap 1,200) [AUTH: 03 §13]
**Response vocabulary:** exactly one of `FIX` / `REJECT` / `DEFER` per unique finding
[AUTH: 03 §10]

---

## 1. Claude blind-review findings

### C-01 — Environment identity collapsed to `uv_lock_sha256`; absent from cache key and RUN_ID
`FIX` — §5.6 defines `ENVIRONMENT_LOCK_SHA256 = SHA256(uv_lock_sha256, docker_image_digest,
cuda_runtime, cuda_driver, torch_version, torch_cuda_build, python_version, gpu_model)` and sets
`env_id` to it; §7.1 consumes it as `RUN_ID`'s environment input and adds it to the
content-addressed cache key; I12 and the S03 negative-test set gain `environment change → miss`;
each cache entry records its producing `env_id` and `RUN_ID`. Closure test: A15, plus the S03
DRY-CACHE environment-change case.

### C-02 — No single-owner reservation for the fixed-FPR ROC estimator; 02 §C1/§C2 never carried
`FIX` — §1.1 now lists 02 §C1, §C2, §C4, §C5, §C8. New §13.1 fixes
`M_PRIMARY(V) = TPR@1%FPR` at a common held-out FPR, names `src/scoring/` as the sole owner of
fixed-FPR ROC interpolation (00 §34A.2 duty 8), prohibits the individual-record cumulative ROC
plus "max TPR at repeated FPR" as a standing exclusion in `specs/deviations/SCORER_EXCEPTIONS.md`,
and requires the tie-safe distinct-threshold estimator. I1 gains the `roc` reservation and an
import-boundary check; I13 requires every headline consumer to route through the one API.
Closure test: A16, A7 (import-boundary case).

### C-03 — S00-A ships no lock and no pins; CPU tooling deferred as TBD_REQUIRES_HARDWARE
`FIX` — §5.2 makes S00-A freeze `requires-python` and commit a `uv.lock` covering the CPU/dev set
(ruff, mypy, pytest, NumPy, SciPy, scikit-learn, pandas, safetensors, tokenizers, plotting); only
the CUDA-coupled set remains unresolved. §5.7 drops the interpreter, CPU-library, plotting and
tooling rows and states that only genuinely H100/CUDA-dependent values may appear. §5.1 preamble
makes the S00-A Dockerfile build the CPU/dev lane from the S00-A lock. §15 lists `uv.lock` as an
S00-A artifact. Closure test: A14, A15.

### C-04 — preflight step 7 consumes unauthenticated recorded state; readiness is forgeable
`FIX` — §10.1 step 7 becomes *verify provenance-bound evidence records*: each input must carry
`RUN_ID`, `ENVIRONMENT_LOCK_SHA256`, its run-manifest path and per-artifact SHA256; step 8
refuses any TRUE flag unless the recorded environment identity equals the current environment
manifest; a record without a valid run manifest is `NON_EVIDENTIARY`. §11 replaces the flat
evidence strings with the bound-record schema and states that a hand-written `{"status":"PASS"}`
is not evidence. Closure test: A4 forged-evidence case, A6 env-mismatch case.

### C-05 — Bundle drops 01 §26 item 19, the only mechanical no-result-peeking attestation
`FIX` — R4 is rewritten to apply R2's higher-requirement rule: 01 §45 fixes the layout and all
twenty 01 §26 fields must be present within it. New §8.3 maps every field to a slot, adding
`03A_RESOLVED_CONFIGS/` (item 8), `11_DATA_INSPECTION_STATEMENT.md` (item 19) and a named
requested-verdict field in `00_INDEX.md` (item 20). §3 tree updated. Closure test: A18.

### C-06 — 01 §17 material-constants rule has no owner, path or invariant
`FIX` — I14 requires target FPR, Min-K fraction, alpha, k, DARE p, SVD rank, bootstrap replicates
and seeds to resolve from `configs/**` and to appear as a module-scope constant in no `src/**`
file. I4 and A2 additionally assert the four spec SHA256 values as literals in
`tests/unit/test_s00_invariants.py`, so regenerating `SPEC_HASHES.json` alone cannot clear the
gate. Closure test: A17 (injected `TARGET_FPR = 0.01` must fail), A2.

### C-07 — `STAGE_PLAN.md` has no location in the repository
`FIX` — D10 declares `stage_acceptance/<stage>/01_PLAN.md` the canonical stage-plan path; the
architect writes it there directly, so plan and bundle cannot diverge. I5 globs
`stage_acceptance/*/01_PLAN.md` and requires a non-empty glob, so the cap is mechanised from S01
onward. §3 tree and §15 updated. Closure test: A3.

### C-08 — Codex provenance record required at S00 is absent
`FIX` — `AI_ENGINEERING_STACK_S00.json` gains a Codex block carrying the 01 §4 field set (client
/ product, client version, exact model name, review date/time UTC, prompt SHA256, repository
commit, review-output SHA256) beside the existing 01 §2.1 Claude block; any field the product
does not expose is written `UNAVAILABLE_NOT_EXPOSED`, never invented. Closure test: A13.

### C-09 — Git-hook enforcement is opt-in and bypassable; I9 is not structural
`FIX` — §8.1 is restructured into enforcement layers with protected branches and stage tags named
as the authority mechanism, the reviewer worktree second, the I15 clean-state gate third, and
hooks explicitly demoted to a convenience guard that `--no-verify` and an unset `core.hooksPath`
bypass; preflight step 0 warns on a mismatched `core.hooksPath`. I9 is relabelled accordingly.
Closure test: A10 (`--no-verify`, unset-hooksPath and reviewer-write cases).

### N-01 — Reviewer isolation unenforced in fact for S00's own review
`FIX` — §8.1 adds a precondition: before S00 implementation review begins the repository must be
an initialised Git repository with a resolvable commit SHA, and each reviewer session must be a
separate worktree at that pinned commit, read-only outside `reviews/<stage>/scratch/`. A review
where `git rev-parse HEAD` fails cannot supply 01 §26 items 4–5 and is not a valid blind review.
F12 records the branch; §19 makes it an acceptance criterion; §20.1 item 3 flags it as currently
unmet. Closure test: A10.

### N-02 — Unreconciled authority conflict on the DRY-G calibration bank
`FIX` — new R7: 02 §C4 governs the production DRY-G gate at `N_FWER_CALIBRATION = 2_000` with
artifact `P0_PRE_DRYG_FAMILYWISE_CALIBRATION.json`; the 00 §34B.1A 200-trial bank and the 00 §42
checklist wording are historical and superseded for that purpose; DRY-B / DRY-D1 positive-control
tolerances are regenerated under 02 §C5 rather than inherited. §12's P0-PRE gate row records the
governing calibration. Closure reference: R7, §12.

---

## 2. Codex blind-review findings

### S00-CBR-001 — BLOCKER — no Git worktree; reviewed revision unidentifiable; production writable
`FIX` — same closure as N-01: §8.1 pinned-worktree precondition with read-only enforcement
outside `scratch/`, F12 failure branch, §19 acceptance criterion, A10 negative test proving a
reviewer-role write to a production path fails. The blind review must be rerun against the pinned
worktree once the repository is initialised [AUTH: 03 §9; 01 §25, §26(4)(5)].

### S00-CBR-002 — MAJOR — dirty tree can produce results attributed to a clean commit
`FIX` — I15 requires the launcher to reject staged, unstaged **and** untracked changes under
production paths (`src/`, `configs/`, `specs/`, `pyproject.toml`, `uv.lock`, `Dockerfile`) before
any run manifest is written; dirty state refuses execution rather than being recorded as a flag.
F11 records the branch; §20 row "no result without provenance" cites it. Owner S09
[AUTH: 01 §35(3), §16, §27, §36].

### S00-CBR-003 — MAJOR / PROMPT_ONLY — F1 acceptance escape; env identity absent from RUN_ID/cache
`FIX` — both halves closed. F1 is rewritten as explicitly **not an acceptance branch**: CPU-lane
S01–S04 development may proceed on the S00-A lock, but S00 acceptance and any evidentiary run may
not. §19 replaces "S00-B closed, or F1 recorded" with a required `S00-B environment evidence
PRESENT` criterion. The identity half is closed by C-01's fix, which puts image digest and CUDA
runtime/driver inside `ENVIRONMENT_LOCK_SHA256` and inside both `RUN_ID` and the cache key.

### S00-CBR-004 — MAJOR — second fold-selection / prohibited ROC path evades basename uniqueness
`FIX` — same closure as C-02: I1 replaces pure basename uniqueness with an API reservation plus an
import-boundary check forbidding `src/analysis/` and `src/attacks/` from defining fold
construction, calibration-threshold estimation or fixed-FPR ROC interpolation; I13 requires every
headline consumer of `M_PRIMARY`, `D_L`, `D_R`, `R_priv`, `G_L`, `G_R` to route through the one
`src/scoring` API; §13.1 asserts exact 1% common-FPR semantics and tie-safe behaviour. Closure
test: A16, A7.

### S00-CBR-005 — MINOR / UNJUSTIFIED_SCOPE — `.gitignore` cites 01 §11, which does not mandate it
`DEFER`
```text
ticket        S00-T01
reason        Round 2 is scoped to the ten accepted findings; this finding concerns the citation
              attached to one zero-burden file, not scientific correctness, leakage, statistical
              validity, reproducibility, provenance or cache correctness [AUTH: 03 §6].
target stage  S01, where run/artifact/cache write paths make the ignore policy substantive
record        reviews/S00/deferred.md
```
Permitted because the finding is non-BLOCKER [AUTH: 03 §10]. The architect does not contest the
finding: `.gitignore` is not named in 01 §11's tree, and its citation must be corrected or the
file removed at S01.

---

## 3. Closure summary

```text
findings received        16   (Claude 11, Codex 5)
FIX                      15
REJECT                    0
DEFER                     1   (S00-CBR-005 -> ticket S00-T01)
unresolved BLOCKER        0
BLOCKER deferred          0   (prohibited by 03 §10)
```

Overlap diagnostic, process use only [AUTH: 03 §15]:

```text
OVERLAP        C-01 / S00-CBR-003 (environment identity)
               C-02 / S00-CBR-004 (second estimator path)
               C-09 + N-01 / S00-CBR-001 (read-only enforcement, pinned worktree)
CLAUDE_ONLY    C-03, C-04, C-05, C-06, C-07, C-08, N-02
CODEX_ONLY     S00-CBR-002, S00-CBR-005
```

High overlap is not treated as evidence of correctness [AUTH: 03 §15].

## 4. Scope attestation

No component was added beyond what the accepted findings require. No new requirement was
introduced. No MINOR or NOTE finding outside the accepted set was actioned. The plan remains
within the 1,200-line cap [AUTH: 03 §13] and no hardware-dependent value was fabricated
[AUTH: 03 §8]. Round 3 is not opened [AUTH: 03 §12].

---

S00_REVIEW_ROUND = 2_OF_2
S00_PLAN_STATUS = READY_FOR_CLOSURE_VERIFICATION
NO_FURTHER_BROAD_ARCHITECTURE_REVIEW = TRUE
