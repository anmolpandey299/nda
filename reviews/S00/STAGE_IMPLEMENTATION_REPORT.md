# STAGE_IMPLEMENTATION_REPORT — S00

**Stage:** `S00` repository bootstrap
**Role:** IMPLEMENTER [AUTH: 01 §3.2]
**Controlling plan:** `stage_acceptance/S00/01_PLAN.md` (accepted, review round 2 of 2)
**Branch:** `stage/s00` [AUTH: 01 §27]

---

## 1. Files created / changed

Root commit; every path below is new. 77 files tracked.

```text
CLAUDE.md                                  operational memory, 111 lines   01 §28
README.md                                  entry point                     01 §11
Makefile                                   11 targets                      01 §33; plan §10
pyproject.toml                             requires-python frozen, tooling 01 §12, §33
uv.lock                                    CPU/dev set, 46 packages        01 §12(5); plan §5.2
Dockerfile                                 cpu-dev lane + gated science    01 §12(8)(9)
.gitignore                                 working-tree noise + cache only 01 §27, §36
.githooks/pre-commit .githooks/pre-push    convenience guard               01 §27, §39 S00
.github/workflows/ci.yml                   CPU gate only, no GPU runner    01 §39 S00, §6.1
specs/*.md (4)                             authority copies                01 §39 S00
specs/SPEC_HASHES.json                     frozen SHA256                   01 §15
specs/deviations/SPEC_DEVIATIONS.md        register                        00 §0; 01 §29
specs/deviations/SCORER_EXCEPTIONS.md      register + standing prohibition 00 §34A.5; 02 §C2
configs/{models,data,training,attacks,recovery,p0,p1}/    structure only    01 §11, §17
src/__init__.py + 11 stage packages        __init__.py only, owner named   01 §11, §39
tests/{unit,integration,synthetic,golden,backend_contract,gpu_smoke}/       01 §11, §22; 02 §C6
tests/conftest.py                          lane markers, no network        01 §22
tests/unit/test_s00_invariants.py          A1-A4, A7-A9, A11-A18
tests/integration/test_s00_command_surface.py   A5, A6, A10, I15
manifests/{models,data,environments,runs}/ provenance paths                01 §13-§16
manifests/environments/AI_ENGINEERING_STACK_S00.json   Claude + Codex      01 §2.1, §4, §48
results/{p0,p1}/ artifacts/{p0_pre,runs,cache}/ logs/                      01 §11
artifacts/p0_pre/P0_PRE_READINESS.json     three flags, provenance-bound   02 §C6
reviews/S00/                               reviews, responses, deferred    03 §2-§11
stage_acceptance/S00/                      13 bundle slots, 20 §26 fields  01 §26, §45
scripts/check_repo_invariants.py           I0-I14, 463 lines               plan §13
scripts/preflight.py                       steps 0-8, env identity, I15    plan §10.1, §11
scripts/capture_environment.sh             01 §12 steps 2-9
scripts/install_git_hooks.sh               core.hooksPath
```

Relocations, not rewrites:

```text
STAGE_PLAN.md            -> stage_acceptance/S00/01_PLAN.md   canonical path, plan D10
CLAUDE_BLIND_REVIEW.md   -> reviews/S00/CLAUDE_REVIEW.md
CODEX_BLIND_REVIEW.md    -> reviews/S00/CODEX_REVIEW.md
00..03_*.md              -> copied to specs/; the root originals were left in place and are
                            guarded against divergence by I4 (byte-identical or violation)
```

## 2. Commands executed

```text
git init -b stage/s00                       # §8.1 precondition: a real pinned worktree
python3 -m venv <scratch>/toolvenv && pip install uv      # uv bootstrap, outside the repo
uv lock                                     # 01 §12(1) -> committed CPU/dev uv.lock
uv sync --extra cpu-dev                     # 46 packages from the lock
bash scripts/install_git_hooks.sh
make format ; make format CHECK=1 ; make lint ; make typecheck
make unit ; make integration ; make synthetic ; make backend-contract ; make gpu-smoke
python scripts/check_repo_invariants.py
make preflight
git add -A && git commit                    # root commit f7e6ee0
```

## 3. Exact test results

```text
make format CHECK=1        PASS   23 files already formatted
make lint                  PASS   All checks passed!            (ruff 0.16.4)
make typecheck             PASS   no issues found in 23 source files (mypy 2.3.1, strict)
make unit                  PASS   39 passed                     (pytest 9.1.1)
make integration           PASS   19 passed
make synthetic             NOT_RUN(EMPTY_AT_S00)                exit 0, never PASS
make backend-contract      NOT_RUN(BACKEND_NOT_INTEGRATED)      exit 0, never PASS
make gpu-smoke             NOT_RUN(NO_GPU)                      exit 0, never PASS
check_repo_invariants      PASS   0 violation(s)
make preflight             PASS   steps 0-8, P0_PRE_READY=false
```

Raw logs: `stage_acceptance/S00/04_TEST_OUTPUTS/`.

Negative controls actually exercised, not asserted:

```text
A4   hand-authored {"status":"PASS"} for all 10 evidence keys      -> all flags stay false
A4   record with run_id + env hash but no run manifest             -> NON_EVIDENTIARY
A6   evidence bound to a different ENVIRONMENT_LOCK_SHA256         -> flag not raised
A6   fake runner failing at step 3                                 -> steps 4-8 never ran
A7   src/analysis/privacy_metrics.py defining tpr_at_fpr           -> I13 violation
A7   make_folds / calibrate_threshold / cross_fit_scores           -> I13 violation
A7   duplicate reserved module without SCORER_EXCEPTION            -> I1 violation
A7   same duplicate WITH a recorded SCORER_EXCEPTION               -> permitted
A10  commit on main; .py staged under reviews/ outside scratch/    -> blocked
A10  git commit --no-verify; unset core.hooksPath                  -> both bypass (by design)
A10  write to a read-only production path as reviewer              -> PermissionError
A15  one altered component, for each of the 8                      -> identity changes
A15  any component TBD                                             -> identity is TBD, not a hash
A17  TARGET_FPR / MIN_K / SEEDS / DARE_P / SVD_RANK / N_BOOTSTRAP / ALPHA as source literals
                                                                   -> I14 violation
A17  TARGET_FPR = load('attacks')['target_fpr']                    -> permitted
I15  staged, unstaged and untracked production changes             -> all three detected
```

## 4. Unresolved issues

Full text in `stage_acceptance/S00/09_UNRESOLVED.md`.

```text
U-01  S00-B environment capture has not run; no H100 was available.       BLOCKS ACCEPTANCE
U-02  S00-CBR-005 (.gitignore citation), deferred at review round 2 to S01, ticket S00-T01.
```

Three implementation defects were found and fixed during the stage, none of them by weakening
a test: the I7 and provider checks matched their own detector source (patterns now built from
parts, provider check narrowed to real integration idioms); `pre-commit` hard-failed on an
unborn HEAD, which blocked the root commit; `pre-commit` hard-failed in a clone with no
toolchain.

Four deviations from the accepted plan are declared in `09_UNRESOLVED.md` rather than made
silently: `scripts/preflight.py` as a fourth script, the `software_gate` block in the readiness
file, `02_CHANGED_FILES.txt` as a bundle slot, and `artifacts/p0_pre/evidence/`.

## 5. TBD_REQUIRES_HARDWARE items

No hardware value was fabricated. No version number, throughput, batch size, VRAM figure or
CUDA measurement appears anywhere in this stage's output [AUTH: 03 §8].

```text
ENVIRONMENT_LOCK_SHA256                      TBD_REQUIRES_HARDWARE
cuda_runtime, cuda_driver                    TBD_REQUIRES_HARDWARE
torch_version, torch_cuda_build              TBD_REQUIRES_HARDWARE
transformers / peft / accelerate versions    TBD_REQUIRES_HARDWARE (CUDA-coupled)
DP library identity + version                TBD_REQUIRES_HARDWARE (selected S00-B, DP smoke S06)
nvidia_smi_capture, gpu_model/uuid/count     TBD_REQUIRES_HARDWARE
docker_image_tag, docker_image_digest        TBD_REQUIRES_HARDWARE
bf16_fp32_tolerance                          TBD_REQUIRES_HARDWARE (measured, 01 §10)
nondeterminism_sources + run-to-run tolerance TBD_REQUIRES_HARDWARE (measured, 01 §30)
batch size, throughput q, peak VRAM, GPU-hours  TBD_REQUIRES_HARDWARE (S10, 00 §0.2.4)
```

Procedure, record and acceptance for each are unchanged from plan §5.7: resolve with
`make env-capture` on the RunPod H100 SXM image per 01 §12 steps 2-9; record in
`manifests/environments/<env_id>.json`; accept with `tests/gpu_smoke/`. The S10 rows record in
`artifacts/p0_pre/P0_00_COMPUTE_BUDGET.json`.

The CPU/dev set is deliberately **not** on this list: `requires-python`, ruff, mypy, pytest,
NumPy, SciPy, scikit-learn, pandas, safetensors, tokenizers and matplotlib are pinned in the
committed `uv.lock`, so `make lint`, `make typecheck` and `make unit` are reproducible from
this commit alone [AUTH: 01 §33, §46; plan §5.2, C-03 closure].

## 6. Git commit / status

```text
branch      stage/s00           (main and experiment-frozen do not exist yet; they are
                                 created at acceptance and are protected [AUTH: 01 §27])
commits     f7e6ee0eadfb9550a1c5a1f74a0029960baedcf5  "S00: repository bootstrap" (root)
            aaac605d51ce04614f76a4b846367f6eef26893c  "S00: acceptance bundle, report,
                                                       AI-stack provenance"
            this report and the refreshed artifact manifest are recorded in the commit that
            follows aaac605, because a manifest cannot hash the commit containing it
core.hooksPath  .githooks
working tree    clean at hand-off; production_tree_dirty() reports []
```

Nothing was pushed. No tag was created; tagging is an acceptance action [AUTH: 01 §27].

## 7. Were any real scientific or model data inspected?

```text
REAL_SCIENTIFIC_DATA_INSPECTED = FALSE
```

No research model was downloaded or run, no dataset acquired, no membership label, privacy
outcome or experimental result generated or viewed. `results/p0/` and `results/p1/` are empty.
Every test runs on CPU with no network. Full statement:
`stage_acceptance/S00/11_DATA_INSPECTION_STATEMENT.md` [AUTH: 01 §26(19), §29].

## 8. S00 acceptance checklist

```text
[x] A1  repo structure; forbidden paths absent (notebooks/, serving/, root STAGE_PLAN.md)
[x] A2  four authority copies; SHA256 match record AND test literals
[x] A3  stage-plan glob non-empty; 01_PLAN.md = 1,186 lines <= 1,200
[x] A4  readiness false / STATISTICAL_STACK_ONLY / false; forged evidence rejected
[x] A5  all 11 make targets exist; three lanes report NOT_RUN, never PASS
[x] A6  step order matches 01 §22; halts at first failure; env-mismatch rejected
[x] A7  reserved API confined to src/scoring/; import boundary blocks a second estimator
[x] A8  no notebook execution path
[x] A9  no importable code under reviews/; packaging restricted to src/
[x] A10 real pinned worktree; hooks install; protected branch and reviews/ blocked;
        --no-verify and unset hooksPath bypasses demonstrated; reviewer write fails
[x] A11 CI invokes only make preflight; no GPU runner; no backend-contract
[x] A12 provenance paths reserved; RUN_ID input list is exactly the eight 01 §15 inputs
[x] A13 AI stack record carries Claude and Codex blocks; unavailable = UNAVAILABLE_NOT_EXPOSED
[x] A14 no fabricated hardware values
[x] A15 environment manifest schema; identity covers all 8 components; TBD stays TBD
[x] A16 M_PRIMARY declared; prohibited estimator idiom absent and detectable
[x] A17 no 01 §17 material constant as a source literal; injections rejected
[x] A18 bundle carries all twenty 01 §26 fields
[x] format / lint / typecheck / unit / integration PASS
[x] scientific spec unchanged (I4 verifies all four documents)
[ ] S00-B environment evidence PRESENT           <- U-01, blocks acceptance [plan §19]
[ ] repo reproducible from the frozen scientific environment  <- follows U-01
```

Stage state:

```text
BACKEND_INTEGRATED = FALSE
SUITE_SCOPE        = STATISTICAL_STACK_ONLY
P0_PRE_READY       = FALSE
S00_ENV_CAPTURE    = TBD_REQUIRES_HARDWARE
```

S01 was not started [AUTH: 01 §41].
