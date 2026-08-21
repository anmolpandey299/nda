# STAGE_IMPLEMENTATION_REPORT — S00

**Stage:** `S00` repository bootstrap
**Role:** IMPLEMENTER [AUTH: 01 §3.2]
**Controlling plan:** `stage_acceptance/S00/01_PLAN.md` (accepted, review round 2 of 2)
**Branch:** `stage/s00` [AUTH: 01 §27]
**Pass:** final implementation-fix pass (FIX 1-9), closing the implementation BLOCKER/MAJOR set

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

## 3A. Final implementation-fix pass — FIX 1 to FIX 9

### Changed files

```text
Makefile                                    rewritten; $(READINESS) expansion repaired
scripts/preflight.py                        hardened evidence chain, recomputed identity,
                                            disjoint namespaces, gate_ok, backend gate CLI
scripts/check_repo_invariants.py            AST/import ownership analysis, case-insensitive
                                            constants, image-pin and CI-frozen invariants
scripts/build_bundle.py                     NEW — bundle generation and exact verification
scripts/make_review_worktree.sh             NEW — pinned read-only reviewer worktree
scripts/capture_environment.sh              image identity read from the image, not the caller
Dockerfile                                  base images pinned by immutable digest
.github/workflows/ci.yml                    uv lock --check + uv sync --frozen
pyproject.toml                              mypy override for the CUDA-coupled torch import
tests/_helpers.py                           NEW — provenance fixture builders
tests/unit/test_s00_final_fixes.py          NEW — 61 regression tests for FIX 1-7
tests/integration/test_s00_command_surface.py  FIX 1 branches, FIX 8 drift, FIX 9 worktree
tests/backend_contract/test_backend_contract.py   NEW — collected skeletons
tests/gpu_smoke/test_cuda_torch_contract.py       NEW — collected skeletons
tests/gpu_smoke/test_env_capture_contract.py      NEW — collected skeletons
tests/unit/test_s00_invariants.py           migrated to the disjoint evidence namespaces
tests/conftest.py                           tests/ on sys.path for shared helpers
```

### What each fix changed, at root cause

```text
FIX 1  The recipe interpolated `$$READINESS`, which the shell expanded to the PID, so the
       readiness file was never opened, the error was swallowed by 2>/dev/null and the
       lane ALWAYS reported NOT_RUN. Branch B was unreachable. The decision now lives in
       preflight.py with three exit codes: 0 run the lane, 1 NOT_RUN, 2 unreadable file =
       hard failure. Both branches and the unreadable case are tested.
FIX 2  An evidence record is now bound end to end: run_manifest must be under
       manifests/runs/, exist, parse, carry every 01 §16 field, and agree with the record
       on RUN_ID and environment identity; artifact hashes are recomputed; path traversal
       is refused. The current environment identity is RECOMPUTED from the frozen
       components, and a manifest whose declared identity disagrees is unusable.
FIX 3  Ownership is enforced by dependency analysis, not naming: aliased sklearn.metrics
       imports, class methods, module scope, and any scope that manipulates both a TPR and
       an FPR array or compares an FPR array to a numeric literal.
FIX 4  Constant patterns are compiled case-insensitively and cover lowercase spellings and
       annotated assignments.
FIX 5  gate_ok joined the SUITE_SCOPE conjunction, and lane vs software-gate evidence live
       in two disjoint directories with exact-case filename matching.
FIX 6  Base images pinned by digest; capture reads the identity baked into the image and
       records a caller-supplied digest as unresolved; the GPU and backend lanes ship
       collected skeletons so they can never collect zero tests.
FIX 7  CI runs `uv lock --check` then `uv sync --frozen`; an invariant fails if either is
       missing.
FIX 8  scripts/build_bundle.py regenerates and verifies the bundle. `make bundle-verify`
       fails if any path outside stage_acceptance/S00/ or reviews/S00/ differs between the
       described commit and HEAD, so the bundle cannot describe stale code.
FIX 9  scripts/make_review_worktree.sh creates a separate worktree at a pinned commit,
       read-only everywhere except reviews/<stage>/scratch/.
```

### Commands

```text
make format CHECK=1 ; make lint ; make typecheck
make unit ; make integration ; make synthetic ; make backend-contract ; make gpu-smoke
make backend-contract READINESS=<fixture true>      # branch B, previously unreachable
python scripts/check_repo_invariants.py --root .
make bundle ; make bundle-verify ; make preflight
bash scripts/make_review_worktree.sh <commit> <dest> S00
```

## 3B. S00-B science-environment finalisation

### Hardware probe outcome — stock pod REJECTED

The first real RunPod H100 probe is recorded in
`manifests/environments/S00B_HARDWARE_PROBE.json` and is explicitly **not** accepted as
S00-B. Observed: NVIDIA H100 80GB HBM3, compute capability 9.0, driver 580.126.09, host CUDA
13.0, `torch.cuda.is_available()` true. Rejected because the stock container ships Python
3.11.10 (violating the frozen `requires-python >=3.13,<3.14`), a preinstalled torch
2.4.1+cu124 with no CPython 3.13 build, no uv, no Docker and no `/etc/pmm-image.json`.

### Selection, and why

```text
SCIENCE_PYTHON      3.13
SCIENCE_TORCH       2.13.0
SCIENCE_CUDA_BUILD  cu130
```

torch 2.13.0 is the current production release carrying a `cp313 manylinux_2_28_x86_64`
wheel; verified against the PyPI release index and the download.pytorch.org cu130 index
rather than assumed. The stock 2.4.1+cu124 is not reused: it has no CPython 3.13 build at
all, so reusing it would break the already-frozen interpreter requirement. The `cu130` build
matches the driver's reported CUDA 13.0 exactly, so no minor-version compatibility fallback
is relied upon. Resolution was proved before pinning: transformers 5.15.1, peft 0.20.0,
accelerate 1.14.0 and opacus 1.6.0 all resolve against it on cp313.

The CUDA runtime is itself locked — torch pulls 15 pinned `nvidia-*` wheels — so the science
image needs no mutable NVIDIA base image and the runtime is hash-pinned like everything else.

One CPU/dev pin moved as a consequence: `tokenizers` 0.23.1 -> 0.22.2, constrained by
transformers 5.15.1. This is the F4 branch working as designed; the S00-B lock is
authoritative and the move is recorded here [AUTH: plan §17 F4].

### Image and workflow — no Docker-in-Docker on RunPod

An image cannot contain its own digest, so identity is sealed in two passes: build and push
the `science` stage, read its immutable digest, then build `science-sealed` FROM that digest
and bake `image_ref`, `image_digest`, `base_image_digest`, `source_git_commit` and
`lane: "science"` into `/etc/pmm-image.json`. `capture_environment.sh` still reads that file
from inside the image and still forces a caller-supplied digest to `TBD_REQUIRES_HARDWARE`.

```text
scripts/build_science_image.sh       build + push + seal, OFF the pod
scripts/bootstrap_runpod_s00b.sh     11 fail-closed gates, then capture and the lanes
stage_acceptance/S00/10_REPRODUCE.md exact commands for all three steps
```

### Not measured, and not invented

`bf16_fp32_tolerance`, `nondeterminism_sources`, batch size, throughput, peak VRAM and the
GPU-hour projections remain `TBD_REQUIRES_HARDWARE` until the sealed image actually runs on
the H100 [AUTH: 03 §8; 00 §0.2.4; 01 §30]. `ENVIRONMENT_LOCK_SHA256` is still unresolved, so
`P0_PRE_READY` remains false and no run can be evidentiary.

## 3C. S00-B platform repair

RunPod rejected the first sealed image with `no matching manifest for linux/amd64 in the
manifest list entries`. Root cause: `scripts/build_science_image.sh` used a plain
`docker build` with no `--platform`, so on the Apple Silicon build host it produced a
`linux/arm64` image, pushed it, and read the digest back out of the local image store.

Repaired in `scripts/build_science_image.sh` only:

```text
both passes now  docker buildx build --platform linux/amd64 --target <stage> --push
digest source    buildx --metadata-file containerimage.digest, never the local image store
verification     docker buildx imagetools inspect must show a linux/amd64 manifest in the
                 PUSHED artifact, for pass 1 and pass 2; anything else fails closed
builder          an idempotent docker-container builder, since the default docker driver
                 cannot reliably push a cross-platform build
arm64            deliberately not built; the execution target is RunPod H100 linux/amd64
```

Preserved unchanged: the two-pass sealed design, immutable parent-digest resolution,
`/etc/pmm-image.json`, the exact `SOURCE_GIT_COMMIT`, and the fail-closed clean-tree gate.
The image-content selection (python 3.13, torch 2.13.0+cu130) is untouched.

The manifest verifier was exercised against seven registry shapes before commit: an index
carrying amd64 plus an `unknown/unknown` attestation, an arm64-only index, single-image
amd64 and arm64, a platform-keyed image map, capitalised JSON keys, and an empty document.
It accepts the four amd64 shapes and rejects the three others.


## 3D. S00-B host-virtualenv contamination repair

Observed on the real H100, inside the sealed linux/amd64 image:

```text
/repo/.venv/bin/python -> /opt/homebrew/opt/python@3.13/bin/python3.13
bash: .venv/bin/python: No such file or directory
```

Root cause confirmed by inspection of the repository, not inferred: the repository had **no
`.dockerignore` at all**, and the host `.venv/bin/python` is a symlink to
`/opt/homebrew/opt/python@3.13/bin/python3.13`. Both lanes ran `uv sync --frozen` and then
`COPY . .`, so the host macOS virtualenv entered the build context and overwrote the Linux
one the image had just built, leaving a dangling symlink. The system Python and the GPU were
fine; only the project interpreter was destroyed, which is why it surfaced at run time on the
H100 rather than at build time.

Repair, minimal and fail-closed:

```text
.dockerignore    NEW. Excludes .venv, **/.venv, venv, **/venv, __pycache__, **/__pycache__,
                 *.py[cod], .pytest_cache, .mypy_cache, .ruff_cache, .DS_Store and their **/
                 forms. .git is deliberately KEPT so /repo is a real pinned worktree and the
                 bootstrap can verify the commit and the clean tree inside the pod.
Dockerfile       Both lanes gain a fail-closed RUN immediately after `COPY . .`: it rejects a
                 .venv/bin/python symlinked into /opt/homebrew, /Users or Homebrew Cellar by
                 name, requires the interpreter to be executable, and then proves
                 sys.platform == "linux" and CPython 3.13. The science lane additionally
                 proves `import torch` and torch.__version__ == 2.13.0+cu130. No GPU is
                 needed for any of it during the image build.
invariant I16    check_build_context: .dockerignore must exclude the host-state patterns and
                 must not exclude .git; every stage containing `COPY . .` must re-prove the
                 image's own interpreter afterwards.
```

`uv sync --frozen --extra cpu-dev --extra science --no-install-project` remains the sole
creator of the environment; nothing is installed by hand. The selected Python, torch, CUDA
build and scientific package set are untouched, as are the two-pass sealed design, the
immutable digest mechanism, `SOURCE_GIT_COMMIT` semantics, the clean-tree gate and the
linux/amd64 platform repair.

### A second, related defect found while wiring I16

`check_image_pins` had been defined but never added to the `CHECKS` tuple, so the Docker
digest-pin invariant had never actually executed — its test passed because the test reads the
Dockerfile directly. Both `check_image_pins` and `check_build_context` are now registered, and
`test_every_invariant_function_is_registered` fails if any `check_*` function is ever left
unwired again.


## 3E. S00-B environment-capture repair

Gates 1-7 of the bootstrap passed on the real H100 inside the valid sealed image; capture
itself failed and left five fields unresolved. Each had a distinct cause, and each is fixed
against the controlling authority rather than by relaxing the requirement.

```text
cuda_runtime            capture required `nvcc`. The science image is runtime-only by design
                        and nvcc belongs to the CUDA development toolkit. Resolved instead
                        from the frozen installed distribution metadata, newest naming first:
                        nvidia-cuda-runtime, -cu13, -cu12. Observed value 13.0.96. Kept
                        separate from torch_cuda_build (13.0) and cuda_driver (580.126.09).
dependency_versions     capture ran `python -m pip freeze`; pip is absent by design. Replaced
                        with importlib.metadata over installed distributions: normalised,
                        deduplicated, sorted, no package manager, no network, no install.
bf16_fp32_tolerance     was defaulted to TBD. Now MEASURED: a fixed-seed, fixed-shape matmul
                        computed in FP32 (reference) and BF16 (compared) on the device, with
                        autocast explicitly disabled so no AMP default decides the arithmetic
                        [AUTH: 01 §10]. Records max/mean absolute error, max relative error
                        and the reference scale. No threshold is chosen anywhere.
nondeterminism_sources  was caller-supplied prose. Now captured structurally: cuDNN version,
                        enabled/deterministic/benchmark/allow_tf32, cuda.matmul.allow_tf32,
                        float32_matmul_precision, deterministic_algorithms,
                        CUBLAS_WORKSPACE_CONFIG, plus the run-to-run tolerance measured over
                        repeated identical FP32 and BF16 passes, which is what 01 §30
                        actually requires ("document the exact source and measure run-to-run
                        tolerance").
environment_lock_sha256 no longer special-cased. It resolves only once every component is
                        present, and publication re-derives it from the finalised manifest
                        and refuses to publish unless it matches.
```

### Transactional publication

The failed run had written `manifests/environments/TBD_REQUIRES_HARDWARE.json`. That is now
impossible: the manifest is built in memory, schema-validated, checked for any unresolved
field, given an identity that must be 64 lowercase hex, and re-derived for equality — only
then is it written to a temporary file in the target directory and `os.replace`d into
`<identity>.json`. A failure writes nothing and leaves no temporary file; a stale
`TBD_REQUIRES_HARDWARE.json` from the old implementation is deleted on the next success. The
filename is also now a forbidden path in the invariant checker.

### What was deliberately NOT done

The real hardware values you reported were used to drive and test the implementation. They
were **not** written into an environment manifest. Fabricating a manifest from reported
values is precisely the caller-supplied-identity failure mode the capture is built to refuse;
the manifest must be produced by capture running on the pod. `ENVIRONMENT_LOCK_SHA256`
therefore remains unresolved in this repository and `P0_PRE_READY` stays false.

The capture body moved from inline bash into `scripts/capture_environment.py` so it can be
exercised against mocked H100 hardware; `capture_environment.sh` is now a wrapper and
`make env-capture` is unchanged. The image-identity anti-forgery behaviour, uv.lock hashing,
nvidia-smi capture, GPU/driver capture and fail-closed semantics were ported unchanged and
are re-tested.


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
branch             stage/s00   (main and experiment-frozen do not exist yet; they are created
                                at acceptance and are protected [AUTH: 01 §27])
candidate commit   22e6b067c185e22753926327d5d238c25ce121d0
                   "S00-B: capture the real H100 environment"
history            f7e6ee0 bootstrap
                   aaac605 acceptance bundle + AI-stack provenance
                   e481543 report / manifest refresh
                   27ab476 final implementation-fix pass (FIX 1-9)
                   4d0bdaf bundle regenerated for 27ab476
                   ecaaaf6 S00-B science image specification
                   9a1bcde S00-B linux/amd64 build repair
                   d4dac4f S00-B host-venv contamination repair
                   22e6b06 S00-B environment-capture repair      <- candidate
bundle commit      HEAD, carrying only stage_acceptance/S00/ and reviews/S00/ artifacts
core.hooksPath     .githooks
working tree       clean; production_tree_dirty() reports []
```

The bundle describes the candidate commit exactly. A committed file cannot contain the SHA256
of the commit that contains it, so `make bundle-verify` proves the equivalent property
mechanically: it fails if any path outside `stage_acceptance/S00/` or `reviews/S00/` differs
between the described commit and HEAD.

Nothing was pushed. No tag was created and S00 is not merged; tagging and merge are
acceptance actions [AUTH: 01 §27].

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
[x] S00-B image specification frozen: python 3.13, torch 2.13.0+cu130, digest-pinned
[x] S00-B build and bootstrap workflow scripted, fail-closed, no Docker-in-Docker
[ ] S00-B image BUILT, pushed and run on the H100    <- next action, off-pod build
[ ] ENVIRONMENT_LOCK_SHA256 resolved                 <- blocks acceptance [plan §19]
[ ] repo reproducible from the frozen scientific environment  <- follows the above
```

Stage state:

```text
BACKEND_INTEGRATED = FALSE
SUITE_SCOPE        = STATISTICAL_STACK_ONLY
P0_PRE_READY       = FALSE
S00_ENV_CAPTURE    = TBD_REQUIRES_HARDWARE
```

S01 was not started [AUTH: 01 §41].
