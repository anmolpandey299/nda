# STAGE_PLAN.md — S00 Repository Bootstrap

**Stage:** `S00`
**Role:** ARCHITECT — READ / SEARCH / PLAN. No production edit. No commit. [AUTH: 01 §3.1]
**Date:** 2026-08-20
**Deliverable of this session:** this file only [AUTH: 01 §3.1]
**Plan-length cap:** 1,200 lines [AUTH: 03 §13]
**Review round:** 2 of 2 — patched to close accepted blind-review findings [AUTH: 03 §12]

---

## 0. Citation shorthand

| Tag | Document |
|---|---|
| `00` | `00_MEASUREMENT_SPEC_v1.9_FINAL_CLOSED.md` |
| `01` | `01_EXECUTION_STACK_LOCK_v2.md` |
| `02` | `02_PRE_EXECUTION_IMPLEMENTATION_CORRECTIONS.md` |
| `03` | `03_REVIEW_GOVERNANCE_LOCK.md` |

Every directory, module, command, manifest, workflow, test lane and review artifact below
carries `[AUTH: <doc> §<section>]`. Anything without authority is excluded, not softened
[AUTH: 03 §4, §7].

Where authority mandates a *thing* but not its *path or name*, the path is listed in the
Design-Choice Register (§4) rather than presented as a requirement [AUTH: 03 §14].

---

## 1. Controlling authority and reconciliations

### 1.1 Controlling sections for S00

```text
01 §11   repository authority (tree)
01 §12   environment lock (9-step procedure)
01 §22   test hierarchy
01 §28   CLAUDE.md content
01 §33   code-quality gate
01 §39   stage build order — S00 build list
02 §C1   primary estimand M_PRIMARY = common-FPR TPR@1%FPR
02 §C2   tie-safe distinct-threshold ROC; one tested estimator function
02 §C4   DRY-G family-wise calibration, N_FWER_CALIBRATION = 2_000
02 §C5   positive-control power recalibration after the estimator fix
02 §C6   backend readiness state
02 §C7   cache identity / SCORING_CODE_HASH
02 §C8   scaffold is a reference fixture, not production
03 §4    traceability
03 §8    hardware-unknown rule
03 §9    read-only enforcement
03 §13   plan-length cap
03 §14   requirement-source rule
```

S00's build list is fixed and closed [AUTH: 01 §39 S00]:

```text
repo structure
CLAUDE.md
pyproject
uv lock
Dockerfile
Makefile
spec copies
git hooks
CI skeleton
No research model run.
```

Nothing outside that list is built in S00.

### 1.2 Reconciliations the architect must declare

These are conflicts *between* authority documents. They are resolved here explicitly so no
reviewer has to guess which reading was silently adopted.

**R1 — spec filenames.**
`01 §11` (tree), `01 §0` (header) and `01 §39 S12` name `00_MEASUREMENT_SPEC_v1.8_FINAL.md`
and `01_EXECUTION_STACK_LOCK.md`. The actual controlling documents are v1.9 FINAL CLOSED and
v2.0 [AUTH: 02 header; 00 §48]. Resolution: `specs/` holds the four current filenames verbatim;
the §11 tree entries are treated as stale labels for the same slots. No content is altered.

**R2 — Codex review at S00.**
`01 §4.2` and `01 §40` do not list S00 as requiring independent Codex review. `03` is binding
"from S00 onward" and requires blind Claude **and** blind Codex for every major stage
[AUTH: 03 header, §2]. Resolution: apply the higher requirement, by analogy to the severity rule
in 03 §11. S00 receives both blind reviews plus mechanical adjudication [AUTH: 03 §2, §11].

**R3 — two verdict vocabularies.**
`03 §11` governs *review closure*: `CLEARED | BLOCKED | INCOMPLETE`.
`01 §5` / `01 §46` govern *stage acceptance*: `ACCEPT | ACCEPT_WITH_NONBLOCKING_NOTES | REJECT`.
Both are recorded; neither substitutes for the other.

**R4 — bundle content vs bundle layout.**
`01 §26` fixes 20 required bundle *fields*; `01 §45` fixes a 10-file *layout*. Resolution: apply
R2's higher-requirement rule — `01 §45`'s layout is the file structure and **all 20 `01 §26`
fields must be present within it**, mapped in §8.3. Items 16–17 name reviewer products
inconsistent with `01 §4`, `01 §5` and `03 §2`; they are slots for the mandated independent
reviewer report(s), i.e. `08_CODEX_REVIEW.md`. No extra reviewer or tool is added.

**R5 — `src` as import root.**
`01 §18` states the CLI shape literally as `python -m src.cli.train`. Adopted verbatim, so
`src/` is itself the top-level import package. This is authority-following, not a style choice.

**R6 — S00 spans one non-hardware step and one hardware step.**
`01 §39 S00` requires `uv lock` and `Dockerfile` inside S00, while `01 §12` requires the lock and
image to be produced *after* installing on the actual H100 image and running smoke tests.
Resolution: S00 is executed as **S00-A** (no hardware) and **S00-B** (H100 environment capture).
S00-A alone is reviewable and its acceptance tests need no H100 and no model download.
S00 stage acceptance requires both [AUTH: 01 §12, §39 S00, §46]. See §5 and §14.

**R7 — DRY-G calibration bank.**
`00 §42`'s checklist line and `00 §34B.1A`'s 200-trial tolerance bank predate `02 §C4`, which
replaces marginal per-statistic null gates with a single family-wise max-statistic gate at
`N_FWER_CALIBRATION = 2_000`. Resolution: **02 §C4 governs the production DRY-G gate**; the
200-trial bank is historical and superseded for that purpose and is not the production FWER gate.
DRY-B / DRY-D1 positive-control tolerances are regenerated under `02 §C5` after the common-FPR
estimator is fixed, not preserved because they previously passed [AUTH: 02 §C4, §C5; 00 §34B.1A,
§42].

---

## 2. Bounded objective and non-goals

### 2.1 Bounded objective

Create the reproducible repository shell that every later stage is required to write into, such
that from S01 onward it is *structurally impossible* to produce a scientific result without
provenance, to run two scoring/cross-fit paths, to let a notebook become the execution path, to
keep a cache valid across a scoring/backend change, or to let a green synthetic suite be read as
backend readiness.

S00 produces no scientific code, no scientific value, and no research-model execution
[AUTH: 01 §39 S00].

### 2.2 Explicit non-goals (excluded from S00)

| Excluded | Owning stage | Authority |
|---|---|---|
| Config resolver, SHA256 utils, manifest writers, RUN_ID, artifact hashing, immutable run dir | S01 | 01 §39 S01 |
| Tiny deterministic Llama fixture, tiny corpus, tiny LoRA, merge fixtures | S02 | 01 §39 S02; 01 §20 |
| Four-family model compatibility gate | S02B | 01 §8C; 01 §50 |
| Parameterised scorer, reference loss, Min-K%, content-addressed cache, cross-fit controller, pooled transforms | S03 | 01 §39 S03; 00 §34A |
| Statistical engine, eligibility, R_priv/R_func, D/G, isotonic, bootstrap, e50, tails, DRY-A…H | S04 | 01 §39 S04; 00 §34B |
| Data acquisition, dedup, splits, canaries, blind-split check | S05 | 01 §39 S05; 00 §6, §7 |
| Training, LoRA rank 32, seeds, DP plumbing | S06 | 01 §39 S06; 01 §8B; 00 §8 |
| Merge operators O1/O2/O3 | S07 | 01 §39 S07; 00 §10 |
| Recovery C1/C2/C2B/O3, conditioning | S08 | 01 §39 S08; 00 §25, §26 |
| Experiment registry / orchestrator / launcher | S09 | 01 §39 S09; 01 §35 |
| 1,000-sequence benchmark, throughput, batch size, VRAM, GPU-hour projections | S10 | 01 §39 S10; 00 §0.2.4 |
| Any real measurement, any result inspection | S11+ | 01 §29, §39 S11; 00 §35 |
| TIES, k=8, epsilon sweep, second architecture, full grid | not authorized | 00 §47 |
| Serving engines (vLLM / TensorRT-LLM / custom) | only if benchmark forces it | 01 §32 |
| Autonomous multi-agent research platform | forbidden | 01 §35 |
| Self-hosted coding agent / LLM gateway on H100 | forbidden | 01 §49 |
| Coverage percentage threshold | forbidden as a criterion | 01 §33 |
| OpenRouter in any engineering path | forbidden | 01 §6.2, §24 |

---

## 3. Repository tree

Minimal structure necessary to satisfy authority. Each line carries its authority.

```text
privacy-model-merging/                                   [AUTH: 01 §11]
│
├── CLAUDE.md                                            [AUTH: 01 §11, §28, §39 S00]
├── README.md                                            [AUTH: 01 §11]
├── Makefile                                             [AUTH: 01 §11, §33, §39 S00]
├── pyproject.toml                                       [AUTH: 01 §11, §12, §39 S00]
├── uv.lock                                              [AUTH: 01 §11, §12(5), §39 S00]   (CPU/dev at S00-A; relocked S00-B)
├── Dockerfile                                           [AUTH: 01 §11, §12(8), §39 S00]
├── .gitignore                                           [AUTH: 01 §11]
│
├── .githooks/                                           [AUTH: 01 §39 S00 "git hooks"; path D3]
│   ├── pre-commit
│   └── pre-push
│
├── .github/workflows/ci.yml                             [AUTH: 01 §39 S00 "CI skeleton"; path D3]
│
├── specs/                                               [AUTH: 01 §11, §39 S00 "spec copies"]
│   ├── 00_MEASUREMENT_SPEC_v1.9_FINAL_CLOSED.md         [AUTH: 01 §11; R1]
│   ├── 01_EXECUTION_STACK_LOCK_v2.md                    [AUTH: 01 §11; R1]
│   ├── 02_PRE_EXECUTION_IMPLEMENTATION_CORRECTIONS.md   [AUTH: 02 header; 03 §4]
│   ├── 03_REVIEW_GOVERNANCE_LOCK.md                     [AUTH: 03 header; 03 §4]
│   ├── SPEC_HASHES.json                                 [AUTH: 01 §15]
│   └── deviations/                                      [AUTH: 00 §0, §34A.5, §42; 01 §29]
│       ├── SPEC_DEVIATIONS.md                           [AUTH: 00 §0; 01 §29; 00 §42]
│       └── SCORER_EXCEPTIONS.md                         [AUTH: 00 §34A.5]
│
├── configs/                                             [AUTH: 01 §11, §17]
│   ├── models/  data/  training/  attacks/  recovery/  p0/  p1/     [AUTH: 01 §11]
│
├── src/                                                 [AUTH: 01 §11; R5]
│   ├── data/        (S05)                               [AUTH: 01 §11, §39 S05]
│   ├── models/      (S02, S06)                          [AUTH: 01 §11, §39 S02/S06]
│   ├── training/    (S06)                               [AUTH: 01 §11, §39 S06]
│   ├── dp/          (S06)                               [AUTH: 01 §11, §39 S06; 00 §8.2]
│   ├── merge/       (S07)                               [AUTH: 01 §11, §39 S07]
│   ├── recovery/    (S08)                               [AUTH: 01 §11, §39 S08]
│   ├── scoring/     (S03) — sole owner of scorer, cross-fit, cache   [AUTH: 01 §11; 00 §34A.1–.4]
│   ├── attacks/     (S03/S04)                           [AUTH: 01 §11; 00 §14, §15]
│   ├── analysis/    (S04)                               [AUTH: 01 §11, §34; 00 §29]
│   ├── provenance/  (S01)                               [AUTH: 01 §11, §13–§16; 02 §C7]
│   └── cli/         (S01+)                              [AUTH: 01 §11, §18]
│
├── tests/                                               [AUTH: 01 §11, §22]
│   ├── unit/                                            [AUTH: 01 §11, §22 Unit]
│   ├── integration/                                     [AUTH: 01 §11, §22 Integration]
│   ├── synthetic/                                       [AUTH: 01 §11, §22; 00 §34B]
│   ├── golden/                                          [AUTH: 01 §11, §22]
│   ├── backend_contract/                                [AUTH: 02 §C6]
│   └── gpu_smoke/                                       [AUTH: 01 §11, §21, §22]
│
├── manifests/                                           [AUTH: 01 §11]
│   ├── models/<model_alias>.json                        [AUTH: 01 §13; 01 §8G]
│   ├── data/<dataset_alias>.json                        [AUTH: 01 §14]
│   ├── environments/<env_id>.json                       [AUTH: 01 §12, §15, §16]
│   ├── environments/AI_ENGINEERING_STACK_S00.json       [AUTH: 01 §2.1, §37, §48]
│   └── runs/<RUN_ID>.json                               [AUTH: 01 §15, §16]
│
├── results/                                             [AUTH: 01 §11, §19]
│   ├── p0/                                              [AUTH: 01 §11; 00 §43]
│   └── p1/                                              [AUTH: 01 §11; 00 §44]
│
├── artifacts/                                           [AUTH: 01 §11]
│   ├── p0_pre/                                          [AUTH: 00 §34B.2, §34C.1; 02 §C3, §C4, §C6]
│   ├── runs/<RUN_ID>/                                   [AUTH: 01 §39 S01 "immutable run directory"; 01 §36]
│   └── cache/                                           [AUTH: 00 §34A.4; 02 §C7]
│
├── reviews/                                             [AUTH: 01 §11]
│   └── <stage>/                                         [AUTH: 03 §9, §10]
│       ├── CLAUDE_REVIEW.md                             [AUTH: 01 §3.3; 03 §3]
│       ├── CODEX_REVIEW.md                              [AUTH: 01 §44; 03 §3]
│       ├── RECONCILIATION.md                            [AUTH: 03 §3 Phase 2, §15]
│       ├── ARCHITECT_FINDING_RESPONSES.md               [AUTH: 03 §10]
│       ├── ADJUDICATION.md                              [AUTH: 03 §11]
│       ├── deferred.md                                  [AUTH: 03 §10]
│       └── scratch/                                     [AUTH: 03 §9]
│
├── stage_acceptance/<stage>/                            [AUTH: 01 §26, §45; path D3]
│   ├── 00_INDEX.md  01_PLAN.md  02_DIFF.patch  03_TEST_COMMANDS.txt
│   ├── 03A_RESOLVED_CONFIGS/                            [AUTH: 01 §26(8)]
│   ├── 04_TEST_OUTPUTS/  05_ARTIFACT_MANIFEST.json  06_IMPLEMENTER_REPORT.md
│   ├── 07_CLAUDE_REVIEW.md  08_CODEX_REVIEW.md  09_UNRESOLVED.md  10_REPRODUCE.md
│   └── 11_DATA_INSPECTION_STATEMENT.md                  [AUTH: 01 §26(19)]
│       01_PLAN.md = canonical stage-plan path (D10)     [AUTH: 01 §3.1, §45; 03 §13]
│
├── logs/<RUN_ID>/                                       [AUTH: 01 §11, §16]
│
└── scripts/                                             [AUTH: 01 §11]
    ├── check_repo_invariants.py                         [AUTH: 01 §15, §18; 03 §9, §13]
    ├── capture_environment.sh                           [AUTH: 01 §12(2)–(9)]
    └── install_git_hooks.sh                             [AUTH: 01 §39 S00 "git hooks"]
```

### 3.1 Deliberately absent

| Absent | Why |
|---|---|
| `notebooks/` | No notebook may be an authoritative execution path [AUTH: 01 §18]. Exploration notebooks, if ever used, live outside authoritative source paths and outside `src/`, and may never appear in a config or Makefile target. |
| `serving/` | Not authorized before the benchmark shows direct batched scoring is insufficient [AUTH: 01 §32]. |
| second scorer / second cross-fit path | Prohibited [AUTH: 00 §34A.2, §34A.5]. |
| coverage gate | Coverage percentage is explicitly not a correctness criterion [AUTH: 01 §33]. |
| orchestration platform, agent runners | Prohibited [AUTH: 01 §35, §49]. |
| any OpenRouter integration | Prohibited in the engineering path [AUTH: 01 §6.2, §24]. |

Empty stage-owned packages are created as `__init__.py` only, with a one-line docstring naming
the owning stage. That is structure, not implementation [AUTH: 01 §11, §39].

---

## 4. Design-Choice Register (03 §14 disclosure)

Items where authority mandates the substance but not the path/name/format. Each is a design
choice made by this plan, disclosed so it is not mistaken for a requirement.

| ID | Choice | Substance mandated by | Note |
|---|---|---|---|
| D1 | `artifacts/` sub-structure `p0_pre/`, `runs/`, `cache/` | 01 §11 (dir), 00 §34B.2, 01 §39 S01, 00 §34A.4 | §11 leaves `artifacts/` unstructured |
| D2 | `specs/deviations/` register directory | 00 §0, 00 §34A.5, 01 §29, 00 §42 | Registers must be findable to be reviewable |
| D3 | Paths `stage_acceptance/<stage>/`, `.githooks/`, `.github/workflows/ci.yml` | 01 §26, §45, §39 S00 | Artifacts mandated; locations chosen |
| D4 | `specs/SPEC_HASHES.json` filename | 01 §15 (spec + lock SHA256 are RUN_ID inputs) | Colocated with the copies it hashes |
| D5 | `make format CHECK=1` verify mode | 01 §33 (`format = PASS` must be checkable) | One target, two modes; avoids a second command |
| D6 | Typechecker = `mypy` | 01 §33 ("mypy or pyright") | Chosen from the authorized set; `pyright` remains permitted if `mypy` fails to resolve in the frozen env |
| D7 | Lane report label `NOT_RUN(<reason>)`, reason ∈ {`EMPTY_AT_S00`, `BACKEND_NOT_INTEGRATED`, `NO_GPU`} | 02 §C6, 00 §34B.3 | Reporting labels only; not project states (§12) |
| D8 | `RESERVED_SINGLE_OWNER` API reservation + import-boundary check in the invariant checker | 00 §34A.2(8), §34A.5; 02 §C1, §C2 | Mechanism for a mandated prohibition; see §13.1 |
| D9 | CLAUDE.md enforced length cap (default 150 lines) | 01 §28 ("short", must not duplicate the spec) | Cap value is a choice; the shortness requirement is not |
| D10 | Canonical stage-plan path `stage_acceptance/<stage>/01_PLAN.md` | 01 §3.1 (`STAGE_PLAN.md`), 01 §45 (`01_PLAN.md` slot), 03 §13 (cap must be checkable) | One file, no copy, so plan and bundle cannot diverge; makes I5 globbable |

### 4.1 PROMPT_ONLY_REQUIREMENT audit [AUTH: 03 §14]

Every element requested by the task prompt, with its authority or its disposition.

| Prompt item | Authority | Disposition |
|---|---|---|
| Repository tree | 01 §11 | Included |
| `pyproject.toml`, `uv.lock`, `Dockerfile` | 01 §12, §39 S00 | Included |
| Python selection | 01 §12 | Included, value TBD (§5) |
| CUDA/PyTorch compatibility test | 01 §12(3), §8C(3,4,12), §21 | Included, values TBD (§5) |
| HF/PEFT compatibility | 01 §12, §8C(6–8); 02 §C6, §C7 | Version freeze in S00; functional gate is S02B |
| CI | 01 §39 S00 | Included |
| H100 environment capture | 01 §12(2)–(9), §9, §16 | Included as S00-B |
| `CLAUDE.md` | 01 §28, §39 S00 | Included |
| Model / data / environment / run manifest | 01 §13, §14, §12+§15, §15+§16 | Included |
| `SCORING_CODE_HASH` | 02 §C7 | Path + field reserved; computed in S03 |
| resolved-config hash; artifact hash; `RUN_ID` | 01 §17, §15, §19, §16, §26(13), §45 | Paths + fields reserved and input list frozen; computed in S01 |
| `reviews/<stage>/`, `.../scratch/` | 01 §11; 03 §9 | Included |
| `stage_acceptance/<stage>/` | 01 §26, §45 | Included; path is D3 |
| Lanes unit/integration/synthetic/golden/gpu_smoke; lane `backend_contract` | 01 §11, §22; 02 §C6 | Included |
| `make format/lint/typecheck/unit/integration/synthetic/gpu-smoke`; `make backend-contract` | 01 §33, §22, §21; 02 §C6 | Included |
| `make preflight` | 01 §33 (ordered gate), 01 §22 (order), 02 §C6 (state) | Included as the gate's invocation surface |
| `BACKEND_INTEGRATED` / `SUITE_SCOPE` / `P0_PRE_READY` initial FALSE | 02 §C6 | Included verbatim |
| Failure states | 01 §36; 03 §11; 00 §0, §34A.5, §34C.1 | Included; enumerated in §12 |
| S00 acceptance tests without H100/model | 01 §20, §39 S00; 03 §8 | Included as §14 |
| Non-goals; 1,200-line cap | 01 §39, §41; 00 §47; 03 §13 | §2.2; enforced in CI via I5 |

No prompt-only requirement is promoted to authority. No prompt item lacked authority, so
nothing had to be excluded on that ground.

---

## 5. Environment strategy

### 5.1 Procedure (the nine steps are authority, verbatim in order)

[AUTH: 01 §12]

```text
1. Claude proposes one mutually compatible environment.     -> S00-A
2. Install on the actual H100 image.                        -> S00-B
3. Run compatibility smoke tests.                           -> S00-B
4. Freeze exact versions.                                   -> S00-B
5. Hash uv.lock.                                            -> S00-B
6. Record nvidia-smi.                                       -> S00-B
7. Record CUDA driver/runtime.                              -> S00-B
8. Build/tag Docker image.                                  -> S00-B
9. Record Docker image digest.                              -> S00-B
```

`No pip install -U occurs during an experiment run` [AUTH: 01 §12]. The Dockerfile installs
strictly from the frozen `uv.lock`; the image is the execution surface. S00-A's Dockerfile builds
the CPU/dev lane from the S00-A lock; S00-B pins the CUDA base image and rebuilds from the
relocked file.

### 5.2 `pyproject.toml` — two-phase content

**S00-A commits:** project metadata; `[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]`
(lane markers, strict marker enforcement); packaging discovery restricted to `src/`
[AUTH: 01 §33; 03 §9]; **`requires-python` frozen**; and a **committed `uv.lock` covering the
CPU/dev set** — ruff, mypy, pytest, NumPy, SciPy, scikit-learn, pandas, safetensors, tokenizers,
plotting — so `make lint` / `make typecheck` / `make unit` are reproducible on a clean machine
from the commit alone [AUTH: 01 §33 "exact versions are frozen in the environment lock", §46].
None of those requires an H100, so 03 §8 does not license deferring them. Only the CUDA-coupled
set is left unresolved: PyTorch (+ CUDA build), CUDA runtime/driver, Transformers, PEFT,
Accelerate, DP library [AUTH: 03 §8].

**S00-B commits:** the CUDA-coupled set pinned to what actually installed on the H100 image, the
same single `uv.lock` relocked to cover both sets, and the resolved interpreter verified against
the frozen `requires-python` [AUTH: 01 §11 (one lock), §12(2)–(5)]. If the H100 resolution moves
any S00-A pin, the move is recorded in the environment manifest and the S00-B lock is
authoritative for scientific execution (F4 unchanged).

Dependency classes frozen by the environment lock [AUTH: 01 §12]:

```text
Python
PyTorch
CUDA runtime
Transformers
PEFT
Accelerate
DP library
NumPy
SciPy
scikit-learn
pandas
safetensors
tokenizers
plotting libraries
test/lint/typecheck tooling
```

Result-format dependencies follow from 01 §19 (Parquet/JSON/JSONL authoritative; CSV only where
00 explicitly requests it, e.g. the 00 §43 `.csv` artifact names).

### 5.3 Python selection rule

No version number is written into this plan [AUTH: 03 §8].

Selection rule, applied in S00-B on the actual image [AUTH: 01 §12(1)–(4)]:

1. Determine the PyTorch build officially supporting H100 SXM (compute capability 9.0) against
   the CUDA runtime present in the RunPod image [AUTH: 01 §9].
2. Choose the newest CPython simultaneously supported by that PyTorch wheel and by every other
   frozen dependency class in §5.2.
3. If no single interpreter satisfies all classes, this is an `actual library incompatibility`,
   one of the few permitted reasons to amend the execution lock [AUTH: 01 §47]. Record it; do
   not substitute versions silently.

### 5.4 CUDA / PyTorch compatibility test

Executed in S00-B on the H100 image [AUTH: 01 §12(3); 01 §8C(3),(4),(12)].

| Check | Authority |
|---|---|
| CUDA visible; device is H100 SXM 80GB; compute capability 9.0 | 01 §9 |
| BF16 text-forward path executes with no quantization | 01 §8C(3),(4); 01 §10 Training |
| FP32 path available for merge/recovery arithmetic | 01 §10 Merge/recovery |
| float64 available for derived scalar statistics | 01 §10 Metrics |
| AMP/autocast cannot silently determine scientific arithmetic (explicit dtype assertions) | 01 §10 |
| Run-to-run determinism measured; nondeterministic kernel sources documented with measured tolerance | 01 §30 |

Recorded in `manifests/environments/<env_id>.json`. Accepted by
`tests/gpu_smoke/test_cuda_torch_contract.py` [AUTH: 01 §21, §22 GPU smoke].

### 5.5 HF / PEFT compatibility — scope split

S00 freezes **versions**, not behaviour.

| In S00-B | Authority |
|---|---|
| Transformers / PEFT / Accelerate resolve and install on the frozen stack | 01 §12 |
| Import-and-version smoke; versions written to the environment manifest; **no model download** | 01 §12(3); 01 §20 |

| Deferred, not S00 | Owning stage | Authority |
|---|---|---|
| LoRA rank-32 MLP-only attach on each core family | S02B | 01 §8B, §8C(6) |
| LoRA save/load byte/provenance stability | S02B | 01 §8C(7) |
| Exact ΔW = BA extraction | S02B | 01 §8C(8) |
| Linear merge construction; P0-C1 exact inversion tolerance | S02B/S07/S08 | 01 §8C(9),(10) |
| Deterministic scorer output within frozen numerical tolerance | S03 | 01 §8C(11) |
| DP smoke for DP-arm models | S06 | 01 §8C(13), §8E |

Pulling any of those into S00 would be `UNJUSTIFIED_SCOPE` [AUTH: 03 §7].

### 5.6 H100 environment capture

`scripts/capture_environment.sh`, run on the image, writes
`manifests/environments/<env_id>.json` [AUTH: 01 §12(5)–(9), §15, §16].

Required fields: `python_version`, `torch_version`, `torch_cuda_build`, `cuda_runtime`,
`cuda_driver`, `nvidia_smi_capture`, `gpu_model`, `gpu_uuid`, `gpu_count`, `uv_lock_sha256`,
`docker_image_tag`, `docker_image_digest`, dependency-version map, `bf16_fp32_tolerance`,
`nondeterminism_sources`, `capture_timestamp_utc`.

**`ENVIRONMENT_LOCK_SHA256`** is the identity of the *complete frozen execution environment*,
never of the lockfile alone:

```text
ENVIRONMENT_LOCK_SHA256 = SHA256( uv_lock_sha256, docker_image_digest,
                                  cuda_runtime, cuda_driver,
                                  torch_version, torch_cuda_build,
                                  python_version, gpu_model )
```

`env_id = ENVIRONMENT_LOCK_SHA256`. It is the value consumed as `RUN_ID`'s `environment lock
SHA256` input and as the environment component of the §7.1 cache key
[AUTH: 01 §12(5)–(9), §15, §16, §30, §32].

Consequence: two images sharing a `uv.lock` but differing in image digest, CUDA runtime or driver
yield different `ENVIRONMENT_LOCK_SHA256`, hence different `RUN_ID`s, so `artifacts/runs/<RUN_ID>/`
cannot collide across runtimes [AUTH: 01 §36], and a runtime/serving change forces a new
environment hash [AUTH: 01 §32].

### 5.7 TBD_REQUIRES_HARDWARE register

No hardware-dependent value is invented [AUTH: 03 §8]. Each entry states how it is measured,
where it is recorded, and what test accepts it. Only genuinely H100/CUDA-dependent values appear
here: `requires-python` and the CPU/dev tool and statistics set are frozen in the S00-A `uv.lock`
(§5.2) and are deliberately absent from this register [AUTH: 01 §33, §46; 03 §8].

| Value | How measured | Recorded in | Accepted by | Owner |
|---|---|---|---|---|
| Exact CUDA runtime version | `capture_environment.sh` on the image | `manifests/environments/<env_id>.json` | `tests/gpu_smoke/test_env_capture_contract.py` | S00-B |
| CUDA driver version | same | same | same | S00-B |
| PyTorch version + CUDA build variant | resolver output on the image | same | same | S00-B |
| Transformers / PEFT / Accelerate versions (CUDA-coupled to the torch build) | install on the image | same | same | S00-B |
| DP library identity + version | must supply per-sample gradients with PEFT LoRA on the frozen stack | same | DP smoke, S06 | S00-B select / S06 accept |
| `ENVIRONMENT_LOCK_SHA256` | §5.6 composite over lock, image digest, CUDA runtime/driver, torch, Python, GPU | `manifests/environments/<env_id>.json` | `tests/gpu_smoke/test_env_capture_contract.py` | S00-B |
| `uv.lock` SHA256 | `sha256sum uv.lock` after step 4 | `manifests/environments/<env_id>.json` | `tests/unit/test_env_manifest_schema.py` (schema) + gpu-smoke contract (values) | S00-B |
| `nvidia-smi` capture | `nvidia-smi` verbatim | same | gpu-smoke contract | S00-B |
| GPU model / UUID / count | `nvidia-smi` + torch device query | same | gpu-smoke contract | S00-B |
| Docker image tag + digest | build/tag, then read digest | same | gpu-smoke contract | S00-B |
| BF16↔FP32 scoring tolerance | measured on the image, fixed sequences | same | gpu-smoke contract | S00-B |
| CUDA nondeterminism source + run-to-run tolerance | repeated identical forward passes | same | gpu-smoke contract | S00-B |
| Batch size; throughput q (seq/s, tokens/s); peak VRAM; projected P0 / P1 GPU-hours | 1,000-sequence benchmark + instrumentation; q applied to the planned envelope | `artifacts/p0_pre/P0_00_COMPUTE_BUDGET.json` | S10 benchmark gate | S10 |

[AUTH for the S10 row: 00 §0.2.4, §0.3.3; 00 §34C.1; 01 §39 S10]

Until S00-B completes, every field above holds the literal string `TBD_REQUIRES_HARDWARE`, and
`S00_ENV_CAPTURE = TBD_REQUIRES_HARDWARE`.

**Consequence, stated as architecture:** `RUN_ID` requires `environment lock SHA256`
[AUTH: 01 §15]. With no environment lock there is no valid `RUN_ID`, hence no valid run manifest,
hence any result produced is `NON_EVIDENTIARY` [AUTH: 01 §16]. S01–S04 are CPU-only development
stages and may proceed before S00-B, but no evidentiary run can start until S00-B closes.

---

## 6. CLAUDE.md

Short and operational; must not duplicate the 6,000-line spec [AUTH: 01 §28]. Enforced cap
D9. Exactly the §28 sections, in order:

```text
1. Project purpose                    [AUTH: 01 §28; 00 §1]
2. Controlling documents              [AUTH: 01 §28] — the four specs/ files + SPEC_HASHES.json
3. Non-negotiable scientific invariants  [AUTH: 01 §28] — pointers only, listed in §6.1
4. Repo commands                      [AUTH: 01 §28, §33] — the §10 table
5. Test commands                      [AUTH: 01 §28, §22]
6. Stage workflow                     [AUTH: 01 §28, §39, §42]
7. Review workflow                    [AUTH: 01 §28, §40; 03 §2, §10, §11, §12]
8. Forbidden actions                  [AUTH: 01 §28] — listed in §6.2
9. Artifact locations                 [AUTH: 01 §28] — the §7 provenance table
```

Plus the mandatory instruction, verbatim in spirit [AUTH: 01 §28]:

> Read the relevant spec sections for the current task. Never summarize the entire spec into
> memory and then work from the summary alone.

### 6.1 Invariant pointers carried in CLAUDE.md

Pointers, not restatements [AUTH: 01 §23; 01 §28]:

```text
one scorer, one cross-fit implementation           00 §34A.1, §34A.2
pooled view never invokes the model                00 §34A.3; 01 §23
cache key includes SCORING_CODE_HASH               02 §C7; 00 §34A.4
no calibration/evaluation overlap                  01 §23; 00 §34B.1 (DRY-H3)
arm-matched thresholding                           01 §23; 00 §15.1
Min-K reduced inline; per-token arrays not persisted  00 §14.2; 01 §23
material constants live in config, not code        01 §17
no notebook is an execution path                   01 §18
no result without a run manifest                   01 §16
no result-peeking before component freeze          01 §29
seeds recorded separately, never one global seed   01 §30
statistics live in importable modules, not plots   01 §34
```

### 6.2 Forbidden actions carried in CLAUDE.md

```text
no production edit by a reviewer session                 01 §3.3; 03 §9
no commit to main / experiment-frozen                    01 §27
no drive-by refactor; one bounded stage per branch       01 §27, §41
no scientific-threshold change; no result-driven tuning  01 §3.2; 00 §34B.1A
no pip install -U during a run                           01 §12
no floating `main` model revision                        01 §8G
no OpenRouter in any engineering path                    01 §6.2, §24
no second scoring / cross-fit path                       00 §34A.2, §34A.5
no overwrite of a failed run directory                   01 §36
no rewrite of published experiment history               01 §27
```

---

## 7. Provenance paths

All paths are reserved at S00; the writers are implemented in S01 unless noted
[AUTH: 01 §39 S01].

| Object | Path | Authority |
|---|---|---|
| Model manifest | `manifests/models/<model_alias>.json` | 01 §13; 01 §8G |
| Data manifest | `manifests/data/<dataset_alias>.json` | 01 §14 |
| Environment manifest | `manifests/environments/<env_id>.json` | 01 §12, §15, §16 |
| AI engineering-stack record | `manifests/environments/AI_ENGINEERING_STACK_S00.json` | 01 §2.1, §37, §48 |
| Run manifest | `manifests/runs/<RUN_ID>.json` | 01 §15, §16 |
| Immutable run directory | `artifacts/runs/<RUN_ID>/` | 01 §39 S01; 01 §36 |
| Run logs | `logs/<RUN_ID>/{stdout.log,stderr.log}` | 01 §11, §16 |
| Result tables | `results/p0/`, `results/p1/` | 01 §11, §19; 00 §43, §44 |
| P0-PRE gate artifacts | `artifacts/p0_pre/` | 00 §34B.2, §34C.1; 02 §C3, §C4 |
| Scoring cache | `artifacts/cache/` (path configurable) | 00 §34A.4; 02 §C7; 01 §17 |
| Spec hashes | `specs/SPEC_HASHES.json` | 01 §15 |
| Deviation registers | `specs/deviations/` | 00 §0, §34A.5; 01 §29; 00 §42 |

### 7.1 Hash and identity fields

| Field | Definition | Written to | Authority |
|---|---|---|---|
| `RUN_ID` | SHA256 over exactly: git commit SHA, spec SHA256, execution-lock SHA256, config SHA256, model revision, data-manifest SHA256, `ENVIRONMENT_LOCK_SHA256` (§5.6), training seed | `manifests/runs/<RUN_ID>.json`, written **before** training/scoring begins | 01 §15 |
| `resolved_config_sha256` | SHA256 of the saved resolved config that makes the invocation reconstructable | run manifest + `artifacts/runs/<RUN_ID>/resolved_config.yaml` | 01 §17, §15, §19 |
| `artifact_sha256[]` | SHA256 per produced artifact | run manifest `artifact_hashes`; stage bundle `05_ARTIFACT_MANIFEST.json` | 01 §16, §26(13), §45 |
| `SCORING_CODE_HASH` | SHA256 over the scorer sources + HF/PEFT backend adapter + masking/reduction + reference-loss + Min-K implementations + resolved scoring config | run manifest; **and** as a component of every cache key | 02 §C7 |
| Result provenance keys | `run_id, artifact_id, seed, model_revision, config_hash, data_manifest_hash, code_commit` on every result table | `results/**` | 01 §19 |

**Frozen scope rule:** `RUN_ID` consumes exactly the eight inputs listed in 01 §15 and no
others. Defining §15's "environment lock SHA256" as the §5.6 composite widens what that one input
*covers*, not how many inputs there are; the count and the input list are unchanged
[AUTH: 01 §12, §15, §32]. `SPEC_HASHES.json` additionally records the SHA256 of documents `02` and `03` for
auditability; those hashes are **not** folded into `RUN_ID`, because §15's input list is
authority and widening it would silently change run identity [AUTH: 01 §15; 03 §14].

**Cache identity rule:** the content-addressed key contains model/artifact hash, record-set
hash, tokenizer hash, precision, max sequence length [AUTH: 00 §34A.4], `SCORING_CODE_HASH`
[AUTH: 02 §C7] **and `ENVIRONMENT_LOCK_SHA256`** [AUTH: 01 §12, §32]. A hand-typed
`scorer_version` string is not sufficient [AUTH: 02 §C7]. Every cache entry additionally records
the `env_id` and `RUN_ID` that produced it. Negative tests owned by S03 [AUTH: 02 §C7; 00 §34B.1
DRY-CACHE]: backend source change → miss; score-reducer change → miss; **environment change →
miss**; unchanged code, config and environment → hit with a bitwise-identical scalar score table.

---

## 8. Review layout

[AUTH: 01 §11; 03 §2, §3, §9, §10, §11, §12]

```text
reviews/<stage>/CLAUDE_REVIEW.md        blind Claude review              03 §3; 01 §3.3
reviews/<stage>/CODEX_REVIEW.md         blind Codex review              03 §3; 01 §44
reviews/<stage>/RECONCILIATION.md       Codex Phase-2 overlap answer    03 §3, §15
reviews/<stage>/ARCHITECT_RESPONSE.md   FIX / REJECT / DEFER per finding 03 §10
reviews/<stage>/ADJUDICATION.md         SXX_ADJUDICATION = ...           03 §11
reviews/<stage>/deferred.md             DEFER tickets                    03 §10
reviews/<stage>/scratch/                reviewer diagnostics only        03 §9
stage_acceptance/<stage>/               01 §45 ten-file bundle           01 §26, §45
```

### 8.1 Read-only enforcement and repository preconditions

[AUTH: 03 §9; 01 §25, §27]

**Precondition — real pinned worktree.** Before S00 implementation review begins, the repository
must be an initialised Git repository with a resolvable commit SHA, and each reviewer session
must be given a **separate worktree checked out at that pinned commit**, read-only on every path
except `reviews/<stage>/scratch/`. A review run against a directory where `git rev-parse HEAD`
fails cannot identify the reviewed revision, cannot supply 01 §26 items 4–5, and is not a valid
blind review [AUTH: 03 §9; 01 §25, §26(4)(5)].

**Enforcement layers, strongest first:**

1. **Protected branches and stage tags are the authority mechanism.** `main` and
   `experiment-frozen` are protected; every accepted stage is tagged; published experiment
   history is never rewritten [AUTH: 01 §27].
2. **Reviewer worktree** at a pinned commit, filesystem read-only outside `scratch/`
   [AUTH: 03 §9].
3. **Evidentiary-execution clean-state gate** (§13 I15) — the gate that actually protects run
   identity [AUTH: 01 §35(3), §16, §36].
4. **Git hooks are a convenience guard only.** `core.hooksPath` is per-clone local config and
   `--no-verify` bypasses it, so hooks are never the mechanism relied upon; preflight step 0
   warns when `core.hooksPath != .githooks` [AUTH: 03 §9 "enforced operationally, not only
   requested"].
5. `scripts/check_repo_invariants.py` asserts no importable code exists under `reviews/` and that
   packaging discovery in `pyproject.toml` is restricted to `src/` [AUTH: 03 §9].

### 8.3 `01 §26` field → `01 §45` slot mapping

Per R4, all twenty `01 §26` fields live inside the `01 §45` layout [AUTH: 01 §26, §45]:

```text
1-3   STAGE_ID / objective / controlling spec sections -> 00_INDEX.md
4-5   base + head git commit                           -> 00_INDEX.md
6-7   diff-patch + changed-file list                   -> 02_DIFF.patch
8     resolved config(s)                               -> 03A_RESOLVED_CONFIGS/ + sha256
9-12  unit / integration / synthetic / GPU-smoke logs  -> 04_TEST_OUTPUTS/
13    artifact manifest + SHA256                       -> 05_ARTIFACT_MANIFEST.json
14    implementer report                               -> 06_IMPLEMENTER_REPORT.md
15    Claude reviewer report                           -> 07_CLAUDE_REVIEW.md
16-17 independent reviewer report(s)                   -> 08_CODEX_REVIEW.md
18    unresolved findings                              -> 09_UNRESOLVED.md
19    explicit statement of whether real scientific
      data were inspected                              -> 11_DATA_INSPECTION_STATEMENT.md
20    requested verdict ACCEPT / REJECT                -> 00_INDEX.md, named field
      reproduction instructions                        -> 10_REPRODUCE.md
```

Item 19 is the only mechanical no-result-peeking attestation in the bundle, mandatory from S11
where peeking rules activate [AUTH: 01 §26(19), §29, §39 S11]; A18 fails a bundle missing any of
the twenty fields. `01_PLAN.md` is the canonical stage-plan path (D10): the architect writes
`stage_acceptance/<stage>/01_PLAN.md` directly, so the plan and the bundle cannot diverge and I5
can glob `stage_acceptance/*/01_PLAN.md` [AUTH: 01 §3.1, §45; 03 §13].

### 8.2 Blindness and loop bounds carried into the layout

Claude and Codex reviews are written independently and frozen before either is shown to the
other; Phase-2 reconciliation answers only the four overlap questions [AUTH: 03 §3].
Maximum two review rounds per stage [AUTH: 03 §12]. Finding budget: BLOCKER unlimited if real,
MAJOR ≤ 6, MINOR ≤ 4, NOTE ≤ 2 [AUTH: 03 §6]. A BLOCKER/MAJOR without a concrete counterexample
is automatically a NOTE [AUTH: 03 §5]. A BLOCKER cannot be deferred [AUTH: 03 §10].

---

## 9. Test lanes

Exactly six lanes. Each is justified; none is added for symmetry.

| Lane | Directory | What it may contain | Authority |
|---|---|---|---|
| unit | `tests/unit/` | pure functions and mathematical identities | 01 §11, §22 Unit |
| integration | `tests/integration/` | multiple modules with tiny fixtures, no network | 01 §11, §22 Integration; 01 §20 |
| synthetic | `tests/synthetic/` | planted-truth scenarios DRY-A…H, DRY-CACHE against frozen tolerances | 01 §11, §22; 00 §34B.1, §34B.1A |
| golden | `tests/golden/` | frozen expected outputs for fixed inputs | 01 §11, §22 Synthetic/golden |
| backend_contract | `tests/backend_contract/` | production HF/PEFT backend contract tests, runnable only after backend integration | 02 §C6 |
| gpu_smoke | `tests/gpu_smoke/` | real model/library/runtime compatibility on H100 | 01 §11, §21, §22 GPU smoke |

Execution order is authority [AUTH: 01 §22]:

```text
UNIT -> INTEGRATION -> SYNTHETIC/GOLDEN -> GPU SMOKE (where applicable) -> SCIENTIFIC GATE
```

`SCIENTIFIC GATE` is a real experiment, not a pytest lane, so it gets no directory
[AUTH: 01 §22].

### 9.1 Why `backend_contract` is a separate lane and not part of `integration`

02 §C6 makes backend integration a distinct gate whose outcome flips `BACKEND_INTEGRATED`, and
states that a green synthetic suite does not override that flag. If backend contract tests lived
inside `integration`, a green `integration` run would be indistinguishable from backend
readiness and the §C6 distinction would stop being mechanical. Separation is what makes the flag
computable rather than asserted [AUTH: 02 §C6].

### 9.2 Empty lanes must not read as PASS

A lane with no collected tests reports `NOT_RUN(EMPTY_AT_S00)`, never `PASS` (D7). Authority:
02 §C6 (green synthetic ≠ readiness) and 00 §34B.3 (a freeze checkbox "does not mean merely that
a script file exists"). At S00 the synthetic, golden, backend_contract and gpu_smoke lanes are
all `NOT_RUN`.

### 9.3 Lane → invariant ownership (reserved, not implemented in S00)

The 01 §23 critical-invariant list is assigned to lanes now so no invariant is orphaned later
[AUTH: 01 §23]:

```text
unit        exact C1 linear inversion; Min-K inline reduction; zero-delta anchor e=1; O3
            analytic floor; varying-alpha rescaling identity
integration same-observation R across compared recovery methods; run-manifest completeness;
            base-reference output cache consistency
synthetic   no evaluation IDs in calibration; no calibration/evaluation overlap; arm-matched
            thresholding; cache version invalidation; realised FPR reporting; no extrapolation
            beyond e support; rank-matched ladder tolerance
golden      pooled view never invokes model; deterministic scalar score tables
gpu_smoke   DP smoke cannot enter inferential tables
```

---

## 10. Command surface

Only traceable commands. Every target is non-interactive [AUTH: 01 §18].

| Command | Semantics | Exit contract | Authority |
|---|---|---|---|
| `make format` | apply `ruff format` | 0 on success | 01 §33 |
| `make format CHECK=1` | verify formatting, mutate nothing | non-zero if reformatting needed | 01 §33 (`format = PASS`); D5 |
| `make lint` | `ruff check` | non-zero on any finding | 01 §33 |
| `make typecheck` | `mypy` over typed core modules | non-zero on error | 01 §33; D6 |
| `make unit` | `pytest tests/unit` | non-zero on failure | 01 §22, §33 |
| `make integration` | `pytest tests/integration` | non-zero on failure | 01 §22, §33 |
| `make synthetic` | `pytest tests/synthetic tests/golden` — one command for the one §22 tier | non-zero on failure | 01 §22, §33; 00 §34B |
| `make backend-contract` | `pytest tests/backend_contract`; refuses to report PASS while `BACKEND_INTEGRATED=FALSE`, reporting `NOT_RUN(BACKEND_NOT_INTEGRATED)` | 0 with explicit NOT_RUN status at S00 | 02 §C6 |
| `make gpu-smoke` | `pytest tests/gpu_smoke`; reports `NOT_RUN(NO_GPU)` when no H100 is present | 0 with explicit NOT_RUN status off-hardware | 01 §21, §22, §33 |
| `make env-capture` | run `scripts/capture_environment.sh` (01 §12 steps 2–9) and write the environment manifest | non-zero if any required field is unresolved | 01 §12 |
| `make preflight` | the ordered §33 gate plus readiness evaluation — §10.1 | non-zero on any gate failure | 01 §33, §22; 02 §C6 |

No other Makefile target is defined at S00. `make env-capture` exceeds the prompt's minimal list
but is required to make 01 §12 steps 2–9 reproducible rather than ad hoc; it is traceable and
therefore included [AUTH: 01 §12; 03 §4].

### 10.1 `make preflight` semantics

```text
step 0  scripts/check_repo_invariants.py           01 §15, §18; 03 §9, §13
step 1  make format CHECK=1                        01 §33
step 2  make lint                                  01 §33
step 3  make typecheck                             01 §33
step 4  make unit                                  01 §22, §33
step 5  make integration                           01 §22, §33
step 6  make synthetic                             01 §22, §33
step 7  verify provenance-bound evidence records (§11)              02 §C6; 01 §16; 00 §34C.1
step 8  write artifacts/p0_pre/P0_PRE_READINESS.json                     02 §C6
```

Preflight stops at the first failing step, preserving the §22 order. Steps 1–6 are CPU-only, so
preflight is exactly what CI runs. Step 7 **never executes** backend-contract, gpu-smoke or the
benchmark, because those need hardware or an integrated backend that CI does not have
[AUTH: 01 §6.1, §49; 02 §C6].

Step 7 also **does not trust file content**. Each evidence record must be a run-manifest-bound
artifact carrying `RUN_ID`, `ENVIRONMENT_LOCK_SHA256` and the SHA256 of every artifact it
attests, and step 8 refuses any TRUE flag unless the recorded `ENVIRONMENT_LOCK_SHA256` equals
the current `manifests/environments/<env_id>.json`. A record without a valid run manifest is
`NON_EVIDENTIARY` and cannot raise a flag [AUTH: 01 §16; 02 §C6; 00 §34B.3].

---

## 11. P0-PRE readiness state

[AUTH: 02 §C6]

Declared at S00 in `artifacts/p0_pre/P0_PRE_READINESS.json`:

```json
{
  "BACKEND_INTEGRATED": false,
  "SUITE_SCOPE": "STATISTICAL_STACK_ONLY",
  "P0_PRE_READY": false,
  "computed_by": "DECLARED_AT_S00",
  "environment_lock_sha256": "TBD_REQUIRES_HARDWARE",
  "evidence": {
    "backend_contract":   {"status": "NOT_RUN(BACKEND_NOT_INTEGRATED)"},
    "synthetic_suite":    {"status": "NOT_RUN(EMPTY_AT_S00)"},
    "cache_assertions":   {"status": "NOT_RUN(EMPTY_AT_S00)"},
    "gpu_smoke":          {"status": "NOT_RUN(NO_GPU)"},
    "benchmark_1000_seq": {"status": "NOT_RUN"}
  }
}
```

Any evidence value other than `NOT_RUN(...)` must be a provenance-bound record
[AUTH: 01 §16; 02 §C6]:

```text
{"status": "PASS", "run_id": ..., "environment_lock_sha256": ...,
 "run_manifest": "manifests/runs/<RUN_ID>.json", "artifact_sha256": {...}}
```

A hand-written `{"status": "PASS"}` file is not evidence and cannot raise a flag.

### 11.1 Transition rule (implemented later, encoded now)

The flags may flip only when all five §C6 post-integration steps hold [AUTH: 02 §C6]:

```text
BACKEND_INTEGRATED = TRUE
SUITE_SCOPE        = INTEGRATED_PRODUCTION_STACK
P0_PRE_READY       = TRUE
```

requires all of:

```text
1. production backend contract tests            PASS      02 §C6
2. full synthetic suite rerun                   PASS      02 §C6; 00 §34B.2
3. cache assertions rerun                       PASS      02 §C6; 00 §34B.1 DRY-CACHE
4. H100 GPU smoke                               PASS      02 §C6; 01 §21
5. 1,000-sequence production benchmark          WRITTEN   02 §C6; 00 §0.2.4, §34C.1
and the 00 §35 / §34C.1 software-gate conjunction:
SCORER_ENGINE = VERIFIED; ANALYSIS_DRY_RUN = PASS; CROSSFIT_NEGATIVE_CONTROL = PASS;
CACHE_ASSERTIONS = PASS; P0_00_COMPUTE_BUDGET.json = WRITTEN
```

### 11.2 The anti-illusion property

`P0_PRE_READY` is computed as a conjunction that **contains** backend and hardware evidence, so
a green synthetic suite alone can never raise it [AUTH: 02 §C6]. Two further conditions make that
conjunction unforgeable: every input is provenance-bound (§11 schema), and step 8 refuses a TRUE
flag when the recorded `ENVIRONMENT_LOCK_SHA256` does not equal the current environment manifest
[AUTH: 01 §16; 02 §C6]. `make preflight` at S00 necessarily emits `false`; A4 asserts the initial
state, that no source path writes `true`, and that a hand-authored PASS file leaves
`P0_PRE_READY` at `false`.

---

## 12. Failure and status states

Only authority-required states. S00 introduces no new project-state vocabulary; the only S00
labels are the lane report reasons of D7.

| State set | Values | Authority | S00 scope |
|---|---|---|---|
| Run outcome | `SUCCESS`, `FAILED_IMPLEMENTATION`, `FAILED_ENVIRONMENT`, `FAILED_RESOURCE`, `FAILED_SCIENTIFIC_GATE`, `CANCELLED` | 01 §36 | Documented in CLAUDE.md; enum implemented in S01/S09. Never overwrite a failed run dir; a rerun gets a new `RUN_ID` [01 §36] |
| Evidentiary status | `NON_EVIDENTIARY` (result without a valid run manifest) | 01 §16 | Documented; enforced from S01 |
| Backend readiness | `BACKEND_INTEGRATED`, `SUITE_SCOPE`, `P0_PRE_READY` | 02 §C6 | **Written at S00** (§11) |
| P0-PRE software gate | `SCORER_ENGINE`, `ANALYSIS_DRY_RUN`, `CROSSFIT_NEGATIVE_CONTROL`, `CACHE_ASSERTIONS`, `P0_00_COMPUTE_BUDGET.json` | 00 §35, §34C.1; 02 §C3, §C4 | Fields reserved, all `NOT_RUN` at S00. DRY-G's production gate is the `02 §C4` family-wise max-statistic calibration at `N_FWER_CALIBRATION = 2_000`, artifact `P0_PRE_DRYG_FAMILYWISE_CALIBRATION.json`; the 200-trial bank is superseded for that purpose (R7) |
| Calendar | `SOFTWARE_SLIP`, `CALENDAR_COMPRESSED_MVRS`, `P1 = NOT_STARTED_CALENDAR` | 00 §34C.1; 00 §42 | Reserved; recorded when the checklist is executed |
| Deviation | `SPEC_DEVIATION`, `IMPLEMENTATION_REPAIR` | 00 §0; 01 §29 | Register file created at S00 |
| Scorer exception | `SCORER_EXCEPTION` | 00 §34A.5 | Register file created at S00 |
| Model gate | `MODEL_COMPATIBILITY_GATE` | 01 §8C | Reserved; S02B |
| Hardware unknown | `TBD_REQUIRES_HARDWARE` | 03 §8 | **Used at S00** (§5.7) |
| Review severity | `BLOCKER`, `MAJOR`, `MINOR`, `NOTE` | 01 §3.3; 03 §5, §6 | Used at S00 review |
| Architect response | `FIX`, `REJECT`, `DEFER` | 03 §10 | Used at S00 review |
| Reviewer verdict | `PASS`, `PASS_WITH_NONBLOCKING_NOTES`, `FAIL` | 01 §3.3 | Used at S00 review |
| Adjudication | `CLEARED`, `BLOCKED`, `INCOMPLETE` | 03 §11 | Used at S00 review |
| Stage acceptance | `ACCEPT`, `ACCEPT_WITH_NONBLOCKING_NOTES`, `REJECT` | 01 §5, §46 | Used at S00 acceptance |
| Scope / process | `UNJUSTIFIED_SCOPE`, `PROMPT_ONLY_REQUIREMENT`, `PLAN_TOO_LARGE`, `REVIEW_PROCESS_FAIL` | 03 §4, §7, §13, §14, §15 | Used at S00 review |

---

## 13. Structural invariants enforced from S00

These are the mechanisms that make the self-check properties structural rather than
aspirational. All are implemented in `scripts/check_repo_invariants.py`, run as `preflight`
step 0 and by CI.

| ID | Invariant | Mechanism | Authority |
|---|---|---|---|
| I1 | Exactly one scorer / cross-fit / fixed-FPR-estimator / cache path | `RESERVED_SINGLE_OWNER` API (`engine`, `crossfit`, `roc`, `cache`, `pooled`) may exist only under `src/scoring/`; **plus an import-boundary check** — no module under `src/analysis/` or `src/attacks/` may define fold construction, calibration-threshold estimation, or fixed-FPR ROC interpolation; they must import them from `src.scoring`. A second path fails unless a `SCORER_EXCEPTION` entry in `specs/deviations/SCORER_EXCEPTIONS.md` names it | 00 §34A.2(8), §34A.5; 02 §C1, §C2; D8; §13.1 |
| I2 | No notebook can become authoritative execution | no `.ipynb` under `src/`; no Makefile target and no file under `configs/` references a notebook | 01 §18 |
| I3 | Reviewer output cannot become production code | no importable `.py` under `reviews/` outside `scratch/`; packaging discovery restricted to `src/`; pre-commit hook blocks the violation | 03 §9 |
| I4 | Spec copies are the authority copies | SHA256 of each `specs/*.md` equals `specs/SPEC_HASHES.json`, **and** the four values are additionally asserted as literals in `tests/unit/test_s00_invariants.py`, so regenerating `SPEC_HASHES.json` alone cannot clear the gate | 01 §15 |
| I5 | Plan-length cap, every stage | every `stage_acceptance/*/01_PLAN.md` ≤ 1,200 lines, glob-based so the check exists unchanged from S01 onward; the glob must be non-empty | 03 §13; D10 |
| I6 | No fabricated hardware values | every hardware-derived field is either populated by S00-B or exactly `TBD_REQUIRES_HARDWARE` | 03 §8 |
| I7 | Readiness cannot be raised by the synthetic lane | `P0_PRE_READY` is written only by `preflight` step 8 as the §11.1 conjunction; no other source path writes it | 02 §C6 |
| I8 | Empty lane ≠ PASS | zero collected tests reports `NOT_RUN(<reason>)` | 02 §C6; 00 §34B.3; D7 |
| I9 | Protected branches — convenience layer only | pre-commit / pre-push block `main` and `experiment-frozen`, but `--no-verify` and an unset `core.hooksPath` bypass them, so the authority mechanism is branch/tag protection plus I15; preflight step 0 warns when `core.hooksPath != .githooks` | 01 §27; 03 §9 |
| I10 | CI cannot masquerade as hardware evidence | CI workflow invokes only `make preflight`; contains no GPU runner and no `backend-contract` invocation | 01 §6.1, §49; 02 §C6 |
| I11 | No result without provenance | `results/**` writers must emit the 01 §19 provenance keys — declared at S00, enforced by the S01 writer and its tests | 01 §16, §19 |
| I12 | Cache cannot survive a scoring, backend or environment change | cache key must include `SCORING_CODE_HASH` **and `ENVIRONMENT_LOCK_SHA256`** — declared at S00, enforced by the S03 negative tests including environment change → miss | 02 §C7; 00 §34A.4; 01 §12, §32 |
| I13 | Headline privacy numbers cannot come from a second estimator | every arm/view/headline consumer of `M_PRIMARY`, `D_L`, `D_R`, `R_priv`, `G_L`, `G_R` routes through the single `src/scoring` fixed-FPR API (§13.1) | 00 §34A.2(8); 02 §C1, §C2 |
| I14 | Material constants live in config, not source | the 01 §17 list — target FPR, Min-K fraction, alpha, k, DARE p, SVD rank, bootstrap replicates, seeds — resolves from `configs/**` and appears as a module-scope constant in no `src/**` file | 01 §17; 02 §C1; 03 §14 |
| I15 | Evidentiary execution refuses a dirty tree | before any run manifest is written, the launcher rejects **staged, unstaged and untracked** changes under production paths (`src/`, `configs/`, `specs/`, `pyproject.toml`, `uv.lock`, `Dockerfile`); a dirty tree refuses execution rather than merely being recorded as dirty | 01 §35(3), §16, §27, §36 |

I11, I12, I13 and I15 are *declarations with a named owning stage* (S01 / S03 / S09), not S00
code: implementing them would require the provenance core, the scorer and the launcher, which S00
must not build [AUTH: 01 §39; 03 §7]. I14's structural half — no 01 §17 constant at module scope
in `src/**` — is checkable without any of them and is checked from S00.

### 13.1 Canonical fixed-FPR estimator [AUTH: 02 §C1, §C2, §C5, §C8; 00 §34A.1–§34A.2]

```text
M_PRIMARY(V) = TPR_OOF/eval( V | FPR = 0.01 )      common-FPR TPR@1%FPR      02 §C1
```

`M_PRIMARY` is the primary privacy estimand for `D_L`, `D_R`, `R_priv`, `G_L`, `G_R` and endpoint
eligibility. The operational TPR/FPR pair from the calibration-only threshold is always reported
beside it and never substituted for it, because a differing operating point would make cross-view
gaps partly an operating-point artifact [AUTH: 02 §C1].

**Ownership.** Fixed-FPR ROC interpolation is duty 8 of the *one* cross-fitting controller and no
analysis-specific script may reproduce it [AUTH: 00 §34A.2]. It is exposed as one tested function
under `src/scoring/`, used by every view and arm [AUTH: 02 §C2]. `src/analysis/` **consumes** it
and may not reimplement it; the 01 §34 rule that statistics live in importable modules is
satisfied by importing this one, not by copying it.

**Prohibited**, recorded as a standing exclusion in `specs/deviations/SCORER_EXCEPTIONS.md`
[AUTH: 02 §C2, §C8]:

```text
individual-record cumulative ROC + "max TPR at repeated FPR"            PROHIBITED
```

**Required instead** [AUTH: 02 §C2]: tie-safe ROC constructed at unique score thresholds, all
observations sharing an exact score added simultaneously, `drop_intermediate = false`, linear
interpolation between valid neighbouring points, one deterministic documented convention.
Estimator tolerances are regenerated after the estimator is fixed rather than inherited because
they previously passed [AUTH: 02 §C5]. The P0 scaffold is a reference fixture and may not be
copied merely because a synthetic fixture passed [AUTH: 02 §C8].

---

## 14. S00 acceptance tests

All run on CPU, with no network, no model download and no H100 [AUTH: 01 §20, §39 S00; 03 §8].

| ID | Test | Asserts | Lane | Authority |
|---|---|---|---|---|
| A1 | `test_repo_structure` | every path in §3 exists; every path in §3.1 is absent | unit | 01 §11, §18, §32, §33 |
| A2 | `test_spec_integrity` | `specs/` holds the four current documents; each SHA256 matches `SPEC_HASHES.json` **and** the literal value asserted in the test file | unit | 01 §15; R1 |
| A3 | `test_plan_length` | the `stage_acceptance/*/01_PLAN.md` glob is non-empty and every match is ≤ 1,200 lines | unit | 03 §13; D10 |
| A4 | `test_readiness_initial_state` | the three §C6 flags exist with values `false / STATISTICAL_STACK_ONLY / false`; no source path can write `P0_PRE_READY=true` at S00; **forged-evidence case** — hand-authored `{"status":"PASS"}` records leave all three flags unchanged | unit | 02 §C6; 01 §16 |
| A5 | `test_command_surface` | every §10 target exists and is invocable; `synthetic`, `backend-contract`, `gpu-smoke` report `NOT_RUN(<reason>)`, not `PASS` | integration | 01 §33; 02 §C6 |
| A6 | `test_preflight_order` | preflight executes steps in §22 order, halts at first failure, emits readiness with `P0_PRE_READY=false`; **env-mismatch case** — evidence whose `ENVIRONMENT_LOCK_SHA256` differs from the current environment manifest cannot raise a flag | integration | 01 §22, §33, §16; 02 §C6 |
| A7 | `test_single_owner_paths` | reserved API confined to `src/scoring/`; an injected duplicate fails; a duplicate with a recorded `SCORER_EXCEPTION` passes; **an `src/analysis/` module defining its own fold selection or fixed-FPR ROC fails the import-boundary check** | unit | 00 §34A.2(8), §34A.5; 02 §C2 |
| A8 | `test_no_notebook_execution_path` | no `.ipynb` under `src/`; no Makefile/config target references a notebook | unit | 01 §18 |
| A9 | `test_reviewer_isolation` | no importable code under `reviews/` outside `scratch/`; packaging discovery limited to `src/` | unit | 03 §9 |
| A10 | `test_git_repo_controls` | repository is a real worktree with a resolvable HEAD; install script sets `core.hooksPath`; a simulated commit on `main` is blocked; a `.py` staged under `reviews/` outside `scratch/` is blocked; **`--no-verify` and unset-`hooksPath` bypasses are demonstrated**, and a reviewer-role write to a production path fails | integration | 01 §27; 03 §9 |
| A11 | `test_ci_workflow_contract` | CI invokes only `make preflight`; no GPU runner; no backend-contract invocation | unit | 01 §6.1, §49; 02 §C6 |
| A12 | `test_provenance_paths_reserved` | `manifests/{models,data,environments,runs}`, `artifacts/{p0_pre,runs,cache}`, `logs/`, `results/{p0,p1}` exist; the declared `RUN_ID` input list is exactly the eight 01 §15 inputs | unit | 01 §11, §13–§16 |
| A13 | `test_ai_stack_record` | `AI_ENGINEERING_STACK_S00.json` carries the Claude block (client version, exposed model name, UTC timestamp, permission mode, repository commit) **and the Codex block** (client/product, client version, exact model name, review date/time UTC, prompt SHA256, repository commit, review-output SHA256); any field the product does not expose is written as `UNAVAILABLE_NOT_EXPOSED`, never invented | unit | 01 §2.1, §4, §48 |
| A14 | `test_no_fabricated_hardware_values` | every hardware-derived field is populated by S00-B or is exactly `TBD_REQUIRES_HARDWARE` | unit | 03 §8 |
| A15 | `test_env_manifest_schema` | any present environment manifest carries every §5.6 field and a well-formed `ENVIRONMENT_LOCK_SHA256` | unit | 01 §12, §15, §16 |
| A16 | `test_canonical_fixed_fpr_estimator` | exactly one fixed-FPR ROC implementation exists, under `src/scoring/`; the prohibited "max TPR at repeated FPR" pattern is absent from `src/**`; `M_PRIMARY` is declared as common-FPR TPR@1%FPR | unit | 02 §C1, §C2, §C8; 00 §34A.2 |
| A17 | `test_material_constants_in_config` | none of target FPR, Min-K fraction, alpha, k, DARE p, SVD rank, bootstrap replicates or seeds appears as a module-scope constant in `src/**`; an injected `TARGET_FPR = 0.01` in `src/analysis/` fails the check | unit | 01 §17; 02 §C1; 03 §14 |
| A18 | `test_acceptance_bundle_completeness` | a bundle carries all twenty 01 §26 fields in their §8.3 slots, including resolved configs, the real-scientific-data-inspection statement and the requested verdict | unit | 01 §26, §45 |

### 14.1 S00-B acceptance (requires hardware; not part of the above)

`tests/gpu_smoke/test_cuda_torch_contract.py` and
`tests/gpu_smoke/test_env_capture_contract.py` accept §5.4 and §5.6. They are
`NOT_RUN(NO_GPU)` until the H100 image exists, and their outcome is the S00-B gate
[AUTH: 01 §12, §21; 03 §8].

---

## 15. Exact files expected to change

[AUTH: 01 §3.1]

**S00-A creates:**

```text
CLAUDE.md
README.md
Makefile
pyproject.toml
Dockerfile
uv.lock                                        (CPU/dev set; requires-python frozen)
.gitignore
.githooks/pre-commit
.githooks/pre-push
.github/workflows/ci.yml
specs/00_MEASUREMENT_SPEC_v1.9_FINAL_CLOSED.md
specs/01_EXECUTION_STACK_LOCK_v2.md
specs/02_PRE_EXECUTION_IMPLEMENTATION_CORRECTIONS.md
specs/03_REVIEW_GOVERNANCE_LOCK.md
specs/SPEC_HASHES.json
specs/deviations/SPEC_DEVIATIONS.md
specs/deviations/SCORER_EXCEPTIONS.md
configs/{models,data,training,attacks,recovery,p0,p1}/.gitkeep
src/__init__.py
src/{data,models,training,dp,merge,recovery,scoring,attacks,analysis,provenance,cli}/__init__.py
tests/{unit,integration,synthetic,golden,backend_contract,gpu_smoke}/__init__.py
tests/conftest.py
tests/unit/test_s00_invariants.py
tests/integration/test_s00_command_surface.py
manifests/{models,data,environments,runs}/.gitkeep
manifests/environments/AI_ENGINEERING_STACK_S00.json
results/{p0,p1}/.gitkeep
artifacts/{p0_pre,runs,cache}/.gitkeep
artifacts/p0_pre/P0_PRE_READINESS.json
reviews/S00/scratch/.gitkeep
stage_acceptance/S00/01_PLAN.md                (canonical stage-plan path, D10)
logs/.gitkeep
scripts/check_repo_invariants.py
scripts/capture_environment.sh
scripts/install_git_hooks.sh
```

**S00-B creates or modifies:**

```text
uv.lock                                        (relocked on the H100 image; authoritative)
pyproject.toml                                 (CUDA-coupled pins added)
manifests/environments/<env_id>.json           (incl. ENVIRONMENT_LOCK_SHA256)
Dockerfile                                     (pinned CUDA base image / digest)
```

**S00 does not touch:** anything under `src/*` beyond `__init__.py`, any config content, any
scientific module. Drive-by refactors are prohibited [AUTH: 01 §27, §41].

---

## 16. Interfaces introduced by S00

[AUTH: 01 §3.1]

```text
scripts/check_repo_invariants.py
    CLI:  python scripts/check_repo_invariants.py [--json]
    Exit: 0 = all invariants hold; 1 = at least one violation
    Out:  one line per violation: <INVARIANT_ID> <path> <reason>
    AUTH: 01 §15, §18; 03 §9, §13

scripts/capture_environment.sh
    CLI:  bash scripts/capture_environment.sh --out manifests/environments/
    Exit: 0 = every required field resolved; 1 = any field unresolved
    AUTH: 01 §12(2)-(9)

scripts/install_git_hooks.sh
    Effect: git config core.hooksPath .githooks
    AUTH:   01 §39 S00

artifacts/p0_pre/P0_PRE_READINESS.json     schema in §11
    AUTH:   02 §C6

manifests/environments/<env_id>.json       fields in §5.6
    AUTH:   01 §12, §15, §16

specs/SPEC_HASHES.json
    {"<filename>": {"sha256": "...", "role": "SCIENTIFIC|ENGINEERING|CLARIFICATION|REVIEW"}}
    AUTH:   01 §15
```

No Python interface is added under `src/` in S00. `src/cli/*` entry points begin at S01
[AUTH: 01 §39 S01].

---

## 17. Failure branches

[AUTH: 01 §3.1]

| Branch | Trigger | Action |
|---|---|---|
| F1 | H100 not yet provisioned | S00-B deferred; `S00_ENV_CAPTURE = TBD_REQUIRES_HARDWARE`; S00-A may clear blind review and CPU-lane S01–S04 development may proceed on the S00-A lock. **F1 is not an acceptance branch**: S00 cannot be accepted without S00-B evidence, and no evidentiary run may start, since no `ENVIRONMENT_LOCK_SHA256` means no valid `RUN_ID` [AUTH: 01 §15, §16, §46] |
| F2 | Dependency classes cannot be mutually resolved on the image | Record as `actual library incompatibility`, a permitted reason to amend the execution lock [AUTH: 01 §47]; do not substitute versions silently |
| F3 | `mypy` cannot resolve in the frozen env | Switch to `pyright`, the other authorized option; record the choice in the environment manifest [AUTH: 01 §33] |
| F4 | Dev-lane lock differs from H100-image lock | The H100-generated `uv.lock` is authoritative [AUTH: 01 §12(2)–(5)]; the dev lane is non-evidentiary |
| F5 | A tool default conflicts with a §11 path | Keep the §11 path; change the tool configuration [AUTH: 01 §11] |
| F6 | Plan exceeds 1,200 lines | `PLAN_TOO_LARGE`; reduce before review [AUTH: 03 §13] |
| F7 | Reviewer finds an untraceable component | `UNJUSTIFIED_SCOPE`; architect answers `FIX`/`REJECT`/`DEFER` with the required evidence [AUTH: 03 §7, §10] |
| F8 | Both blind reviewers miss a planted defect in a review benchmark | `REVIEW_PROCESS_FAIL`; the stage cannot clear [AUTH: 03 §15] |
| F9 | Unresolved BLOCKER after Round 2 | Stage `BLOCKED`; no Round 3 for non-BLOCKER findings [AUTH: 03 §11, §12] |
| F10 | S00 slips past 2026-08-27T23:59:59+01:00 together with the rest of P0-PRE | `SOFTWARE_SLIP = TRUE`, `CALENDAR_COMPRESSED_MVRS = ACTIVE`; no gate is weakened to recover the date [AUTH: 00 §34C.1] |
| F11 | Production tree dirty at launch (staged, unstaged or untracked) | I15 refuses execution before the run manifest is written; the operator commits or stashes; any rerun receives a new `RUN_ID` [AUTH: 01 §35(3), §16, §27, §36] |
| F12 | Repository is not a real Git worktree at review time | S00 implementation review cannot begin; the §8.1 precondition is unmet and 01 §26 items 4–5 cannot be supplied [AUTH: 03 §9; 01 §25, §26] |

---

## 18. Generated artifacts

[AUTH: 01 §3.1]

```text
repository skeleton                                  01 §11, §39 S00
specs/SPEC_HASHES.json                               01 §15
CLAUDE.md                                            01 §28
artifacts/p0_pre/P0_PRE_READINESS.json               02 §C6
manifests/environments/AI_ENGINEERING_STACK_S00.json 01 §2.1, §37, §48
manifests/environments/<env_id>.json      (S00-B)    01 §12, §15, §16
uv.lock + uv_lock_sha256                  (S00-B)    01 §12(5)
Docker image tag + digest record          (S00-B)    01 §12(8),(9)
test logs for every lane, including NOT_RUN lanes    01 §26(9)-(12)
reviews/S00/*                                        03 §2, §3, §10, §11
stage_acceptance/S00/ (01 §45 ten-file bundle)       01 §26, §45
```

---

## 19. Acceptance criteria for S00

Stage acceptance requires all of [AUTH: 01 §46]:

```text
all BLOCKER findings          CLOSED
all MAJOR findings            CLOSED or explicitly disproven with evidence
required tests                PASS            (A1-A18)
required negative controls    PASS            (A4 forged evidence; A5, A7, A9, A10, A16, A17
                                               injected-violation cases)
artifact hashes               present         (specs/SPEC_HASHES.json; bundle 05)
repository                    real Git worktree at a pinned commit       03 §9; §8.1
S00-B environment evidence    PRESENT         (ENVIRONMENT_LOCK_SHA256 resolved; F1 is not an
                                               acceptance branch)
bundle                        all 20 of 01 §26's fields present          01 §26; §8.3
scientific spec               unchanged
```

Review closure requires [AUTH: 03 §11]:

```text
S00_ADJUDICATION = CLEARED
```

Which additionally requires [AUTH: 03 §10, §11]:

```text
every unique finding answered exactly once with FIX / REJECT / DEFER
every REJECT carries an authority citation or technical evidence
every DEFER carries a ticket in reviews/S00/deferred.md
no BLOCKER deferred
```

Then the stage is tagged [AUTH: 01 §27].

---

## 20. Self-check attestation

[AUTH: 03 §4, §5, §7, §8, §13, §14; 00 §34A; 01 §16, §18; 02 §C6, §C7]

| Check | Result | Where satisfied |
|---|---|---|
| No component lacks an authority citation | PASS | §3, §5–§14 carry `[AUTH: …]` on every entry; path-only choices are disclosed in §4 |
| No task-prompt-only requirement became authority | PASS | §4.1 audit; every prompt item mapped to authority; none required exclusion |
| No hardware value fabricated | PASS | §5.7 register holds only genuinely H100/CUDA-dependent values; `requires-python` and the CPU/dev set are frozen in the S00-A lock (§5.2); no version number, throughput, batch size or VRAM figure appears in this plan; I6 + A14 |
| No duplicate future cross-fit / scoring path enabled | PASS | Single `src/scoring/` owner (§3); I1 API reservation **plus import-boundary check** + A7; §13.1 names the one estimator; exceptions only via a recorded `SCORER_EXCEPTION` [00 §34A.5] |
| Primary endpoint cannot silently change | PASS | §13.1 fixes `M_PRIMARY` = common-FPR TPR@1%FPR and prohibits the scaffold estimator [02 §C1, §C2, §C8]; I13 + A16 route every headline consumer through it; I14 + A17 keep target FPR and the other 01 §17 constants in `configs/**` |
| No notebook can become authoritative execution | PASS | `notebooks/` excluded (§3.1); I2 + A8 |
| No result can exist without provenance | PASS | §7 paths + 01 §15 `RUN_ID` input freeze + 01 §19 provenance keys; I11 (S01 owner) and I15 (dirty-tree refusal, S09 owner); `NON_EVIDENTIARY` defined in §12 |
| No cache valid after a scoring, backend or environment change | PASS | §7.1 key includes `SCORING_CODE_HASH` [02 §C7] **and `ENVIRONMENT_LOCK_SHA256`** (§5.6 composite over lock, image digest, CUDA runtime/driver, torch, Python, GPU); I12 + the S03 environment-change → miss test |
| Synthetic PASS cannot imply backend readiness | PASS | §9.1 separate lane, §9.2 empty≠PASS, §11.1 conjunction, §11.2 anti-illusion property; every evidence input is provenance-bound and env-matched, so a hand-written PASS cannot raise a flag; I7 + A4 forged-evidence case |
| Reviewer output cannot modify production code | PASS | §8.1 pinned-worktree precondition + read-only outside `scratch/`; branch/tag protection is the authority layer and hooks are demoted to convenience (I9); I3 + A9 + A10 |
| Plan ≤ 1,200 lines | PASS | Verified by `wc -l`; I5 globs `stage_acceptance/*/01_PLAN.md` (D10) so the check is standing from S01 onward; A3 |

### 20.1 Known open items carried into review

1. **R1–R7 reconciliations** (§1.2) are architect resolutions of authority-internal conflicts,
   all applying the higher-requirement rule [AUTH: 03 §11].
2. **F1 is live and is not an acceptance branch.** S00-B has not been executed, so every §5.7
   value is `TBD_REQUIRES_HARDWARE` and `ENVIRONMENT_LOCK_SHA256` is unresolved. CPU-lane work
   may proceed on the S00-A lock; S00 acceptance and any evidentiary run may not.
3. **The pinned-worktree precondition (§8.1) is unmet until the repository is initialised.**
   S00 implementation review cannot begin before then [AUTH: 03 §9; 01 §26(4)(5)].
4. **The 21–27 August 2026 P0-PRE window is non-compressible** [AUTH: 00 §34C.1]. S00 sits
   inside it. If review consumes the window, the correct outcome is `SOFTWARE_SLIP`, not a
   weakened gate.

---

S00_REVIEW_ROUND = 2_OF_2
S00_ARCHITECTURE_STATUS = READY_FOR_CLOSURE_VERIFICATION
