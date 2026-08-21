I'll start by locating the authority documents in the working tree.

  Read 1 file, listed 1 directory, ran 2 shell commands

Repo state noted: git rev-parse fails — no git repository, so no pinned worktree isolation is possible. Reading the review target now.

  Searched for 5 patterns, read 1 file, ran 16 shell commands

Read-only checks complete. No production file was read outside the five authority inputs; no Codex material exists in the tree and none was read.

CLAUDE_BLIND_REVIEW_VERDICT = FAIL

Mandatory-check results

┌─────────────────────────────────────────┬──────────────────────────────────────────────────────┐
│                  Check                  │                        Result                        │
├─────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ plan ≤ 1,200 lines                      │ PASS — wc -l STAGE_PLAN.md = 1044                    │
├─────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ authority traceability                  │ PASS with C-02/C-06 exceptions below                 │
├─────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ UNJUSTIFIED_SCOPE                       │ PASS — no untraceable component with material burden │
├─────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ PROMPT_ONLY_REQUIREMENT                 │ none found                                           │
├─────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ reproducibility / provenance            │ FAIL — C-01, C-03                                    │
├─────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ backend-readiness separation            │ FAIL — C-04                                          │
├─────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ cache invalidation architecture         │ FAIL — C-01                                          │
├─────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ one future scorer / cross-fit path      │ FAIL — C-02                                          │
├─────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ primary endpoint cannot silently change │ FAIL — C-02, C-06                                    │
├─────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ no-result-peeking path                  │ FAIL — C-05                                          │
├─────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ reviewer read-only isolation            │ PASS with C-09, N-01                                 │
├─────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ clean-machine / H100 reproducibility    │ FAIL — C-03                                          │
└─────────────────────────────────────────┴──────────────────────────────────────────────────────┘

---

MAJOR

C-01 — Environment identity collapsed to uv_lock_sha256; absent from both the cache key and RUN_ID

Authority: 01 §12 (environment lock = pyproject.toml + uv.lock + Dockerfile, steps 7–9 record CUDA driver/runtime and image digest); 01 §15; 01 §32 ("a serving-engine change … requires a new environment hash"); 01 §36; 00 §34A.4.
Location: STAGE_PLAN.md:425 ("the environment lock SHA256 consumed by RUN_ID is uv_lock_sha256"); STAGE_PLAN.md:549; STAGE_PLAN.md:560 (cache key list); STAGE_PLAN.md:809 (I12).
Counterexample (two distinct failures, one root cause):
1. RUN_ID collision. cuda_driver is a node property captured into <env_id>.json but is in neither env_id nor the RUN_ID environment input. Same commit, spec, config, model revision, data manifest, seed and uv.lock, executed on RunPod node A (driver 550.x) and node B (driver 560.x) → both compute RUN_ID = R. artifacts/runs/R/ collides, violating 01 §36 ("Never overwrite a failed run directory. Reruns receive a new RUN_ID"), and run identity no longer separates two runtimes that 01 §30 requires be treated as nondeterminism sources.
2. Stale cache across environments. The §7.1 key is {model/artifact hash, record-set hash, tokenizer hash, precision, max_seq_len, SCORING_CODE_HASH}. SCORING_CODE_HASH covers source files + resolved scoring config only. After an F2 amendment (STAGE_PLAN.md:949) produces a new uv.lock and a new torch/CUDA build, rescoring the same records yields a cache hit: forward outputs computed under E1 are served into a run whose manifest declares E2. DRY-CACHE's "unchanged code/config → hit + bitwise-identical table" passes, so no test detects it. The result table then attributes scores to an environment that never produced them.

Required correction: define environment lock SHA256 as a hash over uv_lock_sha256 + docker_image_digest + cuda_driver + cuda_runtime (i.e. env_id), not uv_lock_sha256 alone; add env_id to the §7.1 content-addressed cache key and to I12; require each cache entry to record its producing env_id and RUN_ID; add an "environment change → miss" negative test to the S03 DRY-CACHE set.

---

C-02 — No single-owner reservation for the fixed-FPR ROC estimator; 02 §C1 and §C2 are never carried into the plan

Authority: 00 §34A.2 item 8 ("fixed-FPR ROC interpolation" is a duty of the one cross-fitting controller; "No analysis-specific script may reproduce this logic independently"); 02 §C1 (M_PRIMARY(V) = common-FPR TPR@1%FPR; a differing operating point "would make cross-view gaps partly an operating-point difference"); 02 §C2 ("expose the estimator as one tested function used by every view/arm"; the scaffold's "max TPR at repeated FPR" must not enter production); 02 §C8; 03 §4.
Location: STAGE_PLAN.md:798 (I1), :270 (D8), :829 (A7), :1022 (self-check claiming PASS); STAGE_PLAN.md:117 and :120 (§2.2 S03/S04 rows); STAGE_PLAN.md:40-41 (§1.1 lists 02 as C6+C7 only). grep '02 §C[0-9]' STAGE_PLAN.md returns only C3, C6, C7 — C1, C2, C5, C8 appear nowhere.
Counterexample: RESERVED_SINGLE_OWNER = {engine, crossfit, cache, pooled}. S03 ships src/scoring/crossfit.py with fixed-FPR interpolation for the five-fold canary path; S04, following 01 §34 ("statistics live in importable modules", src/analysis/metrics.py), ships an independent tpr_at_fpr() used for the headline M_PRIMARY, D_L, D_R, R_priv tables. python scripts/check_repo_invariants.py exits 0 — no reserved basename is duplicated and nothing left src/scoring/. A7 passes. No SCORER_EXCEPTION is required. Two fixed-FPR estimators coexist; because 02 §C2's prohibition on the scaffold routine was never carried into the plan, the S04 copy may legitimately be the ported scaffold implementation that 02 §C2 forbids, and cross-view M_PRIMARY gaps become partly estimator artifacts — precisely the corruption 02 §C1 forbids.
Required correction: add 02 §C1, §C2, §C5, §C8 to §1.1; extend D8/I1's RESERVED_SINGLE_OWNER with the fixed-FPR ROC estimator basename, confined to src/scoring/; state in §2.2 that 00 §34A.2(8) ROC interpolation is owned by the S03 cross-fit controller and may not be reimplemented under src/analysis/; add the corresponding A7 injected-duplicate case; record 02 §C2's named prohibition in specs/deviations/SCORER_EXCEPTIONS.md as a standing exclusion.

---

C-03 — S00-A ships no lock and no pins; CPU tooling is deferred as TBD_REQUIRES_HARDWARE

Authority: 01 §33 ("Exact versions are frozen in the environment lock"); 01 §46 ("repo is reproducible from frozen environment"); 03 §8 (TBD_REQUIRES_HARDWARE is for values requiring real hardware).
Location: STAGE_PLAN.md:336 ("Dependency classes are declared by name without version pins, marked PROPOSED_UNPINNED. requires-python is left unset in S00-A"); STAGE_PLAN.md:443 (ruff/mypy/pytest versions, owner S00-B); STAGE_PLAN.md:894 (uv.lock created only in S00-B); STAGE_PLAN.md:693-698 + I10 (CI runs make preflight steps 1–6); STAGE_PLAN.md:441 (F1 permits S01–S04 before S00-B).
Counterexample: on a clean machine, git clone <repo> && uv sync && make preflight on 2026-08-21 resolves tool set X; the identical commit on 2026-10-01 resolves X′. make lint / make typecheck (01 §33 lint = PASS, typecheck = PASS) can flip PASS→FAIL against a byte-identical tree, and no committed lock exists to reproduce or bisect either run. This is not a short window: F1 explicitly allows S01–S04 to proceed before S00-B, so every S01–S04 stage acceptance under 01 §46 rests on an unpinned resolution. The S00-A Dockerfile is also unbuildable as specified — STAGE_PLAN.md:329 says it "installs strictly from the frozen uv.lock", which S00-A does not produce. None of ruff, mypy, pytest, NumPy, SciPy or pandas requires an H100, so 03 §8 does not license deferring them.
Required correction: S00-A commits a CPU/dev lock (uv.lock with a CPU index, or a locked dev extra) pinning at minimum requires-python, ruff, mypy and pytest, with its SHA256 recorded; S00-B's H100-generated lock remains authoritative for scientific execution (F4 unchanged). Remove ruff/mypy/pytest and the non-GPU library rows from the §5.7 TBD_REQUIRES_HARDWARE register.

---

C-04 — preflight step 7 consumes unauthenticated recorded state, so the readiness conjunction is forgeable

Authority: 02 §C6 (flags flip only after the five steps are run); 00 §34B.3 ("does not mean merely that a script file exists"); 01 §16.
Location: STAGE_PLAN.md:693 (step 7 "read recorded state for backend-contract, gpu-smoke, benchmark"); :694 (step 8 writes the file); :698; :757-761 (§11.2 anti-illusion claim); :804 (I7); :826 (A4).
Counterexample: an implementer hand-writes artifacts/p0_pre/backend_contract_result.json = {"status":"PASS"}, gpu_smoke_result.json = {"status":"PASS"} and a P0_00_COMPUTE_BUDGET.json. make preflight step 7 reads them, step 8 evaluates the §11.1 conjunction and writes P0_PRE_READY=true. No backend was integrated and no H100 ran. I7 is satisfied (only step 8 wrote the flag) and A4 is satisfied (A4 asserts only the S00 initial state and that no source path writes true at S00). The plan specifies no provenance binding on step 7's inputs, so §11.2's anti-illusion property rests on file content any process may author.
Required correction: require every step-7 input to carry the 01 §16 binding (RUN_ID, environment lock SHA256, artifact SHA256); require step 8 to verify the recorded env_id equals the current manifests/environments/<env_id>.json and to refuse any TRUE flag otherwise; extend A4/A6 with a forged-evidence negative case.

---

C-05 — Stage-acceptance bundle drops 01 §26 item 19, the only mechanical no-result-peeking attestation

Authority: 01 §26 ("Every stage submitted for final adjudication contains exactly: … 19. explicit statement of whether real scientific data were inspected"); 01 §29; 03 §11 (higher-severity rule the plan itself invokes in R2).
Location: STAGE_PLAN.md:88-92 (R4 reconciles only items 16–17); STAGE_PLAN.md:226-230 and :580, :975 (layout cites 01 §26, §45 while implementing only §45's file list); STAGE_PLAN.md:823-837 (A1–A15 contain no check). grep -i 'inspect' STAGE_PLAN.md returns only line 135 (a non-goal row) and line 503 (a CLAUDE.md pointer).
Counterexample: at S11 — where 01 §39 states "After this point, result-peeking rules are active" — a bundle containing 00_INDEX.md … 10_REPRODUCE.md is complete per §3, §18 and §19, passes A1 (test_repo_structure checks only §3 paths), and carries no statement of whether real scientific data were inspected. §26 items 8 (resolved configs) and 20 (requested verdict) are dropped by the same reconciliation. R4 resolves this conflict by "pick one" while R2 resolves the same class of conflict by "apply the higher requirement"; the inconsistency is what removes the artifact.
Required correction: apply R2's higher-requirement rule to R4 — map all 20 §26 items onto the §45 slots (item 19 → a dedicated 11_DATA_INSPECTION_STATEMENT.md or a required named field in 00_INDEX.md; item 8 → resolved configs beside 03_TEST_COMMANDS.txt; item 20 → 00_INDEX.md) and add an acceptance test asserting the full §26 item set is present in every bundle.

---

C-06 — The 01 §17 material-constants rule has no owner, no path and no invariant; the primary endpoint's operating point may live in source

Authority: 01 §17 ("No material experimental constant may live only in source code. Examples: … target FPR … Min-K fraction"; "Each CLI invocation must be reconstructable from a saved resolved config"); 00 §16; 02 §C1; 03 §14 (protects specifically against a silent change to the primary privacy endpoint).
Location: 01 §17 appears in the plan only at STAGE_PLAN.md:289, :500, :541, :550 — a CLAUDE.md pointer and the resolved_config_sha256 field. No I1–I12 invariant, no A1–A15 test, no §2.2 owning-stage row, and no §7 provenance row binds any material constant to a config file. STAGE_PLAN.md:871 creates configs/**/.gitkeep only, and §15 states "S00 does not touch … any config content".
Counterexample: S04 defines TARGET_FPR = 0.01 as a module constant in src/analysis/metrics.py. make preflight exits 0, check_repo_invariants.py exits 0, A1–A15 pass. At S11 a one-line edit to 0.05 changes the primary endpoint: resolved_config_sha256 (STAGE_PLAN.md:550) is identical across the two runs, so the saved resolved config no longer reconstructs the invocation as 01 §17 requires, and config SHA256 — a RUN_ID input — does not move. Supporting weakness: I4/A2 compare specs/*.md against specs/SPEC_HASHES.json, which is itself regenerable in the same commit, so the spec copies detect corruption but not a deliberate endpoint edit.
Required correction: add an invariant and acceptance test asserting that the 01 §17 constant list (target FPR, Min-K fraction, alpha, k, DARE p, SVD rank, bootstrap replicates, seeds) resolves from configs/** and appears in no src/** module scope; declare the owning config file and stage for each in §2.2/§7; assert the four spec SHA256 values as literals in tests/unit/test_s00_invariants.py so regenerating SPEC_HASHES.json alone cannot clear the gate.

---

MINOR

C-07 — STAGE_PLAN.md has no location in the repository

Authority: 03 §13. Location: STAGE_PLAN.md:802 (I5), :825 (A3); absent from the §3 tree (:145-243) and from the §15 S00-A creation list (:857-891).
Counterexample: make preflight step 0 runs check_repo_invariants.py, whose I5 must read STAGE_PLAN.md; the plan never places that file in the tree, so the path is undefined (repo root? stage_acceptance/<stage>/01_PLAN.md?) and I5 either errors or passes vacuously on a missing file. A1 (test_repo_structure) does not require it to exist. Later stage plans (S01…) have no declared location at all, so 03 §13's PLAN_TOO_LARGE detection is unmechanized from S01 onward, contradicting the §20 claim that "I5 + A3 make it a standing CI check".
Required correction: add the canonical stage-plan path to §3 and §15 and make I5 glob every stage plan.

C-08 — Codex provenance record required at S00 is absent

Authority: 01 §48 ("At S00, record the exact exposed Claude and Codex versions/models where available"); 01 §4 (seven fields to record at the first Codex review).
Location: STAGE_PLAN.md:835 (A13 covers Claude only); STAGE_PLAN.md:200, :532 (AI_ENGINEERING_STACK_S00.json). grep -i codex STAGE_PLAN.md returns only review-file paths and R2 — no provenance artifact.
Counterexample: R2 (STAGE_PLAN.md:77) makes S00 the first Codex review. After it completes, no file in §3, §15 or §18 holds Codex client/product, client version, exact model name, review date/time UTC, prompt SHA256, repository commit, or review-output SHA256. A13 passes while half the 01 §48 S00 record is missing — the "test passes while implementation is wrong" pattern of 01 §43 Q2.
Required correction: extend AI_ENGINEERING_STACK_S00.json and A13 to the 01 §4 Codex field set, with absent values recorded as absent.

C-09 — Git-hook enforcement is opt-in and bypassable; I9 is not a structural invariant

Authority: 03 §9 ("enforced operationally, not only requested"); 01 §27.
Location: STAGE_PLAN.md:806 (I9 listed under "Structural invariants enforced from S00"), :594 (§8.1 item 4), :834 (A10), :918 (install_git_hooks.sh → git config core.hooksPath .githooks).
Counterexample: git clone <repo> && cd repo && git checkout main && echo x >> src/scoring/engine.py && git add -A && git commit -m x succeeds — core.hooksPath is per-clone local config that install_git_hooks.sh was never run to set. On a clone where it was set, git commit --no-verify and git push --no-verify bypass both hooks. check_repo_invariants.py defines no invariant asserting core.hooksPath, and I10/A11 keep CI to make preflight, so nothing in the repository detects either bypass. A10 tests the hook only after installation.
Required correction: add an invariant asserting core.hooksPath == .githooks so an unhooked clone fails preflight step 0; state server-side branch protection as the actual 01 §27 mechanism and demote I9 to a convenience guard; extend A10 to the --no-verify and unset-hooksPath cases.

---

NOTE

N-01 — Reviewer isolation is unenforced in fact for S00's own review. 03 §9. git rev-parse --show-toplevel in the review tree returns fatal: not a git repository; there is no worktree, no pinned commit, no .githooks/, and no reviews/S00/scratch/. The §8.1 enforcement chain depends entirely on artifacts S00-A creates, and the plan contains no bootstrap provision for the first review it mandates under R2.

N-02 — Unreconciled authority conflict on the DRY-G calibration bank. 00 §42 requires a "200-trial dry-run tolerance calibration artifact frozen"; 02 §C4 sets N_FWER_CALIBRATION = 2_000 and replaces marginal gates with a family-wise max-statistic gate. §1.2 declares R1–R6 but not this one, and §12's reserved DRY-G fields do not say which trial count governs.